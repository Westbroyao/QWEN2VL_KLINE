import os
import argparse
from dataclasses import dataclass
from typing import Dict, List, Any

import torch
from torch.utils.data import Dataset

from datasets import load_dataset
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from PIL import Image


# ----------------- 一些超参数，可以按需改 -----------------

MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"

TRAIN_JSONL = "data_train/train.jsonl"
VAL_JSONL = "data_train/val.jsonl"

OUTPUT_DIR = "experiments/dataqwen2vl_kline_lora"

MAX_LENGTH = 1024          # 文本最大长度，太大会占显存
BATCH_SIZE = 1             # per-device batch size
GRAD_ACCUM = 4             # 累积梯度，相当于总 batch = BATCH_SIZE * GRAD_ACCUM
NUM_EPOCHS = 3
LR = 1e-4
WARMUP_RATIO = 0.03

USE_4BIT = True            # QLoRA: 4bit 量化
DEVICE_MAP = "auto"


# ----------------- 自定义数据集 & 预处理 -----------------


def add_file_prefix_for_images(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Qwen 官方支持本地路径：形如 'file:///abs/path/to/img.png'
    你在 JSON 里目前是相对路径，如 'kline_windows/window_00000_up.png'，
    这里统一加上 'file://+' 绝对路径。
    """
    new_messages = []
    for msg in messages:
        if msg["role"] != "user":
            new_messages.append(msg)
            continue
        new_content = []
        for item in msg["content"]:
            if item.get("type") == "image":
                path = item["image"]
                # 如果还不是 file:// 开头，则转成绝对路径
                if not path.startswith("file://"):
                    abs_path = os.path.abspath(path)
                    item = {
                        **item,
                        "image": "file://" + abs_path
                    }
            new_content.append(item)
        new_messages.append({**msg, "content": new_content})
    return new_messages

def extract_pil_images(messages: List[Dict[str, Any]]) -> List[Image.Image]:
    """
    从 messages 里提取所有 image 项，读取为 PIL.Image。
    只处理本地文件，不支持 http/https。
    """
    images: List[Image.Image] = []
    for msg in messages:
        contents = msg.get("content", [])
        if not isinstance(contents, list):
            continue

        for item in contents:
            if isinstance(item, dict) and item.get("type") == "image":
                path = item.get("image", None)
                if not isinstance(path, str) or path.strip() == "":
                    continue   # 忽略非法项

                # 去掉 file:// 前缀
                if path.startswith("file://"):
                    path = path[len("file://"):]

                # 读取图片
                img = Image.open(path).convert("RGB")
                images.append(img)
    return images


def build_training_example(
    example: Dict[str, Any],
    processor: AutoProcessor,
) -> Dict[str, Any]:
    """
    把一行 JSONL（含 messages）变成模型可以吃的张量：
    - input_ids, attention_mask
    - pixel_values（图像特征）
    - labels（默认= input_ids）
    """
    messages = example["messages"]

    # 1) 先把 image 路径补成 file:// 绝对路径
    messages = add_file_prefix_for_images(messages)

    # 2) 从 messages 里抽出所有图片，读成 PIL.Image
    pil_images = extract_pil_images(messages)   # List[Image]

    # 3) 文本：用 chat template
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # 4) 让 processor 同时处理 text + images
    if len(pil_images) == 0:
        # 没图也行，只是纯文本
        model_inputs = processor(
            text=[text],
            padding="max_length",
            max_length=MAX_LENGTH,
            truncation=True,
            return_tensors="pt",
        )
    else:
        model_inputs = processor(
            text=[text],
            images=pil_images,
            padding="max_length",
            max_length=MAX_LENGTH,
            truncation=True,
            return_tensors="pt",
        )

    # batch_size=1，这里直接取第0个
    input_ids = model_inputs["input_ids"][0]
    attention_mask = model_inputs["attention_mask"][0]

    # Qwen2-VL 的视觉特征字段名通常是 pixel_values
    pixel_values = model_inputs["pixel_values"][0]

    # 简单做法：labels = input_ids
    labels = input_ids.clone()

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "labels": labels,
    }



class QwenKlineDataset(Dataset):
    """把 HuggingFace datasets 包起来，预处理成张量"""

    def __init__(self, hf_split, processor):
        self.dataset = hf_split
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        return build_training_example(example, self.processor)


# ----------------- Data collator -----------------


@dataclass
class DataCollatorForQwenVL:
    """
    Trainer 用的数据整理函数：把 list[dict] 拼成 batch。
    每个样本里已经是张量了，这里简单 stack 一下。
    """

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = torch.stack([f["input_ids"] for f in features])
        attention_mask = torch.stack([f["attention_mask"] for f in features])
        pixel_values = torch.stack([f["pixel_values"] for f in features])
        labels = torch.stack([f["labels"] for f in features])

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels,
        }
        return batch


# ----------------- 模型加载：Qwen2-VL + QLoRA -----------------


def load_model_and_processor():
    # 4bit 量化配置（QLoRA）
    quant_config = None
    if USE_4BIT:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map=DEVICE_MAP,
        quantization_config=quant_config,
    )

    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    # LoRA 配置：只训练部分线性层参数
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, processor


# ----------------- main -----------------


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default=TRAIN_JSONL)
    parser.add_argument("--val_file", type=str, default=VAL_JSONL)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--grad_accum", type=int, default=GRAD_ACCUM)
    parser.add_argument("--learning_rate", type=float, default=LR)
    parser.add_argument("--warmup_ratio", type=float, default=WARMUP_RATIO)
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. 读 JSONL 到 datasets
    data_files = {
        "train": args.train_file,
        "validation": args.val_file,
    }
    raw_datasets = load_dataset("json", data_files=data_files)

    # 2. 加载模型 + processor
    model, processor = load_model_and_processor()

    # 3. 包装成自定义 Dataset
    train_dataset = QwenKlineDataset(raw_datasets["train"], processor)
    eval_dataset = QwenKlineDataset(raw_datasets["validation"], processor)

    data_collator = DataCollatorForQwenVL()

    # 4. 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=10,
        save_steps=500,
        eval_strategy="steps",
        eval_steps=500,
        save_total_limit=2,
        bf16=True if torch.cuda.is_available() else False,
        fp16=False,
        report_to=[],  # 不接 wandb / tensorboard 的话就留空
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # 6. 开始训练
    trainer.train()

    # 7. 保存 LoRA 权重和 processor
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("训练完成，模型已保存到", args.output_dir)


if __name__ == "__main__":
    main()