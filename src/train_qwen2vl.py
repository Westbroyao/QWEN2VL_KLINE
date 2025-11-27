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
from qwen_vl_utils import process_vision_info


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
    messages = add_file_prefix_for_images(messages)

    # 文本：使用 chat template，把多轮对话拼成一个字符串
    # 不加 generation_prompt，因为我们是监督学习，输入里已经含有 assistant 的答案
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # 图像：用 qwen-vl-utils 根据 messages 解析出 PIL 图像
    image_inputs, video_inputs = process_vision_info(messages)

    # 让 processor 统一处理 text + images，得到 input_ids / pixel_values 等
    model_inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding="max_length",
        max_length=MAX_LENGTH,
        truncation=True,
        return_tensors="pt",
    )

    # batch_size=1，这里直接取第0个
    input_ids = model_inputs["input_ids"][0]
    attention_mask = model_inputs["attention_mask"][0]
    pixel_values = model_inputs["pixel_values"][0]  # (C, H, W)

    # 简单做法：labels = input_ids
    # 更高级可以只在 assistant 段计算 loss，这里先不区分
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
        evaluation_strategy="steps",
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