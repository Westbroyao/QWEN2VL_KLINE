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
    TrainerCallback
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
    - 把消息中的本地图片路径补成 file:// 绝对路径；
    - 过滤掉 image 字段为 None 或不是字符串的图片项；
    - 顺手把所有条目里 value 为 None 的 "image"/"text" 键删掉。
    """
    new_messages = []

    for msg in messages:
        contents = msg.get("content", [])
        if not isinstance(contents, list):
            new_messages.append(msg)
            continue

        clean_content = []
        for item in contents:
            if not isinstance(item, dict):
                clean_content.append(item)
                continue

            item_type = item.get("type")

            # 1) 处理图片项
            if item_type == "image":
                path = item.get("image", None)

                # 非法图片项（image 不是非空字符串）直接丢掉
                if not isinstance(path, str) or path.strip() == "":
                    continue

                # 补全 file:// 前缀
                if not path.startswith("file://"):
                    abs_path = os.path.abspath(path)
                    item = {**item, "image": "file://" + abs_path}

            # 2) 统一清理 None 字段：image=None / text=None 都删掉
            cleaned_item = {}
            for k, v in item.items():
                if k in ("image", "text") and v is None:
                    continue
                cleaned_item[k] = v

            clean_content.append(cleaned_item)

        new_messages.append({**msg, "content": clean_content})

    return new_messages


def build_training_example(
    example: Dict[str, Any],
    processor: AutoProcessor,
) -> Dict[str, Any]:
    messages = example["messages"]
    messages = add_file_prefix_for_images(messages)

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    # 注意：这里不做 v[0]，保持 batch 维度=1
    model_inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding="max_length",
        max_length=MAX_LENGTH,
        truncation=True,
        return_tensors="pt",
    )

    # labels 跟 input_ids 同形状：[1, seq_len]
    model_inputs["labels"] = model_inputs["input_ids"].clone()
    
    # 直接返回整个 dict，所有 key 都保留：
    # input_ids, attention_mask, pixel_values, image_grid_thw, ...
    return model_inputs


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
    把若干个样本（每个样本里已经是 batch_size=1 的张量）
    沿着第 0 维拼接成真正的 batch。
    """

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch: Dict[str, Any] = {}
        first = features[0]

        for key, value in first.items():
            if isinstance(value, torch.Tensor):
                # 每个 value 是 [1, ...]，cat 后变成 [batch_size, ...]
                batch[key] = torch.cat([f[key] for f in features], dim=0)
            else:
                # 非张量（一般用不到），直接收集成列表
                batch[key] = [f[key] for f in features]

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

# ----------------- 训练和验证误差 -----------------


class CSVLoggingCallback(TrainerCallback):
    """
    把 Trainer 的 log（train loss / eval loss 等）写入一个 CSV 文件。

    - on_log 会被 Trainer 定期调用（由 logging_steps/eval_steps 控制）
    - logs 里会包含 "loss"（训练）、"eval_loss"（验证）等字段
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.initialized = False
        self.fieldnames = None
        self.file = None
        self.writer = None

    def _init_writer(self, logs: dict):
        # 增加 step / epoch 两个字段
        base_keys = ["step", "epoch"]
        other_keys = [k for k in logs.keys() if k not in base_keys]
        self.fieldnames = base_keys + other_keys

        new_file = not os.path.exists(self.csv_path)
        self.file = open(self.csv_path, "a", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)

        if new_file:
            self.writer.writeheader()

        self.initialized = True

    def on_log(self, args, state, control, logs=None, **kwargs):
        # 每次 Trainer 打 log 时都会进这里（train/eval 都会）
        if logs is None:
            return

        logs = dict(logs)
        # 补充 step / epoch 信息
        logs["step"] = state.global_step
        logs["epoch"] = state.epoch

        if not self.initialized:
            self._init_writer(logs)

        # 只保留我们定义的字段，避免字段顺序乱掉
        row = {k: logs.get(k, None) for k in self.fieldnames}
        self.writer.writerow(row)
        self.file.flush()

    def on_train_end(self, args, state, control, **kwargs):
        if self.file is not None:
            self.file.close()




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
        eval_steps=10,
        save_total_limit=2,
        bf16=True if torch.cuda.is_available() else False,
        fp16=False,
        report_to=[],  # 不接 wandb / tensorboard 的话就留空
    )

    # 5. Trainer
    log_csv_path = os.path.join(training_args.output_dir, "train_eval_log.csv")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[CSVLoggingCallback(log_csv_path)],
    )

    # 6. 开始训练
    trainer.train()

    # 7. 保存 LoRA 权重和 processor
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("训练完成，模型已保存到", args.output_dir)


if __name__ == "__main__":
    main()