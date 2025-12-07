import os
import argparse
from dataclasses import dataclass
from typing import Dict, List, Any
import csv
import numpy as np
import json

import torch
from torch.utils.data import Dataset, DataLoader

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
TEST_JSONL = "data_test/test.jsonl"

MODEL_DIR = "/autodl-tmp/models/qwen/Qwen2-VL-7B-Instruct"
OUTPUT_DIR = "experiments/dataqwen2vl_kline_lora"

MAX_LENGTH = 1024          # 文本最大长度，太大会占显存
BATCH_SIZE = 1             # per-device batch size
GRAD_ACCUM = 4             # 累积梯度，相当于总 batch = BATCH_SIZE * GRAD_ACCUM
NUM_EPOCHS = 1
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

    # 1) full_text: 带有真实 assistant 回答的完整对话文本
    full_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # 2) prompt_text: 只有 system + user，去掉最后 assistant 的“提问部分”
    #    再加一个 generation prompt，让模型在这里开始生成回答
    prompt_messages = messages[:-1]  # 假设最后一条就是 assistant 的答案
    prompt_text = processor.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,  # 模型从这里开始续写
    )

    # 3) 视觉信息
    image_inputs, video_inputs = process_vision_info(messages)

    # 4) 用完整对话文本 full_text 做 encoder，使模型能看到“带答案的目标序列”
    model_inputs = processor(
        text=[full_text],
        images=image_inputs,
        videos=video_inputs,
        padding="max_length",
        max_length=MAX_LENGTH,
        truncation=True,
        return_tensors="pt",
    )
    # model_inputs: {"input_ids": [1, L], "attention_mask": [1, L], "pixel_values": ...}

    input_ids = model_inputs["input_ids"]          # (1, L)
    labels = input_ids.clone()                    # 先复制一份

    tokenizer = processor.tokenizer
    pad_token_id = tokenizer.pad_token_id

    # 5) 先把 padding 全部 mask 掉
    labels[input_ids == pad_token_id] = -100

    # 6) 计算 prompt_text 的长度（多少个 token）
    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
    )["input_ids"]
    prompt_len = len(prompt_ids)

    # 避免被截断导致越界
    seq_len = labels.shape[1]
    prompt_len = min(prompt_len, seq_len)

    # 7) 把 prompt 部分的 label 也设成 -100，只在回答上算 loss
    labels[:, :prompt_len] = -100

    model_inputs["labels"] = labels

    # 保持 batch 维度在外面： (1, L), (1, n_img, C, H, W) ...
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

    # model = Qwen2VLForConditionalGeneration.from_pretrained(
    #     MODEL_NAME,
    #     torch_dtype="auto",
    #     device_map=DEVICE_MAP,
    #     quantization_config=quant_config,
    # )

    # processor = AutoProcessor.from_pretrained(MODEL_NAME)
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_DIR,
        torch_dtype="auto",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_DIR)


    


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
    把 Trainer 的 log 写到 CSV。
    固定列：step, epoch, loss, eval_loss, learning_rate, grad_norm
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.fieldnames = [
            "step",
            "epoch",
            "loss",
            "eval_loss",
            "learning_rate",
            "grad_norm",
        ]
        self.file = None
        self.writer = None
        self.initialized = False

    def _init_writer(self):
        new_file = not os.path.exists(self.csv_path)
        self.file = open(self.csv_path, "a", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        if new_file:
            self.writer.writeheader()
        self.initialized = True

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        if not self.initialized:
            self._init_writer()

        logs = dict(logs)

        row = {k: None for k in self.fieldnames}
        row["step"] = state.global_step
        row["epoch"] = state.epoch

        # 这些 key 如果在 logs 里，就填进去
        for k in ["loss", "eval_loss", "learning_rate", "grad_norm"]:
            if k in logs:
                row[k] = logs[k]

        self.writer.writerow(row)
        self.file.flush()

    def on_train_end(self, args, state, control, **kwargs):
        if self.file is not None:
            self.file.close()


# ----------------- 保存验证集输出结果留待评测 -----------------

class EvalPromptDataset(Dataset):
    """
    简单的包装：从 HF 的 raw_dataset 里按索引取出 messages。
    """

    def __init__(self, raw_dataset):
        self.raw_dataset = raw_dataset

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        example = self.raw_dataset[idx]
        return {
            "id": idx,
            "messages": example["messages"],
        }

@dataclass
class EvalCollator:
    """
    把一批 raw samples:
      [{"id":..., "messages":[...]}...]
    变成：
      - model_inputs: 可以直接喂给 model.generate(**model_inputs)
      - ids: 每条样本的索引
      - labels_text: 每条样本的 ground-truth 文本
      - prompt_lens: 每条样本的 prompt 长度（后面用来截掉 prompt，只看新生成的部分）
    """
    processor: Any   # AutoProcessor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_prompt_texts: List[str] = []
        batch_image_inputs: List[Any] = []
        batch_video_inputs: List[Any] = []
        ids: List[int] = []
        labels_text: List[str] = []

        for feat in features:
            idx = feat["id"]
            messages = feat["messages"]

            if len(messages) < 2:
                # 防御：没法构成正常 (system, user, assistant) 的，直接跳过
                # 这里简单返回空；上层可以决定怎么处理
                continue

            # prompt = system + user，不包含最后一条 assistant 答案
            prompt_messages = messages[:-1]
            prompt_messages = add_file_prefix_for_images(prompt_messages)

            # 文本 prompt
            prompt_text = self.processor.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,  # 在 assistant 开始处继续生成
            )

            # 视觉信息
            image_inputs, video_inputs = process_vision_info(prompt_messages)

            batch_prompt_texts.append(prompt_text)
            batch_image_inputs.append(image_inputs)   # 每个样本自己的 image 列表
            ids.append(idx)

            # 取最后一条 assistant 的文本作为 ground-truth
            gt_text_parts = []
            last_msg = messages[-1]
            for item in last_msg.get("content", []):
                if isinstance(item, dict) and item.get("type") == "text":
                    t = item.get("text", "")
                    if t:
                        gt_text_parts.append(t)
            gt_text = "\n".join(gt_text_parts)
            labels_text.append(gt_text)

        # 如果这一 batch 都被跳过了（极端情况），返回空
        if not batch_prompt_texts:
            return {
                "model_inputs": None,
                "ids": [],
                "labels_text": [],
                "prompt_lens": [],
            }

        # 一次性用 processor 处理一个 batch
        proc_inputs = self.processor(
            text=batch_prompt_texts,
            images=batch_image_inputs,
            return_tensors="pt",
            padding=True,  # 按 batch 最长样本做 padding
        )

        # prompt 长度：直接用 attention_mask 里 1 的个数
        attention_mask = proc_inputs["attention_mask"]
        prompt_lens = attention_mask.sum(dim=1)  # (batch_size,)

        return {
            "model_inputs": proc_inputs,   # 里面有 input_ids / attention_mask / pixel_values / image_grid_thw ...
            "ids": ids,
            "labels_text": labels_text,
            "prompt_lens": prompt_lens,    # tensor
        }


def save_eval_predictions_streaming(
    model,
    processor,
    raw_dataset,         # ⚠️ HF Dataset，需包含 "messages"
    output_path: str,
    max_new_tokens: int = 128,
    eval_num: int = 20,
    batch_size: int = 16,
    num_workers: int = 16,
):
    """
    对 raw_dataset 批量做推理，仅用 prompt（system+user）作为输入，
    不把真实的 assistant 回答喂给 generate。

    - raw_dataset[i]["messages"] = [system, user, assistant(with JSON)]
    - 只把 system+user 作为生成条件
    - 生成结果和 label 写入 jsonl
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    device = next(model.parameters()).device
    tokenizer = processor.tokenizer
    model.eval()

    # 只取前 eval_num 个样本（防止一次太多）
    if eval_num is not None:
        raw_dataset = raw_dataset.select(range(min(len(raw_dataset), eval_num)))

    # 包装成 torch Dataset + DataLoader
    eval_dataset = EvalPromptDataset(raw_dataset)
    collator = EvalCollator(processor=processor)

    dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,      # 可以先用 0 或 4，看表现
        pin_memory=True,
    )

    print(f"Saving eval predictions (prompt-only, batched) to {output_path} ...")

    total_saved = 0

    with open(output_path, "w", encoding="utf-8") as f, torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            model_inputs = batch["model_inputs"]
            ids = batch["ids"]
            labels_text = batch["labels_text"]
            prompt_lens = batch["prompt_lens"]  # (batch_size,)

            # 有可能这一批被全部跳过（极端情况）
            if model_inputs is None or len(ids) == 0:
                continue

            # 把张量搬到 GPU
            for k, v in model_inputs.items():
                if isinstance(v, torch.Tensor):
                    model_inputs[k] = v.to(device, non_blocking=True)
            prompt_lens = prompt_lens.to(device)

            # 调 generate：一次处理一个 batch
            gen_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
            )    # (batch_size, total_len)

            gen_ids = gen_ids.cpu()
            prompt_lens_cpu = prompt_lens.cpu()

            batch_size_eff = gen_ids.shape[0]

            for i in range(batch_size_eff):
                seq = gen_ids[i]
                prompt_len = int(prompt_lens_cpu[i].item())
                answer_ids = seq[prompt_len:]   # 截掉 prompt，只留新增 token

                pred_text = tokenizer.decode(
                    answer_ids, skip_special_tokens=True
                )

                rec = {
                    "id": int(ids[i]),
                    "prediction": pred_text,
                    "label": labels_text[i],
                }
                json.dump(rec, f, ensure_ascii=False)
                f.write("\n")
                total_saved += 1

            # 释放当前 batch 的显存
            del gen_ids
            torch.cuda.empty_cache()

            if (batch_idx + 1) % 10 == 0:
                print(f"  processed {total_saved} samples ...")

    print(f"Done. Total {total_saved} samples saved to {output_path}.")

    print("Done.")


# ----------------- main -----------------


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default=TRAIN_JSONL)
    parser.add_argument("--val_file", type=str, default=VAL_JSONL)
    parser.add_argument("--test_file", type=str, default=TEST_JSONL)
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
        "test": args.test_file
    }
    raw_datasets = load_dataset("json", data_files=data_files)

    # 2. 加载模型 + processor
    model, processor = load_model_and_processor()

    # 3. 包装成自定义 Dataset
    train_dataset = QwenKlineDataset(raw_datasets["train"], processor)
    eval_dataset = QwenKlineDataset(raw_datasets["validation"], processor)
    test_dataset = QwenKlineDataset(raw_datasets["test"], processor)

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
        logging_steps=500,
        save_steps=2000,
        eval_strategy="steps",
        eval_steps=2000,
        save_total_limit=2,
        bf16=True if torch.cuda.is_available() else False,
        fp16=False,
        report_to=[],  # 不接 wandb / tensorboard 的话就留空
        dataloader_num_workers=16,
        dataloader_pin_memory=True,
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

    # 先尝试从最近的 checkpoint 断点续训，如果失败（比如没有 checkpoint），就从头开始
    try:
        print("尝试从最近的 checkpoint 断点重连训练……")
        trainer.train(resume_from_checkpoint=True)
    except Exception as e:
        print(f"断点重连失败（{e}），改为从头开始训练。")
        trainer.train()
    
    torch.cuda.empty_cache() # 训练完清缓存

    
    # 7. 训练结束后，在验证集上生成，并把结果写到 jsonl


    save_path = os.path.join(args.output_dir, "eval_predictions.jsonl")
    eval_num = 1000  # 后续可以选择预测大小
    
    save_eval_predictions_streaming(
        model=model,
        processor=processor,
        raw_dataset=raw_datasets["test"],  # 注意拿的是 .dataset (Dataset的raw_dataset)
        output_path=save_path,
        max_new_tokens=64,
        eval_num=eval_num         
    )

    # 8. 保存 LoRA 权重和 processor
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("训练完成，模型已保存到", args.output_dir)


if __name__ == "__main__":
    main()