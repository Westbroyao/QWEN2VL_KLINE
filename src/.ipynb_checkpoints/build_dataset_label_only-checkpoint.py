import os
import json
import numpy as np
from typing import List, Dict, Any

# ======== 配置区域，根据你自己的路径改一下 ========

NPZ_PATH = "data_proc/windows_30_5_with_labels.npz"   # 含 X / y / labels 的 npz
IMAGE_DIR = "data_images/kline_windows"                     # 存放K线图的文件夹
OUT_TRAIN_JSONL = "data_train/train.jsonl"
OUT_VAL_JSONL = "data_train/val.jsonl"

TRAIN_RATIO = 0.8       # 训练/验证划分比例
RANDOM_SEED = 42        # 保证可复现
DATA_NUMBER = 3000       # 先拿小样本测试，后续提升样本数

# 生成的图片文件名格式（和你画图脚本保持一致）
# 之前画图脚本里是：window_{i:05d}{label_part}.png，其中 label_part 形如 "_up"
IMG_NAME_TEMPLATE = "window_{idx:05d}_{label}.png"


# ======== 工具函数 ========

def load_npz(npz_path: str):
    """读取npz，返回标签（字符串）和时间索引等。"""
    data = np.load(npz_path, allow_pickle=True)
    files = set(data.files)

    # X 和 y 虽然这里用不到，但很多时候你后面可能会扩展
    X = data["X"]
    y = data["y"] if "y" in files else None
    time_index = data["time_index"] if "time_index" in files else None

    if "labels_str" in files:
        labels_str = data["labels_str"]
    elif "labels_int" in files:
        # 如果只有整数标签，就转成字符串
        labels_int = data["labels_int"]
        labels_str = np.empty_like(labels_int, dtype=object)
        labels_str[labels_int == 1] = "up"
        labels_str[labels_int == 0] = "flat"
        labels_str[labels_int == -1] = "down"
    else:
        raise ValueError("npz 中没有 labels_str 或 labels_int，无法构造标签。")

    return X, y, time_index, labels_str


def build_single_sample(idx: int,
                        label: str,
                        image_dir: str) -> Dict[str, Any]:
    """
    构造一条用于Qwen2-VL微调的样本（多模态对话格式）。
    这里只输出 label，不输出理由；后面你可以自己扩展 reason 字段。
    """
    # 根据索引和标签构造图片文件名
    img_name = IMG_NAME_TEMPLATE.format(idx=idx, label=label)
    img_path = os.path.join(image_dir, img_name)

    if not os.path.exists(img_path):
        # 如果严格按模板找不到，可以选择抛错或跳过
        raise FileNotFoundError(f"找不到图片文件: {img_path}")

    # system prompt：说明任务
    system_prompt = (
        "你是一名量化分析师，擅长分析加密货币K线图。"
        "现在给你的是BTCUSDT过去30个交易日的日K线图。"
        "请判断接下来5个交易日的总体价格走势，相比当前价格是上涨(up)、"
        "下跌(down)，还是大致震荡(flat)。"
        "只需要输出一个JSON，其中包含字段\"label\"，"
        "label ∈ {\"up\",\"flat\",\"down\"}。"
    )

    # user 内容：图像 + 一段简短说明
    user_text = (
        "下面的图片是BTCUSDT过去30个交易日的日K线图。"
        "请根据图形判断未来5日的总体价格方向，并在JSON中给出label。"
    )

    # assistant 内容：监督信号，模型要学着输出这个JSON字符串
    assistant_json = json.dumps({"label": label}, ensure_ascii=False)

    sample = {
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": user_text}
                ]
            },
            {
                "role": "assistant",
                "content": assistant_json
            }
        ]
    }
    return sample


def build_dataset(npz_path: str,
                  image_dir: str,
                  train_ratio: float = 0.8,
                  seed: int = 42,
                  data_number: int = None):
    """从npz和图片目录构造train/val两个列表。"""
    X, y, time_index, labels_str = load_npz(npz_path)
    n_samples = min(len(labels_str), data_number)
    indices = np.arange(n_samples)

    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    n_train = int(n_samples * train_ratio)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_samples: List[Dict[str, Any]] = []
    val_samples: List[Dict[str, Any]] = []

    # 构造训练集
    for idx in train_idx:
        label = labels_str[idx]
        sample = build_single_sample(idx=int(idx),
                                     label=str(label),
                                     image_dir=image_dir)
        train_samples.append(sample)

    # 构造验证集
    for idx in val_idx:
        label = labels_str[idx]
        sample = build_single_sample(idx=int(idx),
                                     label=str(label),
                                     image_dir=image_dir)
        val_samples.append(sample)

    return train_samples, val_samples


def save_jsonl(samples: List[Dict[str, Any]], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            line = json.dumps(s, ensure_ascii=False)
            f.write(line + "\n")
    print(f"保存 {len(samples)} 条样本到 {out_path}")


# ======== main入口 ========

def main():
    train_samples, val_samples = build_dataset(
        NPZ_PATH,
        IMAGE_DIR,
        train_ratio=TRAIN_RATIO,
        seed=RANDOM_SEED,
        data_number=DATA_NUMBER
    )

    save_jsonl(train_samples, OUT_TRAIN_JSONL)
    save_jsonl(val_samples, OUT_VAL_JSONL)


if __name__ == "__main__":
    main()