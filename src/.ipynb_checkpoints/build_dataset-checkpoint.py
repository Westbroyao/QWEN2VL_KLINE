import os
import json
import numpy as np
from typing import List, Dict, Any

# ======== 配置区域，根据你自己的路径改一下 ========

NPZ_PATH = "data_proc/windows_30_5_multi_with_labels_train_val_resampling.npz"   # 含 X / y / labels 的 npz
IMAGE_DIR = "data_images/kline_windows"                     # 存放K线图的文件夹
OUT_TRAIN_JSONL = "data_train/train.jsonl"
OUT_VAL_JSONL = "data_train/val.jsonl"

TRAIN_RATIO = 0.9       # 训练/验证划分比例
RANDOM_SEED = 42        # 保证可复现
DATA_NUMBER = 300000       # 先拿小样本测试

# 生成的图片文件名格式（和你画图脚本保持一致）
# 之前画图脚本里是：window_{i:05d}{label_part}.png，其中 label_part 形如 "_up"
IMG_NAME_TEMPLATE = "window_{idx:05d}_{label}.png"


# ======== 工具函数 ========

def load_npz(npz_path: str):
    """读取npz，返回X、y、时间索引和字符串标签。"""
    data = np.load(npz_path, allow_pickle=True)
    files = set(data.files)

    X = data["X"]                          # (n_samples, 30, 5)
    y = data["y"] if "y" in files else None
    time_index = data["time_index"] if "time_index" in files else None

    if "labels_str" in files:
        labels_str = data["labels_str"]
    elif "labels_int" in files:
        labels_int = data["labels_int"]
        labels_str = np.empty_like(labels_int, dtype=object)
        labels_str[labels_int == 1] = "up"
        labels_str[labels_int == 0] = "flat"
        labels_str[labels_int == -1] = "down"
    else:
        raise ValueError("npz 中没有 labels_str 或 labels_int，无法构造标签。")

    return X, y, time_index, labels_str


def generate_reason(label: str, x_window: np.ndarray) -> str:
    """
    根据标签 + 这30天窗口的K线数据，生成稍微聪明一点的reason。
    x_window: shape (30, 5) = [open, high, low, close, volume]
    """

    close = x_window[:, 3].astype(float)
    vol = x_window[:, 4].astype(float)
    n = len(close)

    # 防止奇怪数据
    if n < 2 or np.any(close <= 0):
        # 回退到简单模板
        if label == "up":
            return "近期价格整体偏强，因此判断未来5个交易日上涨的概率较大。"
        elif label == "down":
            return "近期价格走势偏弱，因此判断未来5个交易日下跌风险较高。"
        else:
            return "近期价格缺乏明确趋势，因此判断未来5个交易日大概率维持震荡。"

    # 30日整体收益
    ret_30 = close[-1] / close[0] - 1.0

    # 最近5日收益（如果不足5根，就用后半段）
    k_recent = min(5, n - 1)
    ret_recent = close[-1] / close[-k_recent - 1] - 1.0

    # 简单日收益率 & 波动率
    daily_ret = np.diff(close) / close[:-1]
    vol_30 = float(np.std(daily_ret))  # 不年化，只做相对比较

    # 成交量变化：最近5天 vs 之前20天
    if n > 10:
        recent_vol_mean = float(np.mean(vol[-5:]))
        past_vol_mean = float(np.mean(vol[-min(20, n - 5):-5]))
        vol_ratio = recent_vol_mean / past_vol_mean if past_vol_mean > 0 else 1.0
    else:
        vol_ratio = 1.0

    # 文案片段
    # 趋势描述
    if ret_30 > 0.15:
        trend_text = "过去30个交易日整体呈现明显的上涨趋势"
    elif ret_30 > 0.03:
        trend_text = "过去30个交易日价格缓慢抬升"
    elif ret_30 < -0.15:
        trend_text = "过去30个交易日整体呈现较为明显的下跌趋势"
    elif ret_30 < -0.03:
        trend_text = "过去30个交易日价格温和走弱"
    else:
        trend_text = "过去30个交易日价格整体围绕区间震荡"

    # 近期变化
    if ret_recent > 0.05:
        recent_text = "最近一周出现了较为明显的反弹"
    elif ret_recent > 0.01:
        recent_text = "最近一周价格略有回升"
    elif ret_recent < -0.05:
        recent_text = "最近一周出现了较为明显的回落"
    elif ret_recent < -0.01:
        recent_text = "最近一周价格略有走弱"
    else:
        recent_text = "最近一周价格变化不大"

    # 波动率描述（BTC 波动本身就大，这里阈值略宽松）
    if vol_30 > 0.06:
        vol_text = "整体波动幅度较大"
    elif vol_30 > 0.03:
        vol_text = "整体波动处于中等水平"
    else:
        vol_text = "整体波动相对温和"

    # 成交量描述
    if vol_ratio > 1.3:
        volmsg = "成交量相较此前阶段有明显放大，说明市场参与度提升"
    elif vol_ratio < 0.8:
        volmsg = "成交量较此前阶段有所萎缩，说明交易情绪略显谨慎"
    else:
        volmsg = "成交量与此前阶段大致持平，市场情绪相对平稳"

    # 最后一句根据 label 决策方向
    if label == "up":
        tail = "在此背景下，多头力量相对占优，因此判断未来5个交易日上涨的概率较大。"
    elif label == "down":
        tail = "在此背景下，空头压力偏强，因此判断未来5个交易日继续走弱的风险较高。"
    else:  # flat
        tail = "在此背景下，多空力量大致平衡，因此判断未来5个交易日大概率延续区间震荡。"

    reason = f"{trend_text}，{recent_text}，{vol_text}，{volmsg}，{tail}"
    return reason


def build_single_sample(idx: int,
                        label: str,
                        image_dir: str,
                        x_window: np.ndarray) -> Dict[str, Any]:
    ...
    img_name = IMG_NAME_TEMPLATE.format(idx=idx, label=label)
    img_path = os.path.join(image_dir, img_name)

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"找不到图片文件: {img_path}")

    
    # system prompt：说明任务
    system_prompt = (
        "你是一名量化分析师，擅长分析中国A股K线图。\n"
        "现在给你的是中国A股过去90个交易日的日K线图。\n"
        "你的任务是判断接下来15个交易日的总体价格走势："
        "相比当前价格是上涨(up)、下跌(down)，还是大致震荡(flat)，"
        "并给出简要理由。\n"
        "请只输出一个JSON，字段仅包括\"label\"。"
        "其中 label ∈ {\"up\",\"flat\",\"down\"}，"
    )

    user_text = (
        "下面的图片是中国A股过去90个交易日的日K线图。\n"
        "请根据图形判断未来15日的总体价格方向"
        "最终以JSON形式返回（只包含label一个字段）。"
    )

    reason = generate_reason(label, x_window)
    assistant_json = json.dumps(
        {"label": label},
        ensure_ascii=False
    )

    sample = {
        "messages": [
            {
                "role": "system",
                # 🔧 改成 list[{"type":"text"}]
                "content": [
                    {"type": "text", "text": system_prompt}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text",  "text": user_text}
                ]
            },
            {
                "role": "assistant",
                # 🔧 也改成 list[{"type":"text"}]
                "content": [
                    {"type": "text", "text": assistant_json}
                ]
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
        label = str(labels_str[idx])
        x_window = X[idx]            # (30, 5)
        sample = build_single_sample(
            idx=int(idx),
            label=label,
            image_dir=image_dir,
            x_window=x_window,
        )
        train_samples.append(sample)

    # 构造验证集
    for idx in val_idx:
        label = str(labels_str[idx])
        x_window = X[idx]
        sample = build_single_sample(
            idx=int(idx),
            label=label,
            image_dir=image_dir,
            x_window=x_window,
        )
        val_samples.append(sample)

    return train_samples, val_samples


def save_jsonl(samples: List[Dict[str, Any]], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            line = json.dumps(s, ensure_ascii=False)
            f.write(line + "\n")
    print(f"保存 {len(samples)} 条样本到 {out_path}")


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