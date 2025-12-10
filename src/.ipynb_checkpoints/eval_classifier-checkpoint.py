# eval_classifier.py
import os
import argparse

import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


# 标签映射要和训练时保持一致
LABEL2ID = {"up": 0, "down": 1, "flat": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def load_probs_tensor(probs_path: str):
    """
    从 test_probs.pt 中读取数据。
    期望格式为 [N, 4]:
      [:,0] = p(label=0)
      [:,1] = p(label=1)
      [:,2] = p(label=2)
      [:,3] = y_true_id (0/1/2)
    """
    if not os.path.exists(probs_path):
        raise FileNotFoundError(f"Cannot find probs tensor at {probs_path}")

    tensor = torch.load(probs_path, map_location="cpu")
    if tensor.ndim != 2 or tensor.shape[1] != 4:
        raise ValueError(
            f"Expected probs tensor shape [N, 4], got {tuple(tensor.shape)}"
        )

    probs = tensor[:, :3].numpy()            # [N,3]
    y_true = tensor[:, 3].long().numpy()     # [N]

    return probs, y_true


def analyze_classification(probs: np.ndarray, y_true: np.ndarray):
    """
    根据概率和真实标签，计算分类指标并打印。
    """
    # 预测标签 = 概率最大的那个
    y_pred = probs.argmax(axis=1)

    labels = [LABEL2ID["up"], LABEL2ID["down"], LABEL2ID["flat"]]
    target_names = [ID2LABEL[i] for i in labels]

    print("=== 基本信息 ===")
    print(f"样本数: {len(y_true)}")
    acc = (y_pred == y_true).mean()
    print(f"整体 Accuracy: {acc:.4f}")
    print()

    print("=== 每类 precision / recall / f1 ===")
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    print(report)

    print("=== 混淆矩阵 ===")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # 行: 真实标签, 列: 预测标签
    print("行: true (up, down, flat)   列: pred (up, down, flat)")
    print(cm)
    print()

    print("=== 标签分布对比 ===")
    # 真实标签分布
    unique_true, counts_true = np.unique(y_true, return_counts=True)
    print("真实标签分布:")
    for lab_id, cnt in zip(unique_true, counts_true):
        name = ID2LABEL.get(int(lab_id), str(lab_id))
        print(f"  {name:>5s}: {cnt:6d} ({cnt / len(y_true):.2%})")

    # 预测标签分布
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    print("预测标签分布:")
    for lab_id, cnt in zip(unique_pred, counts_pred):
        name = ID2LABEL.get(int(lab_id), str(lab_id))
        print(f"  {name:>5s}: {cnt:6d} ({cnt / len(y_true):.2%})")

    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        type=str,
        help="eval_predictions 文件路径（json 或 jsonl）",
    )
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print("文件不存在：", args.path)
        return

    probs_path = args.path
    print(f"[Info] Loading probs tensor from {probs_path}")

    probs, y_true = load_probs_tensor(probs_path)
    analyze_classification(probs, y_true)


if __name__ == "__main__":
    main()
