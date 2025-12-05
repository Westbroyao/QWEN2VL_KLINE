#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
eval.py

用来评估 Qwen2-VL K线微调的预测效果。

"""

import argparse
import json
import os
from typing import List, Dict, Any, Tuple
import re


ALLOWED_LABELS = {"up", "down", "flat"}


def load_records(path: str) -> List[Dict[str, Any]]:
    """
    既支持 jsonl（一行一个 json），也支持 json 数组（[ {...}, {...}, ... ]）.
    """
    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)

        if first_char == "[":
            # json 数组
            data = json.load(f)
            if isinstance(data, list):
                return data
            else:
                raise ValueError("JSON 文件顶层不是数组。")
        else:
            # jsonl
            records = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
            return records


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    prediction/label 字段本身是个字符串，这里尝试解析出里面的 JSON 对象。

    优先级：
    1）先正常用 json.loads 解析（处理格式完全正确的情况）
    2）失败时，截取第一个 '{' 到最后一个 '}' 再试一次
    3）如果还是失败，用正则粗暴地提取 "label" 和 "reason" 字段
       （容忍后半截 JSON 缺失，比如 prediction 没有收尾的引号/大括号）
    """
    text = text.strip()
    if not text:
        return {}

    # 1. 先尝试直接解析
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 2. 尝试从中间截取 JSON 子串（完整闭合的情况）
    try:
        if "{" in text and "}" in text:
            start = text.index("{")
            end = text.rindex("}") + 1
            sub = text[start:end]
            obj = json.loads(sub)
            if isinstance(obj, dict):
                return obj
    except Exception:
        pass

    # 3. 容错模式：JSON 已经烂掉（比如 prediction 少了结尾 " }）
    #    用正则尽量把 label / reason 挖出来
    result: Dict[str, Any] = {}

    # 提取 label（up/flat/down）
    m_label = re.search(r'"label"\s*:\s*"(up|down|flat)"',
                        text, re.IGNORECASE)
    if m_label:
        result["label"] = m_label.group(1).lower()

    # 提取 reason：从 "reason": " 之后一路取到行尾，再把尾部的 " 和 } 去掉
    m_reason = re.search(r'"reason"\s*:\s*"(.*)', text)
    if m_reason:
        reason = m_reason.group(1).strip()
        # 去掉末尾可能残留的引号/大括号/空格
        reason = reason.rstrip('"} ')
        result["reason"] = reason

    return result


def eval_predictions(
    records: List[Dict[str, Any]],
    show_examples: int = 10,
) -> None:
    """
    对 prediction / label 里的 "label" 字段做分类评估。
    """

    total = len(records)
    valid = 0
    correct = 0

    # label -> index
    idx_map = {"up": 0, "flat": 1, "down": 2}
    idx_to_label = ["up", "flat", "down"]
    # 3x3 混淆矩阵: row=true, col=pred
    cm = [[0, 0, 0] for _ in range(3)]

    error_examples: List[Tuple[int, str, str, str, str]] = []

    for rec in records:
        rec_id = rec.get("id", None)

        pred_text = rec.get("prediction", "")
        label_text = rec.get("label", "")

        pred_obj = extract_json_from_text(pred_text)
        label_obj = extract_json_from_text(label_text)

        pred_label = str(pred_obj.get("label", "")).strip().lower()
        true_label = str(label_obj.get("label", "")).strip().lower()

        if pred_label not in ALLOWED_LABELS or true_label not in ALLOWED_LABELS:
            # 这条样本无法用于分类评估
            continue

        valid += 1
        if pred_label == true_label:
            correct += 1

        pi = idx_map[pred_label]
        ti = idx_map[true_label]
        cm[ti][pi] += 1

        if pred_label != true_label and len(error_examples) < show_examples:
            pred_reason = str(pred_obj.get("reason", "")).strip()
            true_reason = str(label_obj.get("reason", "")).strip()
            error_examples.append(
                (rec_id, true_label, pred_label, true_reason, pred_reason)
            )

    print("=" * 80)
    print(f"总样本数 total records: {total}")
    print(f"可用于评估的样本数 valid records: {valid}")
    if valid == 0:
        print("没有有效样本（prediction/label 解析不到合法的 label 字段）")
        return

    acc = correct / valid
    print(f"方向预测准确率 accuracy (on valid) = {acc:.4f}")
    print()

    # 打印混淆矩阵
    print("混淆矩阵 Confusion Matrix (rows=true, cols=pred):")
    header = "            " + "  ".join(f"{lbl:>8}" for lbl in idx_to_label)
    print(header)
    for i, row in enumerate(cm):
        row_total = sum(row)
        row_str = "  ".join(f"{n:8d}" for n in row)
        print(f"{idx_to_label[i]:>8}  {row_str}   | sum={row_total}")
    print()

    # === 计算每类的 Precision / Recall / F1 ===
    print("每类指标 Per-class metrics:")
    print(f"{'label':>8}  {'P':>8}  {'R':>8}  {'F1':>8}  {'support':>8}")
    for k, label in enumerate(idx_to_label):
        tp = cm[k][k]
        fp = sum(cm[i][k] for i in range(3)) - tp
        fn = sum(cm[k][j] for j in range(3)) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        support   = sum(cm[k])  # 该类作为真值的样本数

        print(f"{label:>8}  {precision:8.4f}  {recall:8.4f}  {f1:8.4f}  {support:8d}")
    print()

    # 打印一些错例
    if error_examples:
        print(f"部分错例 (最多展示 {show_examples} 条)：")
        for rec_id, true_label, pred_label, true_reason, pred_reason in error_examples:
            print("-" * 80)
            print(f"id = {rec_id}")
            print(f"true label = {true_label}, pred label = {pred_label}")
            print(f"[GT  reason] {true_reason}")
            print(f"[Pred reason] {pred_reason}")
    else:
        print("没有发现预测错误的样本。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        type=str,
        help="eval_predictions 文件路径（json 或 jsonl）",
    )
    parser.add_argument(
        "--show-examples",
        type=int,
        default=10,
        help="最多展示多少条错例",
    )
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print("文件不存在：", args.path)
        return

    records = load_records(args.path)
    eval_predictions(records, show_examples=args.show_examples)


if __name__ == "__main__":
    main()