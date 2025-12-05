#!/usr/bin/env bash
set -e

# 定位到仓库根目录
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=${PYTHON:-python}

# 默认的评估文件路径（可以按你的真实路径改）
DEFAULT_EVAL_FILE="$ROOT_DIR/experiments/qwen2vl_kline_lora/eval_predictions.jsonl"

# 如果脚本带了第一个参数，就用参数；否则用默认路径
EVAL_FILE="${1:-$DEFAULT_EVAL_FILE}"

echo "Using eval file: $EVAL_FILE"
echo

# 运行 eval.py
$PYTHON "$ROOT_DIR/src/eval.py" "$EVAL_FILE" --show-examples 10 2>&1 | tee results/eval.log

cp experiments/qwen2vl_kline_lora/train_eval_log.csv result/train_eval_log.csv