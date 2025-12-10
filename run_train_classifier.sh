#!/usr/bin/env bash
set -euo pipefail

########################
# 基本配置（按需修改）
########################

# 指定使用哪块GPU（如果只有一块卡，这行可以不改）
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# 训练数据 & 输出目录
TRAIN_FILE="data_train/train.jsonl"
VAL_FILE="data_train/val.jsonl"
OUTPUT_DIR="experiments/qwen2vl_kline_lora"

# 训练超参数
NUM_EPOCHS=1
BATCH_SIZE=1          # 对应 train_qwen2vl.py 里的 per_device_train_batch_size
GRAD_ACCUM=4          # 有效 batch = BATCH_SIZE * GRAD_ACCUM
LR=1e-4
WARMUP_RATIO=0.03

########################
# 路径 & 日志
########################

# 切到项目根目录（假设脚本放在项目根目录或 scripts/ 下）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
# 如果你把脚本放到 scripts/ 目录，可以改成：
# PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

mkdir -p experiments/qwen2vl_kline_lora
LOG_FILE="experiments/qwen2vl_kline_lora/train_$(date +%Y%m%d_%H%M%S).log"

echo "Project root: ${PROJECT_ROOT}"
echo "Logging to  : ${LOG_FILE}"
echo "Using GPU   : ${CUDA_VISIBLE_DEVICES}"
echo

########################
# 启动训练
########################

python -u src/train_qwen2vl_classifier.py \
  --train_file "${TRAIN_FILE}" \
  --val_file "${VAL_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_epochs "${NUM_EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --grad_accum "${GRAD_ACCUM}" \
  --learning_rate "${LR}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  2>&1 | tee "${LOG_FILE}"