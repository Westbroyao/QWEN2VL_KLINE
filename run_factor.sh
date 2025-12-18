#!/usr/bin/env bash
set -euo pipefail

########################
# 基本配置（按需修改）
########################

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# 模式：train / factor_from_csv / scores_to_npz
MODE="${MODE:-factor_from_csv}"

# 数据 & 输出目录
TRAIN_FILE="data_train/train.jsonl"
VAL_FILE="data_train/val.jsonl"
TEST_FILE="data_test/test.jsonl"
OUTPUT_DIR="experiments/qwen2vl_kline_lora"

# 训练超参数
NUM_EPOCHS=1
BATCH_SIZE=1          # per_device_train_batch_size
GRAD_ACCUM=4          # 有效 batch = BATCH_SIZE * GRAD_ACCUM
LR=1e-4
WARMUP_RATIO=0.03

# 因子 / 回测相关
CSV_GLOB="data_raw/*.csv"
KLINE_OUT_DIR="experiments/kline_images_2025"

RAW_SCORES_PATH="${OUTPUT_DIR}/raw_scores/factor_scores.parquet"
FACTOR_NPZ_PATH="experiments/factors/factor_llm.npz"

START_DATE="2025-01-01"
WINDOW=90
STEP=10
NUM_WORKERS=20

# 因子权重
FACTOR_W_UP=1.0
FACTOR_W_DOWN=-0.5
FACTOR_W_FLAT=0.0

########################
# 日志
########################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
cd "${PROJECT_ROOT}"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "$(dirname "${FACTOR_NPZ_PATH}")"
mkdir -p "$(dirname "${RAW_SCORES_PATH}")"

LOG_FILE="${OUTPUT_DIR}/${MODE}_$(date +%Y%m%d_%H%M%S).log"

echo "Project root : ${PROJECT_ROOT}"
echo "Mode         : ${MODE}"
echo "Logging to   : ${LOG_FILE}"
echo "Using GPU    : ${CUDA_VISIBLE_DEVICES}"
echo

########################
# 按模式启动
########################

if [[ "${MODE}" == "train" ]]; then
  echo "[Run] train mode"

  python -u src/train_qwen2vl_classifier_factor.py \
    --mode train \
    --train_file "${TRAIN_FILE}" \
    --val_file "${VAL_FILE}" \
    --test_file "${TEST_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_epochs "${NUM_EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --grad_accum "${GRAD_ACCUM}" \
    --learning_rate "${LR}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    2>&1 | tee "${LOG_FILE}"

elif [[ "${MODE}" == "factor_from_csv" ]]; then
  echo "[Run] factor_from_csv mode"

  BASE_DIR="${RESUME_FROM:-${OUTPUT_DIR}}"

  python -u src/train_qwen2vl_classifier_factor.py \
    --mode factor_from_csv \
    --resume_from "${BASE_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --csv_glob "${CSV_GLOB}" \
    --kline_out_dir "${KLINE_OUT_DIR}" \
    --raw_scores_path "${RAW_SCORES_PATH}" \
    --factor_npz_path "${FACTOR_NPZ_PATH}" \
    --start_date "${START_DATE}" \
    --window "${WINDOW}" \
    --step "${STEP}" \
    --num_workers "${NUM_WORKERS}" \
    --batch_size "${BATCH_SIZE}" \
    --factor_w_up "${FACTOR_W_UP}" \
    --factor_w_down "${FACTOR_W_DOWN}" \
    --factor_w_flat "${FACTOR_W_FLAT}" \
    2>&1 | tee "${LOG_FILE}"

elif [[ "${MODE}" == "scores_to_npz" ]]; then
  echo "[Run] scores_to_npz mode (no model inference)"

  python -u src/train_qwen2vl_classifier_factor.py \
    --mode scores_to_npz \
    --raw_scores_path "${RAW_SCORES_PATH}" \
    --factor_npz_path "${FACTOR_NPZ_PATH}" \
    --factor_w_up "${FACTOR_W_UP}" \
    --factor_w_down "${FACTOR_W_DOWN}" \
    --factor_w_flat "${FACTOR_W_FLAT}" \
    2>&1 | tee "${LOG_FILE}"

else
  echo "Unknown MODE: ${MODE}"
  exit 1
fi