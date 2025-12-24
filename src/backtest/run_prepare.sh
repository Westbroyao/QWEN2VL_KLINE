#!/usr/bin/env bash
set -euo pipefail

########################################
# 可调参数区：只改这里即可
########################################
PYTHON_BIN="python3"

# 原始 CSV 目录
DATA_DIR="./data"

# 需要保存的字段（逗号分隔）
FIELDS="open,high,low,close,volume,amount,pre_close"

# 日期范围（留空表示不限制）
START_DATE="2015-01-01"
END_DATE=""

# 仅用于快速测试：限制读取文件数；全量设为 0
MAX_FILES="0"

# 存储 dtype：float64 更稳；float32 更省内存更快
DTYPE="float32"

# 输出面板文件（npz 会包含 data/dates/symbols/fields）
OUT_PANEL="panel.npz"

# 是否 gzip 压缩：1/0
COMPRESS="1"
########################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

"${PYTHON_BIN}" prepare_panel.py \
  --data_dir "${DATA_DIR}" \
  --fields "${FIELDS}" \
  --start "${START_DATE}" \
  ${END_DATE:+--end "${END_DATE}"} \
  --max_files "${MAX_FILES}" \
  --dtype "${DTYPE}" \
  --out_path "${OUT_PANEL}" \
  --compress "${COMPRESS}"
