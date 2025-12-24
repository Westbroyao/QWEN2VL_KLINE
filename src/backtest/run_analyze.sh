#!/usr/bin/env bash
set -euo pipefail

########################################
# 可调参数区：只改这里即可
########################################
PYTHON_BIN="python3"

IN_CSV="results/backtest_result.csv"
OUT_DIR="analysis_out"

# 年化因子：日频A股一般用252
ANNUALIZATION="252"

# 分析区间（留空表示不限制）
START_DATE=""
END_DATE=""
########################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

"${PYTHON_BIN}" analyze_result.py \
  --in_csv "${IN_CSV}" \
  --out_dir "${OUT_DIR}" \
  --annualization "${ANNUALIZATION}" \
  --start "${START_DATE}" \
  --end "${END_DATE}"
