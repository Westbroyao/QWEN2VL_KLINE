#!/usr/bin/env bash
set -euo pipefail

########################################
# 可调参数区：只改这里即可
########################################
PYTHON_BIN="python3"

# 准备阶段输出的面板文件（npz：data/dates/symbols/fields）
PANEL_PATH="panel.npz"

# 回测区间（留空表示不切）
START_DATE=""
END_DATE=""

# 成本与现金（年化现金利率，例如 0.02=2% p.a.）
COMMISSION_BPS="3"
SLIPPAGE_BPS="0"
CASH_RATE="0.00"
# 成交价模式
EXEC_MODE="open_rebalance_flat"   # close_close | open_rebalance | open_rebalance_flat

# 输出
OUT_CSV="results/backtest_result.csv"

# -------- Mode A: factor_file --------
FACTOR_PATH="factors_out/factor_llm_sanitized.npz"

# 四种模式切换：
#   manual         : CSV 调仓（rebalance_date,symbol,weight）
#   factor_file    : 外部输入 factor.npz（含 factor/dates/symbols）
#   factor_builder : 外部输入 因子构造方式 + 通用选股器
#   strategy       : 外部输入 策略函数（直接输出 target_w）
MODE="factor_file"

# -------- Mode D: manual --------
MANUAL_PATH="manual/manual_rebalance.csv"
MANUAL_FILL_FORWARD="1"
MANUAL_NORMALIZE="1"
MANUAL_STRICT="0"

# factor -> weights（Mode A/C 共用）
TOPK="30"
REBALANCE_EVERY="30"
DESCENDING="1"        # 1=选最大；0=选最小
WEIGHT_MODE="equal"
# 组合构建模式（仅 factor 模式）
PORTFOLIO_MODE="both"   # topk | quintile | both
N_GROUPS="5"
GROUP_ID="-1"           # -1=全部分组；否则 1..N_GROUPS
FACTOR_FILL_FORWARD="1"
# factor 日期填充策略：ffill | nan | intersect
FACTOR_DATE_FILL="ffill"

# -------- Mode C: factor_builder --------
FACTOR_BUILDER_SPEC="factors.mom:make_factor"
BUILDER_PARAMS='{"lookback":20}'
SAVE_FACTOR_NPZ="factors_out/factor_cached.npz"

# -------- Mode B: strategy --------
STRATEGY_SPEC="strategies.mom_topk:make_target_weights"
STRATEGY_PARAMS='{"lookback":20,"topk":30,"rebalance_every":5}'
########################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# 确保可 import 本地 factors/strategies
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

mkdir -p "$(dirname "${OUT_CSV}")"

# optional: guard manual file existence
# if [[ "${MODE}" == "manual" && ! -f "${MANUAL_PATH}" ]]; then
#   echo "[ERROR] manual file not found: ${MANUAL_PATH}"
#   exit 1
# fi

COMMON_ARGS=(
  --panel_path "${PANEL_PATH}"
  --out_csv "${OUT_CSV}"
  --commission_bps "${COMMISSION_BPS}"
  --slippage_bps "${SLIPPAGE_BPS}"
  --cash_rate "${CASH_RATE}"
  --exec_mode "${EXEC_MODE}"
  --portfolio_mode "${PORTFOLIO_MODE}"
  --n_groups "${N_GROUPS}"
  --group_id "${GROUP_ID}"
  --factor_fill_forward "${FACTOR_FILL_FORWARD}"
  --factor_date_fill "${FACTOR_DATE_FILL}"
)
if [[ -n "${START_DATE}" ]]; then COMMON_ARGS+=( --start "${START_DATE}" ); fi
if [[ -n "${END_DATE}" ]]; then   COMMON_ARGS+=( --end "${END_DATE}" );   fi

MODE_ARGS=()
case "${MODE}" in
  manual)
    MODE_ARGS+=( --manual_path "${MANUAL_PATH}" )
    MODE_ARGS+=( --manual_fill_forward "${MANUAL_FILL_FORWARD}" )
    MODE_ARGS+=( --manual_normalize "${MANUAL_NORMALIZE}" )
    MODE_ARGS+=( --manual_strict "${MANUAL_STRICT}" )
    ;;
  factor_file)
    MODE_ARGS+=( --factor_path "${FACTOR_PATH}" )
    MODE_ARGS+=( --topk "${TOPK}" --rebalance_every "${REBALANCE_EVERY}" --descending "${DESCENDING}" --weight_mode "${WEIGHT_MODE}" )
    ;;
  factor_builder)
    MODE_ARGS+=( --factor_builder "${FACTOR_BUILDER_SPEC}" --builder_params "${BUILDER_PARAMS}" )
    MODE_ARGS+=( --topk "${TOPK}" --rebalance_every "${REBALANCE_EVERY}" --descending "${DESCENDING}" --weight_mode "${WEIGHT_MODE}" )
    if [[ -n "${SAVE_FACTOR_NPZ}" ]]; then
      MODE_ARGS+=( --save_factor_npz "${SAVE_FACTOR_NPZ}" )
      mkdir -p "$(dirname "${SAVE_FACTOR_NPZ}")"
    fi
    ;;
  strategy)
    MODE_ARGS+=( --strategy "${STRATEGY_SPEC}" --strategy_params "${STRATEGY_PARAMS}" )
    ;;
  *)
    echo "[ERROR] Unknown MODE='${MODE}'. Use: manual | factor_file | factor_builder | strategy"
    exit 1
    ;;
esac

echo "[INFO] MODE=${MODE}"
echo "[INFO] PANEL_PATH=${PANEL_PATH}"
echo "[INFO] OUT_CSV=${OUT_CSV}"

"${PYTHON_BIN}" run_backtest.py "${COMMON_ARGS[@]}" "${MODE_ARGS[@]}"
