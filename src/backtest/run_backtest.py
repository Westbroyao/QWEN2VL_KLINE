# run_backtest.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib
import json
import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

TRADING_DAYS = 252
EPS = 1e-12


def load_panel_npz(path: str):
    z = np.load(path, allow_pickle=False)
    data = z["data"]          # (T,N,F)
    dates = z["dates"]        # (T,) datetime64[ns]
    symbols = z["symbols"]    # (N,) str
    fields = z["fields"]      # (F,) str
    fi = {str(f): i for i, f in enumerate(fields.tolist())}
    return data, dates, symbols, fields, fi


def load_factor_npz(path: str):
    z = np.load(path, allow_pickle=False)
    factor = z["factor"]
    dates = z["dates"]
    symbols = z["symbols"]
    return factor, dates, symbols


def load_callable(spec: str):
    if ":" not in spec:
        raise ValueError("Callable spec must be like 'module:function'")
    module_name, fn_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, fn_name, None)
    if fn is None or not callable(fn):
        raise ValueError(f"Cannot load callable '{fn_name}' from module '{module_name}'")
    return fn


def parse_json_arg(text: str) -> Dict:
    if not text:
        return {}
    text = text.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Cannot parse JSON: {text}") from e


def assert_alignment(
    base_dates: np.ndarray,
    base_symbols: np.ndarray,
    other_dates: np.ndarray,
    other_symbols: np.ndarray,
) -> None:
    if base_dates.shape != other_dates.shape or not np.array_equal(base_dates, other_dates):
        raise ValueError("Factor dates are not aligned with panel dates")
    if base_symbols.shape != other_symbols.shape or not np.array_equal(base_symbols, other_symbols):
        raise ValueError("Factor symbols are not aligned with panel symbols")


def align_panel_and_factor(
    data: np.ndarray,
    dates: np.ndarray,
    symbols: np.ndarray,
    factor: np.ndarray,
    f_dates: np.ndarray,
    f_symbols: np.ndarray,
    date_fill: str = "ffill",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.asarray(data)
    factor = np.asarray(factor, dtype=np.float64)

    pd64 = dates.astype("datetime64[ns]")
    fd64 = f_dates.astype("datetime64[ns]")

    if pd64.size == 0 or fd64.size == 0:
        raise ValueError("Empty dates in panel or factor")

    ps = np.array([str(s).strip().upper() for s in symbols.tolist()], dtype=object)
    fs = np.array([str(s).strip().upper() for s in f_symbols.tolist()], dtype=object)

    f_sym_pos = {}
    for j in range(len(fs)):
        f_sym_pos[fs[j]] = j

    panel_cols = [j for j in range(len(ps)) if ps[j] in f_sym_pos]
    if not panel_cols:
        raise ValueError("No overlapping symbols between panel and factor")
    factor_cols = np.array([f_sym_pos[ps[j]] for j in panel_cols], dtype=np.int64)

    panel_start, panel_end = pd64.min(), pd64.max()
    factor_start, factor_end = fd64.min(), fd64.max()
    start = panel_start if panel_start > factor_start else factor_start
    end = panel_end if panel_end < factor_end else factor_end

    mask_range = (pd64 >= start) & (pd64 <= end)
    if not np.any(mask_range):
        raise ValueError("No overlapping date range between panel and factor")

    data2 = data[mask_range, :, :][:, panel_cols, :]
    dates2 = pd64[mask_range]
    symbols2 = symbols[panel_cols]

    fd_int = fd64.astype("int64")
    order = np.argsort(fd_int, kind="mergesort")
    fd_sorted = fd_int[order]
    factor_sorted = factor[order, :]

    d_int = dates2.astype("int64")
    pos = np.searchsorted(fd_sorted, d_int, side="right") - 1
    hit = (pos >= 0) & (fd_sorted[pos] == d_int)
    rows_src = np.full(d_int.shape[0], -1, dtype=np.int64)
    rows_src[hit] = pos[hit]

    T2 = int(dates2.shape[0])
    N2 = int(symbols2.shape[0])
    factor2 = np.full((T2, N2), np.nan, dtype=np.float64)

    if np.any(hit):
        factor2[hit, :] = factor_sorted[rows_src[hit], :][:, factor_cols]

    if date_fill == "intersect":
        if not np.any(hit):
            raise ValueError("No overlapping exact dates between panel and factor (intersect mode)")
        data2 = data2[hit, :, :]
        dates2 = dates2[hit]
        factor2 = factor2[hit, :]
        return data2, dates2, symbols2, factor2

    if date_fill == "ffill":
        last = -1
        for t in range(T2):
            if hit[t]:
                last = t
            else:
                if last != -1:
                    factor2[t, :] = factor2[last, :]

    return data2, dates2, symbols2, factor2


def ensure_target_shape(target_w: np.ndarray, T: int, N: int) -> np.ndarray:
    target_w = np.asarray(target_w, dtype=np.float64)
    if target_w.shape != (T, N):
        raise ValueError(f"target weights shape {target_w.shape} does not match ({T}, {N})")
    target_w[~np.isfinite(target_w)] = 0.0
    return target_w


def load_manual_rebalance_csv(path: str) -> pd.DataFrame:
    """
    Load manual rebalance instructions from CSV.
    Expected columns include date/symbol/weight with flexible names.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    date_col = next((c for c in ["rebalance_date", "trade_date", "date"] if c in df.columns), None)
    sym_col = next((c for c in ["symbol", "ts_code", "code"] if c in df.columns), None)
    w_col = next((c for c in ["weight", "w"] if c in df.columns), None)

    if date_col is None or sym_col is None or w_col is None:
        raise ValueError(f"manual csv must contain date/symbol/weight columns. got columns={df.columns.tolist()}")

    df = df[[date_col, sym_col, w_col]].rename(columns={date_col: "date", sym_col: "symbol", w_col: "weight"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")

    df = df.dropna(subset=["date", "symbol", "weight"])
    return df


def target_weights_from_manual_rebalance(
    rebalance_df: pd.DataFrame,
    dates: np.ndarray,
    symbols: np.ndarray,
    fill_forward: bool = True,
    normalize_if_over_1: bool = True,
    strict: bool = False,
) -> np.ndarray:
    T = int(dates.shape[0])
    N = int(symbols.shape[0])

    date_ints = dates.astype("datetime64[ns]").astype("int64")
    date_to_pos = {int(date_ints[i]): i for i in range(T)}
    sym_to_j = {str(symbols[j]).strip().upper(): j for j in range(N)}

    target = np.full((T, N), np.nan, dtype=np.float64)
    has_reb = np.zeros(T, dtype=bool)

    cash_alias = {"CASH", "__CASH__", "ALL_CASH"}

    for dt, grp in rebalance_df.groupby("date"):
        dt64 = np.datetime64(dt.to_datetime64()).astype("datetime64[ns]")
        key = int(dt64.astype("int64"))
        if key not in date_to_pos:
            if strict:
                raise ValueError(f"rebalance date {str(dt)[:10]} not in backtest dates range")
            continue
        t = date_to_pos[key]
        row = np.zeros(N, dtype=np.float64)

        if (grp["symbol"].isin(list(cash_alias))).any():
            row[:] = 0.0
        else:
            for _, r in grp.iterrows():
                sym = str(r["symbol"]).strip().upper()
                w = float(r["weight"])
                j = sym_to_j.get(sym)
                if j is None:
                    if strict:
                        raise ValueError(f"symbol '{sym}' not in panel universe")
                    continue
                row[j] = w

        s = float(row.sum())
        if normalize_if_over_1 and s > 1.0 + EPS:
            row /= s

        target[t, :] = row
        has_reb[t] = True

    if not fill_forward:
        target[np.isnan(target)] = 0.0
        return target

    last = np.zeros(N, dtype=np.float64)
    for t in range(T):
        if has_reb[t]:
            last = target[t, :].copy()
        target[t, :] = last

    return target


def weights_from_factor_topk_np(
    factor: np.ndarray,
    topk: int,
    rebalance_every: int,
    descending: bool = True,
    tradable: Optional[np.ndarray] = None,
    weight_mode: str = "equal",
    fill_forward: bool = True,
    keep_prev_on_empty: bool = True,
) -> np.ndarray:
    """
    Strategy-agnostic selection: pick top-k based on factor and assign weights.
    """
    factor = np.asarray(factor, dtype=np.float64)
    T, N = factor.shape
    w = np.zeros((T, N), dtype=np.float64)

    if T == 0 or N == 0 or topk <= 0:
        return w
    if rebalance_every <= 0:
        raise ValueError("rebalance_every must be positive")

    updated = np.zeros(T, dtype=bool)

    for t in range(0, T, rebalance_every):
        s = factor[t, :]
        valid = np.isfinite(s)
        if tradable is not None:
            valid &= tradable[t, :]

        k = min(topk, int(valid.sum()))
        if k <= 0:
            continue

        s2 = s.copy()
        if descending:
            s2[~valid] = -np.inf
            idx = np.argpartition(s2, -k)[-k:]
            idx = idx[np.isfinite(s2[idx])]
        else:
            s2[~valid] = np.inf
            idx = np.argpartition(s2, k)[:k]
            idx = idx[np.isfinite(s2[idx])]

        if idx.size == 0:
            continue

        if weight_mode == "equal":
            w[t, idx] = 1.0 / float(idx.size)
        else:
            raise ValueError(f"Unsupported weight_mode={weight_mode}")

        updated[t] = True

    if not fill_forward:
        return w

    last = np.zeros(N, dtype=np.float64)
    for t in range(T):
        if updated[t]:
            last = w[t, :].copy()
        else:
            if (t % rebalance_every == 0) and (not updated[t]) and (not keep_prev_on_empty):
                last = np.zeros(N, dtype=np.float64)
        w[t, :] = last

    return w


def weights_from_factor_quantile_np(
    factor: np.ndarray,
    n_groups: int = 5,
    group_id: int = 1,
    rebalance_every: int = 1,
    tradable: Optional[np.ndarray] = None,
    weight_mode: str = "equal",
    fill_forward: bool = True,
    keep_prev_on_empty: bool = True,
) -> np.ndarray:
    factor = np.asarray(factor, dtype=np.float64)
    T, N = factor.shape
    w = np.zeros((T, N), dtype=np.float64)

    if n_groups <= 1:
        raise ValueError("n_groups must be >= 2")
    if group_id < 1 or group_id > n_groups:
        raise ValueError(f"group_id must be in [1, {n_groups}]")
    if rebalance_every <= 0:
        raise ValueError("rebalance_every must be positive")
    if T == 0 or N == 0:
        return w

    updated = np.zeros(T, dtype=bool)

    for t in range(0, T, rebalance_every):
        s = factor[t, :]
        valid = np.isfinite(s)
        if tradable is not None:
            valid &= tradable[t, :]

        idx_valid = np.where(valid)[0]
        if idx_valid.size == 0:
            continue

        order = np.argsort(s[idx_valid], kind="mergesort")
        idx_sorted = idx_valid[order]

        groups = np.array_split(idx_sorted, n_groups)
        bucket = groups[group_id - 1]
        if bucket.size == 0:
            continue

        if weight_mode == "equal":
            w[t, bucket] = 1.0 / float(bucket.size)
        else:
            raise ValueError(f"Unsupported weight_mode={weight_mode}")

        updated[t] = True

    if not fill_forward:
        return w

    last = np.zeros(N, dtype=np.float64)
    for t in range(T):
        if updated[t]:
            last = w[t, :].copy()
        else:
            if (t % rebalance_every == 0) and (not updated[t]) and (not keep_prev_on_empty):
                last = np.zeros(N, dtype=np.float64)
        w[t, :] = last

    return w


def run_backtest_np(
    close: np.ndarray,
    target_w: np.ndarray,
    commission_bps: float,
    slippage_bps: float,
    cash_rate: float,
    initial_nav: float = 1.0,
    open_px: Optional[np.ndarray] = None,
    exec_mode: str = "close_close",
):
    close = np.asarray(close, dtype=np.float64)
    target_w = np.asarray(target_w, dtype=np.float64)

    T, N = close.shape
    if T == 0 or N == 0:
        zero = np.zeros(T, dtype=np.float64)
        return {
            "port_ret_gross": zero.copy(),
            "turnover": zero.copy(),
            "cost_rate": zero.copy(),
            "port_ret_net": zero.copy(),
            "nav": np.full(T, float(initial_nav), dtype=np.float64),
            "cash_weight": zero.copy(),
            "cash_ret": zero.copy(),
        }

    close[~np.isfinite(close)] = np.nan
    target_w[~np.isfinite(target_w)] = 0.0

    cost_bps = (commission_bps + slippage_bps) * 1e-4
    cash_r_daily = (1.0 + float(cash_rate)) ** (1.0 / float(TRADING_DAYS)) - 1.0

    if exec_mode == "close_close":
        ret = np.zeros((T, N), dtype=np.float64)
        if T > 1:
            r = close[1:, :] / close[:-1, :] - 1.0
            r[~np.isfinite(r)] = 0.0
            ret[1:, :] = r

        w = np.zeros((T, N), dtype=np.float64)
        if T > 1:
            w[1:, :] = target_w[:-1, :]
        w[~np.isfinite(w)] = 0.0

        wsum = w.sum(axis=1)
        bad = wsum > 1.0 + EPS
        if np.any(bad):
            w[bad, :] = w[bad, :] / wsum[bad][:, None]

        w_prev = np.zeros_like(w)
        if T > 1:
            w_prev[1:, :] = w[:-1, :]
        turnover = np.abs(w - w_prev).sum(axis=1)

        cost_rate = turnover * cost_bps

        cash_w = np.clip(1.0 - w.sum(axis=1), 0.0, None)
        cash_ret = cash_w * cash_r_daily

        port_ret_gross = (w * ret).sum(axis=1) + cash_ret
        port_ret_net = port_ret_gross - cost_rate
        nav = np.cumprod(1.0 + port_ret_net) * float(initial_nav)

        return {
            "port_ret_gross": port_ret_gross,
            "turnover": turnover,
            "cost_rate": cost_rate,
            "port_ret_net": port_ret_net,
            "nav": nav,
            "cash_weight": cash_w,
            "cash_ret": cash_ret,
        }

    if open_px is None:
        raise ValueError(f"exec_mode={exec_mode} requires open prices")
    open_px = np.asarray(open_px, dtype=np.float64)
    open_px[~np.isfinite(open_px)] = np.nan

    w_intra = np.zeros((T, N), dtype=np.float64)
    if T > 1:
        w_intra[1:, :] = target_w[:-1, :]
    w_intra[~np.isfinite(w_intra)] = 0.0

    wsum = w_intra.sum(axis=1)
    bad = wsum > 1.0 + EPS
    if np.any(bad):
        w_intra[bad, :] = w_intra[bad, :] / wsum[bad][:, None]

    w_prev = np.zeros_like(w_intra)
    if T > 1:
        w_prev[1:, :] = w_intra[:-1, :]

    diff = np.abs(w_intra - w_prev).sum(axis=1)
    is_reb = diff > EPS
    is_reb[0] = False

    if exec_mode == "open_rebalance":
        turnover = diff
    elif exec_mode == "open_rebalance_flat":
        turnover = diff.copy()
        if np.any(is_reb):
            turnover[is_reb] = np.abs(w_prev[is_reb, :]).sum(axis=1) + np.abs(w_intra[is_reb, :]).sum(axis=1)
    else:
        raise ValueError(f"Unknown exec_mode={exec_mode}")

    cost_rate = turnover * cost_bps

    gap_ret = np.zeros((T, N), dtype=np.float64)
    if T > 1:
        g = open_px[1:, :] / close[:-1, :] - 1.0
        g[~np.isfinite(g)] = 0.0
        gap_ret[1:, :] = g

    intra_ret = np.zeros((T, N), dtype=np.float64)
    x = close / open_px - 1.0
    x[~np.isfinite(x)] = 0.0
    intra_ret[:, :] = x

    cash_r_half = np.sqrt(1.0 + cash_r_daily) - 1.0

    w_gap = w_prev.copy()
    if exec_mode == "open_rebalance_flat" and np.any(is_reb):
        w_gap[is_reb, :] = 0.0

    cash_gap = np.clip(1.0 - w_gap.sum(axis=1), 0.0, None)
    cash_intra = np.clip(1.0 - w_intra.sum(axis=1), 0.0, None)

    port_ret_gap = np.zeros(T, dtype=np.float64)
    port_ret_intra = np.zeros(T, dtype=np.float64)

    if T > 1:
        port_ret_gap[1:] = (w_gap[1:, :] * gap_ret[1:, :]).sum(axis=1) + cash_gap[1:] * cash_r_half
        port_ret_intra[1:] = (w_intra[1:, :] * intra_ret[1:, :]).sum(axis=1) + cash_intra[1:] * cash_r_half

    port_ret_gross = (1.0 + port_ret_gap) * (1.0 + port_ret_intra) - 1.0
    port_ret_net = (1.0 + port_ret_gap) * (1.0 + (port_ret_intra - cost_rate)) - 1.0

    nav = np.cumprod(1.0 + port_ret_net) * float(initial_nav)

    cash_w_close = np.clip(1.0 - w_intra.sum(axis=1), 0.0, None)
    cash_ret = cash_w_close * cash_r_daily

    return {
        "port_ret_gross": port_ret_gross,
        "turnover": turnover,
        "cost_rate": cost_rate,
        "port_ret_net": port_ret_net,
        "nav": nav,
        "cash_weight": cash_w_close,
        "cash_ret": cash_ret,
    }


def max_drawdown(nav: pd.Series) -> float:
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(dd.min())


def perf_report(bt: pd.DataFrame, annualization: int = 252) -> Dict[str, float]:
    bt = bt.dropna(subset=["nav", "port_ret_net"]).sort_index()
    r = bt["port_ret_net"].values
    nav = bt["nav"]

    total_return = float(nav.iloc[-1] / nav.iloc[0] - 1.0)

    if isinstance(bt.index, pd.DatetimeIndex):
        years = (bt.index[-1] - bt.index[0]).days / 365.25
    else:
        years = len(bt) / float(annualization)

    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1.0 / max(years, EPS)) - 1.0

    vol = float(np.std(r, ddof=1) * np.sqrt(annualization))
    mean = float(np.mean(r) * annualization)
    sharpe = mean / vol if vol > EPS else np.nan

    mdd = max_drawdown(nav)

    return {
        "TotalReturn": total_return,
        "CAGR": float(cagr),
        "AnnVol": vol,
        "Sharpe": float(sharpe),
        "MaxDrawdown": float(mdd),
        "AvgDailyTurnover": float(bt["turnover"].mean()),
        "AnnCostApprox": float(bt["cost_rate"].mean() * annualization),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel_path", type=str, required=True)
    ap.add_argument("--start", type=str, default="")
    ap.add_argument("--end", type=str, default="")
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--rebalance_every", type=int, default=1)
    ap.add_argument("--descending", type=int, default=1, help="1 picks largest factors, 0 picks smallest")
    ap.add_argument("--weight_mode", type=str, default="equal")
    ap.add_argument("--portfolio_mode", type=str, default="topk", choices=["topk", "quintile", "both"])
    ap.add_argument("--n_groups", type=int, default=5, help="factor quantile groups (default 5)")
    ap.add_argument("--group_id", type=int, default=-1, help="-1=all groups; else 1..n_groups")
    ap.add_argument("--factor_fill_forward", type=int, default=1, help="factor modes: 1=hold until next rebalance")
    ap.add_argument("--commission_bps", type=float, default=3.0)
    ap.add_argument("--slippage_bps", type=float, default=0.0)
    ap.add_argument("--cash_rate", type=float, default=0.0, help="annual cash rate, e.g. 0.02 for 2% p.a.")
    ap.add_argument(
        "--exec_mode",
        type=str,
        default="close_close",
        choices=["close_close", "open_rebalance", "open_rebalance_flat"],
        help=(
            "Execution mode: close_close=trade at close (default); "
            "open_rebalance=rebalance at open and hold intraday; "
            "open_rebalance_flat=rebalance at open with prior close flat"
        ),
    )
    ap.add_argument("--manual_path", type=str, default="", help="CSV instructions: rebalance_date,symbol,weight")
    ap.add_argument("--manual_fill_forward", type=int, default=1, help="1=hold until next rebalance, 0=only set on rebalance date")
    ap.add_argument("--manual_normalize", type=int, default=1, help="1=normalize weights if sum>1")
    ap.add_argument("--manual_strict", type=int, default=0, help="1=error on missing symbol/date, 0=skip")
    ap.add_argument("--factor_path", type=str, default="", help="npz file containing factor/dates/symbols")
    ap.add_argument(
        "--factor_date_fill",
        type=str,
        default="ffill",
        choices=["ffill", "nan", "intersect"],
        help="factor date handling: ffill=carry forward, nan=keep NaN, intersect=keep only factor dates",
    )
    ap.add_argument("--factor_builder", type=str, default="", help="module:function that constructs factor")
    ap.add_argument("--builder_params", type=str, default="{}")
    ap.add_argument("--strategy", type=str, default="", help="module:function that outputs target weights")
    ap.add_argument("--strategy_params", type=str, default="{}")
    ap.add_argument("--save_factor_npz", type=str, default="", help="optional path to cache computed factor")
    ap.add_argument("--out_csv", type=str, default="backtest_result.csv")
    args = ap.parse_args()

    data, dates, symbols, fields, fi = load_panel_npz(args.panel_path)
    if "close" not in fi:
        raise ValueError(f"'close' not found in fields={fields.tolist()}")

    mask = np.ones(dates.shape[0], dtype=bool)
    if args.start:
        start_dt = np.datetime64(pd.to_datetime(args.start))
        mask &= dates >= start_dt
    if args.end:
        end_dt = np.datetime64(pd.to_datetime(args.end))
        mask &= dates <= end_dt

    data = data[mask, :, :]
    dates = dates[mask]

    close = data[:, :, fi["close"]]
    T, N = close.shape
    open_px = None
    if args.exec_mode != "close_close":
        if "open" not in fi:
            raise ValueError(f"'open' not found in fields={fields.tolist()} but exec_mode={args.exec_mode} requires open")
        open_px = data[:, :, fi["open"]]

    mode_flags = {
        "manual": bool(args.manual_path),
        "factor_path": bool(args.factor_path),
        "factor_builder": bool(args.factor_builder),
        "strategy": bool(args.strategy),
    }
    selected = [m for m, flag in mode_flags.items() if flag]
    if len(selected) != 1:
        raise ValueError("Provide exactly one of --manual_path, --factor_path, --factor_builder, --strategy")
    selected_mode = selected[0]

    builder_params = parse_json_arg(args.builder_params)
    strategy_params = parse_json_arg(args.strategy_params)

    factor = None
    target_w = None

    if selected_mode == "manual":
        manual_df = load_manual_rebalance_csv(args.manual_path)
        target_w = target_weights_from_manual_rebalance(
            rebalance_df=manual_df,
            dates=dates,
            symbols=symbols,
            fill_forward=bool(args.manual_fill_forward),
            normalize_if_over_1=bool(args.manual_normalize),
            strict=bool(args.manual_strict),
        )

    elif selected_mode == "strategy":
        strat_fn = load_callable(args.strategy)
        target_w = strat_fn(
            data=data,
            fi=fi,
            dates=dates,
            symbols=symbols,
            **strategy_params,
        )
    else:
        if selected_mode == "factor_path":
            factor, f_dates, f_symbols = load_factor_npz(args.factor_path)
            data, dates, symbols, factor = align_panel_and_factor(
                data=data,
                dates=dates,
                symbols=symbols,
                factor=factor,
                f_dates=f_dates,
                f_symbols=f_symbols,
                date_fill=args.factor_date_fill,
            )
            close = data[:, :, fi["close"]]
            T, N = close.shape
            if args.exec_mode != "close_close":
                if "open" not in fi:
                    raise ValueError(f"'open' not found in fields={fields.tolist()} but exec_mode={args.exec_mode} requires open")
                open_px = data[:, :, fi["open"]]
        elif selected_mode == "factor_builder":
            builder_fn = load_callable(args.factor_builder)
            factor = builder_fn(
                data=data,
                fi=fi,
                dates=dates,
                symbols=symbols,
                **builder_params,
            )

        if factor is None:
            raise ValueError("Factor builder returned None")
        if factor.shape != (T, N):
            raise ValueError(f"Factor shape {factor.shape} does not match ({T}, {N})")

        if args.save_factor_npz:
            save_dates = dates.astype("datetime64[ns]")
            np.savez_compressed(args.save_factor_npz, factor=factor, dates=save_dates, symbols=symbols)

        if args.portfolio_mode not in ("topk", "quintile", "both"):
            raise ValueError(f"Unknown portfolio_mode={args.portfolio_mode}")

        root, ext = os.path.splitext(args.out_csv)
        if not ext:
            ext = ".csv"

        if args.portfolio_mode in ("topk", "both"):
            target_w = weights_from_factor_topk_np(
                factor=factor,
                topk=args.topk,
                rebalance_every=args.rebalance_every,
                descending=bool(args.descending),
                tradable=None,
                weight_mode=args.weight_mode,
                fill_forward=bool(args.factor_fill_forward),
            )
            target_w = ensure_target_shape(target_w, T, N)

            res = run_backtest_np(
                close=close,
                target_w=target_w,
                commission_bps=args.commission_bps,
                slippage_bps=args.slippage_bps,
                cash_rate=args.cash_rate,
                initial_nav=1.0,
                open_px=open_px,
                exec_mode=args.exec_mode,
            )
            bt = pd.DataFrame(res, index=pd.to_datetime(dates))
            rep = perf_report(bt)
            print("\n===== Performance (TOPK) =====")
            for k, v in rep.items():
                print(f"{k:>16s}: {v: .6f}")
            bt.to_csv(args.out_csv, encoding="utf-8-sig")
            print(f"\n[INFO] saved: {args.out_csv}")

        if args.portfolio_mode in ("quintile", "both"):
            groups = list(range(1, args.n_groups + 1)) if args.group_id == -1 else [args.group_id]
            summary = []
            bt_by_group = {}

            for g in groups:
                wq = weights_from_factor_quantile_np(
                    factor=factor,
                    n_groups=args.n_groups,
                    group_id=g,
                    rebalance_every=args.rebalance_every,
                    tradable=None,
                    weight_mode=args.weight_mode,
                    fill_forward=bool(args.factor_fill_forward),
                )
                wq = ensure_target_shape(wq, T, N)

                res_q = run_backtest_np(
                    close=close,
                    target_w=wq,
                    commission_bps=args.commission_bps,
                    slippage_bps=args.slippage_bps,
                    cash_rate=args.cash_rate,
                    initial_nav=1.0,
                    open_px=open_px,
                    exec_mode=args.exec_mode,
                )
                bt_q = pd.DataFrame(res_q, index=pd.to_datetime(dates))
                bt_by_group[g] = bt_q
                rep_q = perf_report(bt_q)
                rep_q["Group"] = f"Q{g}"
                summary.append(rep_q)

                out_q = f"{root}_Q{g}{ext}"
                bt_q.to_csv(out_q, encoding="utf-8-sig")
                print(f"[INFO] saved: {out_q}")

            low_g = 1
            high_g = args.n_groups
            if (low_g in bt_by_group) and (high_g in bt_by_group):
                bt_low = bt_by_group[low_g]
                bt_high = bt_by_group[high_g]

                r_ls = bt_high["port_ret_net"].to_numpy(dtype=np.float64) - bt_low["port_ret_net"].to_numpy(dtype=np.float64)
                r_ls[~np.isfinite(r_ls)] = 0.0

                turnover_ls = bt_high["turnover"].to_numpy(dtype=np.float64) + bt_low["turnover"].to_numpy(dtype=np.float64)
                cost_ls = bt_high["cost_rate"].to_numpy(dtype=np.float64) + bt_low["cost_rate"].to_numpy(dtype=np.float64)

                ls = pd.DataFrame(index=bt_high.index)
                if r_ls.size > 0:
                    r_ls[0] = 0.0
                    turnover_ls[0] = 0.0
                    cost_ls[0] = 0.0

                ls["port_ret_net"] = r_ls
                ls["turnover"] = turnover_ls
                ls["cost_rate"] = cost_ls
                ls["port_ret_gross"] = ls["port_ret_net"] + ls["cost_rate"]

                nav_ls = np.ones_like(r_ls, dtype=np.float64)
                if r_ls.size > 1:
                    nav_ls[1:] = np.cumprod(1.0 + r_ls[1:])
                ls["nav"] = nav_ls

                rep_ls = perf_report(ls)
                rep_ls["Group"] = f"LS(Q{high_g}-Q{low_g})"
                summary.append(rep_ls)

                out_ls = f"{root}_LS_Q{high_g}-Q{low_g}{ext}"
                ls.to_csv(out_ls, encoding="utf-8-sig")
                print(f"[INFO] saved: {out_ls}")

            if summary:
                summ_df = pd.DataFrame(summary).set_index("Group").sort_index()
                out_summ = f"{root}_quintile_summary{ext}"
                summ_df.to_csv(out_summ, encoding="utf-8-sig")
                print(f"[INFO] saved: {out_summ}")

                print("\n===== Performance (QUINTILES) =====")
                print(summ_df.to_string(float_format=lambda x: f"{x: .6f}"))

        return

    target_w = ensure_target_shape(target_w, T, N)

    res = run_backtest_np(
        close=close,
        target_w=target_w,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
        cash_rate=args.cash_rate,
        initial_nav=1.0,
        open_px=open_px,
        exec_mode=args.exec_mode,
    )

    dates_pd = pd.to_datetime(dates)
    bt = pd.DataFrame(res, index=dates_pd)

    rep = perf_report(bt)
    print("\n===== Performance =====")
    for k, v in rep.items():
        print(f"{k:>16s}: {v: .6f}")

    bt.to_csv(args.out_csv, encoding="utf-8-sig")
    print(f"\n[INFO] saved: {args.out_csv}")


if __name__ == "__main__":
    main()
