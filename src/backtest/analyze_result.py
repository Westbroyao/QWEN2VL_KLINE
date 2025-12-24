# analyze_result.py
# -*- coding: utf-8 -*-

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---- Global matplotlib font config (Times New Roman) ----
mpl.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",     # 让数学公式字体也更接近 Times 系
    "axes.unicode_minus": False,    # 避免负号显示异常

    # 可选：统一字号（不想改就删掉这些）
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})

EPS = 1e-12

def max_drawdown(nav: pd.Series) -> float:
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(dd.min())


def perf_report(df: pd.DataFrame, annualization: int = 252) -> dict:
    df = df.dropna(subset=["nav", "port_ret_net"]).copy()
    df = df.sort_index()

    nav = df["nav"]
    r = df["port_ret_net"].values

    total_return = float(nav.iloc[-1] / nav.iloc[0] - 1.0)

    # 年数：优先用真实日期跨度；如果 index 不是 DatetimeIndex，就用样本长度近似
    if isinstance(df.index, pd.DatetimeIndex):
        years = (df.index[-1] - df.index[0]).days / 365.25
    else:
        years = len(df) / float(annualization)
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1.0 / max(years, EPS)) - 1.0

    vol = float(np.std(r, ddof=1) * np.sqrt(annualization))
    mean = float(np.mean(r) * annualization)
    sharpe = mean / vol if vol > EPS else np.nan

    mdd = max_drawdown(nav)
    calmar = float(cagr / abs(mdd)) if mdd < 0 else np.nan

    avg_turnover = float(df["turnover"].mean()) if "turnover" in df.columns else np.nan
    ann_cost = float(df["cost_rate"].mean() * annualization) if "cost_rate" in df.columns else np.nan

    return {
        "TotalReturn": total_return,
        "CAGR": float(cagr),
        "AnnVol": vol,
        "Sharpe": float(sharpe),
        "MaxDrawdown": float(mdd),
        "Calmar": calmar,
        "AvgDailyTurnover": avg_turnover,
        "AnnCostApprox": ann_cost,
    }


def read_bt_csv_with_date_index(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    candidates = []
    for c in ["date", "Unnamed: 0", "trade_date"]:
        if c in df.columns:
            candidates.append(c)
    if not candidates:
        candidates.append(df.columns[0])

    best_col = None
    best_ratio = -1.0
    best_dt = None

    for c in candidates:
        dt = pd.to_datetime(df[c], errors="coerce")
        ratio = float(dt.notna().mean())
        if ratio > best_ratio:
            best_ratio = ratio
            best_col = c
            best_dt = dt

    if best_ratio < 0.8:
        raise ValueError(
            f"Cannot parse a valid date column from {candidates}. "
            f"Best candidate '{best_col}' parse_ratio={best_ratio:.2f}. "
            f"Please check the first column / date format in {path}."
        )

    df = df.assign(__date=best_dt).dropna(subset=["__date"]).set_index("__date")
    df.index.name = "date"
    df = df.sort_index()
    return df


def plot_nav_and_dd(df: pd.DataFrame, out_dir: str):
    nav = df["nav"].dropna()
    peak = nav.cummax()
    dd = nav / peak - 1.0

    plt.figure()
    nav.plot()
    plt.title("NAV")
    plt.xlabel("Date")
    plt.ylabel("nav")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "nav.png"), dpi=150)
    plt.close()

    plt.figure()
    dd.plot()
    plt.title("Drawdown")
    plt.xlabel("Date")
    plt.ylabel("drawdown")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "drawdown.png"), dpi=150)
    plt.close()

    # optional: log-nav
    plt.figure()
    np.log(nav).plot()
    plt.title("log(NAV)")
    plt.xlabel("Date")
    plt.ylabel("log(nav)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "log_nav.png"), dpi=150)
    plt.close()


def plot_turnover_cost(df: pd.DataFrame, out_dir: str):
    has_turn = "turnover" in df.columns
    has_cost = "cost_rate" in df.columns
    if not (has_turn or has_cost):
        return

    plt.figure()
    if has_turn:
        df["turnover"].plot()
    plt.title("Turnover (daily)")
    plt.xlabel("Date")
    plt.ylabel("turnover")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "turnover.png"), dpi=150)
    plt.close()

    if has_cost:
        plt.figure()
        df["cost_rate"].plot()
        plt.title("Cost rate (daily)")
        plt.xlabel("Date")
        plt.ylabel("cost_rate")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "cost_rate.png"), dpi=150)
        plt.close()


def monthly_returns(df: pd.DataFrame) -> pd.Series:
    nav = df["nav"].dropna().sort_index()
    m = nav.resample("ME").last().pct_change().dropna()
    m.index = m.index.to_period("M").astype(str)
    return m


def rolling_1y_returns(df: pd.DataFrame, annualization: int = 252) -> pd.Series:
    nav = df["nav"].dropna().sort_index()
    rr = nav.pct_change(annualization).dropna()
    rr.name = "rolling_1y_return"
    return rr


def drawdown_durations(nav: pd.Series) -> pd.DataFrame:
    nav = nav.dropna().sort_index()
    if nav.empty:
        return pd.DataFrame(columns=["start", "end", "length"])

    peak = nav.cummax()
    underwater = nav < (peak * (1.0 - EPS))

    durations = []
    start = None
    length = 0
    idx = nav.index

    for i, uw in enumerate(underwater.values):
        if uw:
            if start is None:
                start = idx[i]
                length = 1
            else:
                length += 1
        else:
            if start is not None:
                end = idx[i - 1]
                durations.append((start, end, length))
                start = None
                length = 0

    if start is not None:
        durations.append((start, idx[-1], length))

    return pd.DataFrame(durations, columns=["start", "end", "length"])


def yearly_perf_table(df: pd.DataFrame, annualization: int = 252) -> pd.DataFrame:
    df = df.dropna(subset=["nav", "port_ret_net"]).copy()
    df = df.sort_index()
    if not isinstance(df.index, pd.DatetimeIndex) or df.empty:
        return pd.DataFrame()

    rows = []
    for y, sub in df.groupby(df.index.year):
        if len(sub) < 5:
            continue
        rep = perf_report(sub, annualization=annualization)
        rep["Year"] = str(int(y))
        rows.append(rep)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).set_index("Year").sort_index()
    return out


def plot_rolling_1y(rr: pd.Series, out_dir: str):
    if rr is None or rr.empty:
        return

    plt.figure()
    rr.plot()
    plt.title("Rolling 1Y Return")
    plt.xlabel("Date")
    plt.ylabel("return")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rolling_1y_return.png"), dpi=150)
    plt.close()

    plt.figure()
    rr.hist(bins=50)
    plt.title("Rolling 1Y Return (hist)")
    plt.xlabel("return")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rolling_1y_return_hist.png"), dpi=150)
    plt.close()


def discover_quintile_siblings(in_csv: str):
    root, ext = os.path.splitext(in_csv)
    if not ext:
        ext = ".csv"

    q_files = glob.glob(f"{root}_Q*{ext}")
    q_map = {}
    for p in q_files:
        base = os.path.basename(p)
        try:
            q_part = base.split("_Q")[-1].split(ext)[0]
            g = int(q_part)
            q_map[g] = p
        except Exception:
            continue

    ls_candidates = glob.glob(f"{root}_LS_Q*-Q*{ext}")
    ls_path = ls_candidates[0] if ls_candidates else ""

    summ_path = f"{root}_quintile_summary{ext}"
    if not os.path.exists(summ_path):
        summ_path = ""

    return q_map, ls_path, summ_path


def analyze_quintiles(
    in_csv: str,
    out_dir: str,
    annualization: int,
    start: str = "",
    end: str = "",
):
    q_map, ls_path, summ_path = discover_quintile_siblings(in_csv)

    if len(q_map) < 2:
        return

    bt_by_group = {}
    for g in sorted(q_map.keys()):
        bt = read_bt_csv_with_date_index(q_map[g])
        if start:
            bt = bt.loc[bt.index >= pd.to_datetime(start)]
        if end:
            bt = bt.loc[bt.index <= pd.to_datetime(end)]
        bt_by_group[g] = bt

    def _norm_nav(s: pd.Series) -> pd.Series:
        s = s.dropna()
        if s.empty:
            return s
        return s / float(s.iloc[0])

    plt.figure()
    for g, bt in bt_by_group.items():
        if "nav" not in bt.columns:
            continue
        _norm_nav(bt["nav"]).plot(label=f"Q{g}")
    plt.title("NAV (Quintiles, normalized to 1)")
    plt.xlabel("Date")
    plt.ylabel("nav")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "nav_quintiles.png"), dpi=150)
    plt.close()

    plt.figure()
    for g, bt in bt_by_group.items():
        if "nav" not in bt.columns:
            continue
        nav = _norm_nav(bt["nav"])
        if nav.empty:
            continue
        dd = nav / nav.cummax() - 1.0
        dd.plot(label=f"Q{g}")
    plt.title("Drawdown (Quintiles)")
    plt.xlabel("Date")
    plt.ylabel("drawdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "drawdown_quintiles.png"), dpi=150)
    plt.close()

    plt.figure()
    for g, bt in bt_by_group.items():
        if "nav" not in bt.columns:
            continue
        nav = _norm_nav(bt["nav"])
        if nav.empty:
            continue
        np.log(nav).plot(label=f"Q{g}")
    plt.title("log(NAV) (Quintiles)")
    plt.xlabel("Date")
    plt.ylabel("log(nav)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "log_nav_quintiles.png"), dpi=150)
    plt.close()

    rows = []
    for g, bt in bt_by_group.items():
        if ("nav" not in bt.columns) or ("port_ret_net" not in bt.columns):
            continue
        rep = perf_report(bt, annualization=annualization)
        rep["Group"] = f"Q{g}"
        rows.append(rep)

    if ls_path and os.path.exists(ls_path):
        bt_ls = read_bt_csv_with_date_index(ls_path)
        if start:
            bt_ls = bt_ls.loc[bt_ls.index >= pd.to_datetime(start)]
        if end:
            bt_ls = bt_ls.loc[bt_ls.index <= pd.to_datetime(end)]

        if ("nav" in bt_ls.columns) and ("port_ret_net" in bt_ls.columns):
            rep_ls = perf_report(bt_ls, annualization=annualization)
            rep_ls["Group"] = "LS"
            rows.append(rep_ls)

            plt.figure()
            _norm_nav(bt_ls["nav"]).plot()
            plt.title("NAV (Long-Short, normalized to 1)")
            plt.xlabel("Date")
            plt.ylabel("nav")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "nav_long_short.png"), dpi=150)
            plt.close()

    if rows:
        perf_df = pd.DataFrame(rows).set_index("Group")
        order = [f"Q{i}" for i in sorted(bt_by_group.keys())]
        if "LS" in perf_df.index:
            order.append("LS")
        perf_df = perf_df.loc[[idx for idx in order if idx in perf_df.index]]
        perf_df.to_csv(os.path.join(out_dir, "perf_quintiles.csv"), encoding="utf-8-sig")

    for g, bt in bt_by_group.items():
        if "nav" not in bt.columns:
            continue
        mret = monthly_returns(bt)
        mret.to_csv(os.path.join(out_dir, f"monthly_returns_Q{g}.csv"), encoding="utf-8-sig")

    if ls_path and os.path.exists(ls_path):
        bt_ls = read_bt_csv_with_date_index(ls_path)
        if start:
            bt_ls = bt_ls.loc[bt_ls.index >= pd.to_datetime(start)]
        if end:
            bt_ls = bt_ls.loc[bt_ls.index <= pd.to_datetime(end)]
        if "nav" in bt_ls.columns:
            mret_ls = monthly_returns(bt_ls)
            mret_ls.to_csv(os.path.join(out_dir, "monthly_returns_LS.csv"), encoding="utf-8-sig")

    if summ_path and os.path.exists(summ_path):
        try:
            df_s = pd.read_csv(summ_path)
            df_s.to_csv(os.path.join(out_dir, "quintile_summary.csv"), index=False, encoding="utf-8-sig")
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="analysis_out")
    ap.add_argument("--annualization", type=int, default=252)
    ap.add_argument("--start", type=str, default="")
    ap.add_argument("--end", type=str, default="")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = read_bt_csv_with_date_index(args.in_csv)

    if args.start:
        df = df.loc[df.index >= pd.to_datetime(args.start)]
    if args.end:
        df = df.loc[df.index <= pd.to_datetime(args.end)]

    rep = perf_report(df, annualization=args.annualization)
    print("\n===== Performance =====")
    for k, v in rep.items():
        print(f"{k:>16s}: {v: .6f}")

    pd.DataFrame([rep]).to_csv(
        os.path.join(args.out_dir, "perf_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    ydf = yearly_perf_table(df, annualization=args.annualization)
    if not ydf.empty:
        ydf.to_csv(os.path.join(args.out_dir, "perf_by_year.csv"), encoding="utf-8-sig")

    dd_dur = drawdown_durations(df["nav"])
    dd_dur.to_csv(os.path.join(args.out_dir, "drawdown_durations.csv"), index=False, encoding="utf-8-sig")
    if not dd_dur.empty:
        stats = {
            "Count": int(len(dd_dur)),
            "MaxLength": float(dd_dur["length"].max()),
            "MeanLength": float(dd_dur["length"].mean()),
            "MedianLength": float(dd_dur["length"].median()),
        }
    else:
        stats = {"Count": 0, "MaxLength": np.nan, "MeanLength": np.nan, "MedianLength": np.nan}
    pd.DataFrame([stats]).to_csv(os.path.join(args.out_dir, "drawdown_duration_stats.csv"), index=False, encoding="utf-8-sig")

    rr = rolling_1y_returns(df, annualization=args.annualization)
    rr.to_csv(os.path.join(args.out_dir, "rolling_1y_return.csv"), encoding="utf-8-sig")
    plot_rolling_1y(rr, args.out_dir)

    # 保存月度收益
    mret = monthly_returns(df)
    mret.to_csv(os.path.join(args.out_dir, "monthly_returns.csv"), encoding="utf-8-sig")

    # 出图
    plot_nav_and_dd(df, args.out_dir)
    plot_turnover_cost(df, args.out_dir)

    analyze_quintiles(
        in_csv=args.in_csv,
        out_dir=args.out_dir,
        annualization=args.annualization,
        start=args.start,
        end=args.end,
    )

    q_perf_path = os.path.join(args.out_dir, "perf_quintiles.csv")
    q_sum_path = os.path.join(args.out_dir, "quintile_summary.csv")

    qdf = None
    if os.path.exists(q_perf_path):
        qdf = pd.read_csv(q_perf_path, index_col=0)
    elif os.path.exists(q_sum_path):
        tmp = pd.read_csv(q_sum_path)
        if "Group" in tmp.columns:
            qdf = tmp.set_index("Group")
        else:
            qdf = tmp.set_index(tmp.columns[0])

    if qdf is not None and len(qdf) > 0:
        print("\n===== Performance (QUINTILES / LS) =====")
        print(qdf.to_string(float_format=lambda x: f"{x: .6f}"))

    print(f"\n[INFO] outputs saved under: {args.out_dir}")
    print("[INFO] figures: nav.png, log_nav.png, drawdown.png, turnover.png, cost_rate.png")
    print("[INFO] figures+: rolling_1y_return.png, rolling_1y_return_hist.png")
    print("[INFO] table: monthly_returns.csv")
    print("[INFO] table+: perf_summary.csv, perf_by_year.csv, drawdown_durations.csv, drawdown_duration_stats.csv, rolling_1y_return.csv")


if __name__ == "__main__":
    main()
