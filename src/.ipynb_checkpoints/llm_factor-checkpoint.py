import os
from typing import List
import pandas as pd
import mplfinance as mpf

# 保证 FIGSIZE 定义和你原来的一致，比如：
FIGSIZE = (4, 3)  # 如果你项目里已经有就不用再定义

def df_window_from_csv_slice(
    win: pd.DataFrame,
    date_col: str,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
) -> pd.DataFrame:
    """
    把 CSV 中某个窗口的 DataFrame 切片 win
    转成 mplfinance 所需格式：
      index = DatetimeIndex
      columns = ["Open","High","Low","Close","Volume"]
    """
    df_win = win[[date_col, open_col, high_col, low_col, close_col, volume_col]].copy()
    df_win[date_col] = pd.to_datetime(df_win[date_col])
    df_win = df_win.set_index(date_col)
    df_win.columns = ["Open", "High", "Low", "Close", "Volume"]
    return df_win


def plot_window_mpf(
    df_win: pd.DataFrame,
    out_path: str,
    figsize=FIGSIZE,
):
    """
    用和你原来 plot_one_window 完全一致的风格画一张 K 线图。
    """
    # market_colors：红涨绿跌
    mc = mpf.make_marketcolors(
        up='r',
        down='g',
        inherit=True,
    )
    s = mpf.make_mpf_style(marketcolors=mc)

    fig, axes = mpf.plot(
        df_win,
        type="candle",
        volume=True,
        style=s,
        ylabel="Price",
        ylabel_lower="Volume",
        figsize=figsize,
        figratio=(3, 1),
        figscale=1.0,
        tight_layout=True,
        returnfig=True,
    )
    # 去掉 x 轴刻度和 label
    for ax in axes:
        ax.set_xticklabels([])
        ax.set_xlabel("")

    fig.savefig(fname=out_path, dpi=120, bbox_inches="tight")

    import matplotlib.pyplot as plt
    plt.close(fig)

import numpy as np
import glob

def generate_kline_images_from_many_csv_mpf(
    csv_paths: List[str],
    out_dir: str = "kline_images",
    symbol_col: str = "ts_code",
    date_col: str = "trade_date",
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
    window: int = 90,
    start_date: str = "2025-01-01",
    step: int = 5,        # 🔹 新增：滑动步长（单位：交易日），默认 5
):
    """
    多个 CSV 里生成 K 线窗口图片，画图风格与 plot_one_window 完全一致（mplfinance）。

    参数
    ----
    csv_paths : List[str]
        多个 CSV 文件路径。
    out_dir : str
        图片输出目录。
    window : int
        窗口长度（例如 90 表示 90 个交易日）。
    start_date : str
        只保留窗口最后一天 >= start_date 的样本。
    step : int
        滑动步长（end_idx 的步长），例如 5 表示每 5 个交易日取一个窗口。

    返回
    ----
    samples : List[dict]
        每个元素:
        {
          "symbol": str,
          "date":   pd.Timestamp,
          "image_path": "file:///abs/path/to/png"
        }
    """
    os.makedirs(out_dir, exist_ok=True)
    start_dt = pd.to_datetime(start_date)

    samples = []

    for csv_path in csv_paths:
        print(f"[generate] processing {csv_path}")
        df = pd.read_csv(csv_path)
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values([symbol_col, date_col])

        for symbol, g in df.groupby(symbol_col):
            g = g.sort_values(date_col).reset_index(drop=True)
            if len(g) < window:
                continue

            # 🔹 这里用 step 控制 end_idx 的滑动步长
            for end_idx in range(window - 1, len(g), step):
                win = g.iloc[end_idx - window + 1: end_idx + 1]
                date_end = win[date_col].iloc[-1]

                if date_end < start_dt:
                    continue

                # === 构造 df_win & 画图（风格对齐） ===
                df_win = df_window_from_csv_slice(
                    win,
                    date_col=date_col,
                    open_col=open_col,
                    high_col=high_col,
                    low_col=low_col,
                    close_col=close_col,
                    volume_col=volume_col,
                )

                # 文件名：symbol + 最后一天日期
                fname = f"{symbol}_{date_end.date()}.png".replace(":", "_").replace("/", "_")
                out_path = os.path.join(out_dir, fname)

                plot_window_mpf(df_win, out_path)  # 用和你原来一样的风格

                samples.append({
                    "symbol": symbol,
                    "date": date_end,
                    "image_path": "file://" + os.path.abspath(out_path),
                })

    print(f"[generate] total samples: {len(samples)}")
    return samples

csv_paths = glob.glob("autodl-tmp/QWEN2VL_KLINE/data_raw/*.csv")

samples = generate_kline_images_from_many_csv(
    csv_paths=csv_paths,
    out_dir="kline_images_2025plus",
    symbol_col="ts_code",      # 改成你的列名
    date_col="trade_date",    # 改成你的列名
    window=90,
    start_date="2025-01-01",
)
