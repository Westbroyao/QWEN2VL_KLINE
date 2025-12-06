import os
import numpy as np
import pandas as pd
import mplfinance as mpf
import matplotlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Tuple

# 用无界面后端，避免多进程里卡在 GUI
matplotlib.use("Agg")

NPZ_PATH = "data_proc/windows_30_5_multi_with_labels_train_val_resampling.npz"  # 或 btc_windows_30_5.npz
OUT_DIR = "data_images/kline_windows"
MAX_PLOTS = 100000   # 最多生成多少张图
FIGSIZE = (4, 3)   # 图像大小（英寸）

# 并行进程数：默认用 CPU 数的一半，你也可以手动改，比如 4
NUM_WORKERS = max(1, (os.cpu_count() or 4) // 2)


def load_windows(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]                    # (n_samples, 30, 5)
    time_index = data["time_index"]  # (n_samples,)
    labels_str = data["labels_str"] if "labels_str" in data.files else None
    return X, time_index, labels_str


def window_to_dataframe(x_window: np.ndarray, start_time) -> pd.DataFrame:
    """
    x_window: shape (30, 5) = [open, high, low, close, volume]
    start_time: 这个窗口的第一天日期
    """
    index = pd.date_range(start=start_time, periods=len(x_window), freq="D")
    df = pd.DataFrame(
        x_window,
        index=index,
        columns=["Open", "High", "Low", "Close", "Volume"],
    )
    return df


def plot_one_window(args: Tuple[int, np.ndarray, any, Optional[str], str]):
    """
    单个进程里画一张图并保存到磁盘。

    args: (idx, x_window, start_time, label_str, out_dir)
    """
    idx, x_window, start_time, label_str, out_dir = args

    df_win = window_to_dataframe(x_window, start_time)

    # 文件名里带上编号和标签（如果有）
    label_part = f"_{label_str}" if label_str is not None else ""
    fname = f"window_{idx:05d}{label_part}.png"
    out_path = os.path.join(out_dir, fname)
    # market_colors
    mc = mpf.make_marketcolors(
        up='r',    # 上涨：红
        down='g',  # 下跌：绿
        inherit=True
    )
    # 设置 style
    s = mpf.make_mpf_style(marketcolors=mc)
    # 画 K 线图并保存
    # returnfig=True 方便我们显式关闭 figure，避免内存泄漏
    fig, axes = mpf.plot(
        df_win,
        type="candle",
        volume=True,
        style=s,
        ylabel="Price",
        ylabel_lower="Volume",
        figsize=FIGSIZE,
        figratio=(3, 1),
        figscale=1.0,
        tight_layout=True,
        returnfig=True,
    )
    for ax in axes:          # axes 是 [price_ax, volume_ax]
        ax.set_xticklabels([])
        ax.set_xlabel("")
    fig.savefig(fname=out_path, dpi=120, bbox_inches="tight")

    # 显式关闭 figure，防止每个进程里堆太多图像对象
    import matplotlib.pyplot as plt
    plt.close(fig)

    return idx  # 返回索引，主进程用来打印进度


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    X, time_index, labels_str = load_windows(NPZ_PATH)

    n_samples = X.shape[0]
    n_plots = min(n_samples, MAX_PLOTS)
    print(f"总样本数: {n_samples}, 将绘制前 {n_plots} 个窗口。")
    print(f"使用并行进程数: {NUM_WORKERS}")

    # 准备任务列表：每个任务包含画图所需的全部信息
    tasks = []
    for i in range(n_plots):
        x_window = X[i]
        start_time = time_index[i]
        label_i = labels_str[i] if labels_str is not None else None
        tasks.append((i, x_window, start_time, label_i, OUT_DIR))

    # 并行执行
    finished = 0
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(plot_one_window, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                _ = fut.result()
            except Exception as e:
                print(f"[警告] 绘制样本 {idx} 时出错: {e}")
            finished += 1
            if finished % 10 == 0 or finished == n_plots:
                print(f"已生成 {finished}/{n_plots} 张图")

    print(f"所有图像已保存到: {OUT_DIR}/")


if __name__ == "__main__":
    main()