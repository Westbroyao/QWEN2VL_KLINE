import os
import numpy as np
import pandas as pd
import mplfinance as mpf


NPZ_PATH = "data_proc/btc_windows_30_5_with_labels.npz"  # 或 btc_windows_30_5.npz
OUT_DIR = "data_images/kline_windows"
MAX_PLOTS = 3000   # 最多生成多少张图，防止一次性太多
FIGSIZE = (6, 4)  # 图像大小（单位是英寸，6x4 不会太大）


def load_windows(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]             # (n_samples, 30, 5)
    time_index = data["time_index"]  # (n_samples,)
    # 可选标签，如果存在：
    labels_str = data["labels_str"] if "labels_str" in data.files else None
    return X, time_index, labels_str


def window_to_dataframe(x_window, start_time):
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


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    X, time_index, labels_str = load_windows(NPZ_PATH)

    n_samples = X.shape[0]
    n_plots = min(n_samples, MAX_PLOTS)
    print(f"总样本数: {n_samples}, 将绘制前 {n_plots} 个窗口。")

    for i in range(n_plots):
        x_window = X[i]  # (30, 5)
        # 构造 DataFrame（这里用虚拟索引，如果你想用真实日期可以自己改）
        df_win = window_to_dataframe(x_window, time_index[i])

        # 文件名里带上样本编号和标签（如果有）
        label_part = ""
        if labels_str is not None:
            label_part = f"_{labels_str[i]}"

        fname = f"window_{i:05d}{label_part}.png"
        out_path = os.path.join(OUT_DIR, fname)

        # 使用 mplfinance 画 K 线
        mpf.plot(
            df_win,
            type="candle",
            volume=True,
            style="yahoo",
            ylabel="Price",
            ylabel_lower="Volume",
            figsize=(4, 3),          # 整体更小
            figratio=(3, 1),         # 主图:副图 = 3:1，不会太空
            figscale=1.0,
            tight_layout=True,
            savefig=dict(fname=out_path, dpi=120, bbox_inches="tight"),
        )
        

        if (i + 1) % 10 == 0 or i == n_plots - 1:
            print(f"已生成 {i + 1}/{n_plots} 张图")

    print(f"所有图像已保存到: {OUT_DIR}/")


if __name__ == "__main__":
    main()