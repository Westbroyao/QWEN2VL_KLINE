import os
import glob
import pandas as pd
import numpy as np

# 配置参数：输入窗口长度 & 输出窗口长度
INPUT_WINDOW = 30   # 过去30天
OUTPUT_WINDOW = 5   # 未来5天

# 多文件模式下：原始 CSV 所在目录 & 输出 npz 路径
CSV_DIR = "data_raw"                     # 里面放很多 .csv
OUT_NPZ_PATH = "data_proc/windows_30_5_multi.npz"


def load_btcusdt_csv(csv_path: str) -> pd.DataFrame:
    """
    从CSV读取日K数据，并按时间排序，做基础清洗。

    这里沿用你原来的BTCUSDT列名格式：
    open_time,open,high,low,close,volume,...

    如果以后是A股的CSV，只要把这里的列名适配一下就行。
    """
    df = pd.read_csv(csv_path)

    # 转时间并按时间排序（重要：保证时间顺序）
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date").reset_index(drop=True)

    # 只保留我们关心的列
    df_features = df[["trade_date", "open", "high", "low", "close", "volume"]].copy()
    return df_features


# def build_windows_for_df(df: pd.DataFrame,
#                          input_window: int = INPUT_WINDOW,
#                          output_window: int = OUTPUT_WINDOW):
#     """
#     对【单个】时间序列 DataFrame 构造滑动窗口。

#     返回：z
#     - X: (n_samples, input_window, n_features)
#     - y: (n_samples, output_window)
#     - time_index: (n_samples,) 每个样本的预测起点时间
#     """
#     feature_cols = ["open", "high", "low", "close", "volume"]
#     data = df[feature_cols].values.astype(float)

#     close = df["close"].values.astype(float)

#     n_total = len(df)
#     n_samples = n_total - input_window - output_window + 1
#     if n_samples <= 0:
#         raise ValueError(
#             f"数据太少，无法构造窗口：总长度={n_total}, "
#             f"需要至少 {input_window + output_window} 条。"
#         )

#     X_list = []
#     y_list = []
#     time_index = []

#     for start in range(n_samples):
#         end_input = start + input_window
#         end_output = end_input + output_window

#         x_window = data[start:end_input, :]       # (input_window, n_features)
#         y_window = close[end_input:end_output]    # (output_window,)

#         X_list.append(x_window)
#         y_list.append(y_window)
#         time_index.append(df["trade_date"].iloc[end_input])

#     X = np.stack(X_list, axis=0)
#     y = np.stack(y_list, axis=0)
#     t = np.array(time_index)

#     return X, y, t


def build_windows_for_df(df: pd.DataFrame,
                         input_window: int = INPUT_WINDOW,
                         output_window: int = OUTPUT_WINDOW,
                         slide_step: int = 15):
    """
    对【单个】时间序列 DataFrame 构造滑动窗口，设置每15天滑动一次。

    返回：z
    - X: (n_samples, input_window, n_features)
    - y: (n_samples, output_window)
    - time_index: (n_samples,) 每个样本的预测起点时间
    """
    feature_cols = ["open", "high", "low", "close", "volume"]
    data = df[feature_cols].values.astype(float)

    close = df["close"].values.astype(float)

    n_total = len(df)
    n_samples = (n_total - input_window - output_window) // slide_step + 1  # 每15天滑动一次
    if n_samples <= 0:
        raise ValueError(
            f"数据太少，无法构造窗口：总长度={n_total}, "
            f"需要至少 {input_window + output_window} 条。"
        )

    X_list = []
    y_list = []
    time_index = []

    for start in range(0, n_samples * slide_step, slide_step):
        end_input = start + input_window
        end_output = end_input + output_window

        # 窗口内的数据
        x_window = data[start:end_input, :]       # (input_window, n_features)
        y_window = close[end_input:end_output]    # (output_window,)

        X_list.append(x_window)
        y_list.append(y_window)
        time_index.append(df["trade_date"].iloc[end_input])

    X = np.stack(X_list, axis=0)
    y = np.stack(y_list, axis=0)
    t = np.array(time_index)

    return X, y, t

def build_windows_from_multi_csv(
        csv_dir: str,
        pattern: str = "*.csv",
        input_window: int = INPUT_WINDOW,
        output_window: int = OUTPUT_WINDOW):
    """
    遍历多个CSV文件，对每个文件做滑窗，然后把所有样本拼在一起。

    额外返回一个 symbols 数组，记录每条样本来自哪个文件（方便之后分析）。
    """
    csv_paths = sorted(glob.glob(os.path.join(csv_dir, pattern)))
    if not csv_paths:
        raise FileNotFoundError(f"在目录 {csv_dir} 下没有找到匹配 {pattern} 的CSV文件")

    all_X = []
    all_y = []
    all_t = []
    all_symbols = []

    total_files = len(csv_paths)
    total_samples = 0

    print(f"发现 {total_files} 个CSV文件，开始构造滑动窗口...")

    for idx, csv_path in enumerate(csv_paths, start=1):
        try:
            df = load_btcusdt_csv(csv_path)
            X, y, t = build_windows_for_df(df, input_window, output_window, 15)
        except Exception as e:
            print(f"[{idx:04d}/{total_files}] 处理文件 {csv_path} 失败，跳过。原因: {e}")
            continue

        n_samples = X.shape[0]
        total_samples += n_samples

        all_X.append(X)
        all_y.append(y)
        all_t.append(t)

        # 从文件名中提取一个symbol标签，例如 BTCUSDT 或 600000.SH
        fname = os.path.basename(csv_path)
        symbol = os.path.splitext(fname)[0]
        all_symbols.extend([symbol] * n_samples)

        print(f"[{idx:04d}/{total_files}] {fname}: 生成 {n_samples} 个样本。")

    if not all_X:
        raise RuntimeError("所有CSV都处理失败，没有生成任何样本。")

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    t_all = np.concatenate(all_t, axis=0)
    symbols_all = np.array(all_symbols)

    print(f"总共生成样本数: {total_samples}")
    print("X_all 形状:", X_all.shape)
    print("y_all 形状:", y_all.shape)
    print("time_index 示例:", t_all[:5])
    print("symbols 示例:", symbols_all[:5])

    return X_all, y_all, t_all, symbols_all


if __name__ == "__main__":
    os.makedirs("data_proc", exist_ok=True)

    # 单文件测试（保留，方便对比）
    # csv_path = "data_raw/BTCUSDT_binance_1d.csv"
    # df = load_btcusdt_csv(csv_path)
    # X, y, t = build_windows_for_df(df)
    # np.savez("data_proc/btc_windows_30_5.npz", X=X, y=y, time_index=t)

    # 多文件模式
    X_all, y_all, t_all, symbols_all = build_windows_from_multi_csv(
        csv_dir=CSV_DIR,
        pattern="*.csv",
        input_window=INPUT_WINDOW,
        output_window=OUTPUT_WINDOW,
    )

    out_path = OUT_NPZ_PATH
    np.savez(out_path,
             X=X_all,
             y=y_all,
             time_index=t_all,
             symbols=symbols_all)
    print(f"已保存到 {out_path}")