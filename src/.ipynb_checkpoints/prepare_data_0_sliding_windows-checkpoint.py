import pandas as pd
import numpy as np


# 配置参数：输入窗口长度 & 输出窗口长度
INPUT_WINDOW = 30   # 过去30天
OUTPUT_WINDOW = 5   # 未来5天


def load_btcusdt_csv(csv_path: str) -> pd.DataFrame:
    """
    从CSV读取BTCUSDT日K数据，并按时间排序，做基础清洗。
    默认列名来自前面Binance脚本生成的CSV：
    open_time,open,high,low,close,volume,close_time,...
    """
    df = pd.read_csv(csv_path)

    # 转时间并按时间排序（重要：保证时间顺序）
    df["open_time"] = pd.to_datetime(df["open_time"])
    df = df.sort_values("open_time").reset_index(drop=True)

    # 只保留我们关心的列（你可以按需增减）
    # 这里用 open/high/low/close/volume 作为输入特征
    df_features = df[["open_time", "open", "high", "low", "close", "volume"]].copy()

    return df_features


def build_windows(df: pd.DataFrame,
                  input_window: int = INPUT_WINDOW,
                  output_window: int = OUTPUT_WINDOW):
    """
    用滑动窗口构造训练样本：
    - X: 形状 (n_samples, input_window, n_features)
    - y: 形状 (n_samples, output_window)
         这里的 y 用的是未来5天的收盘价（绝对价格）

    你后面可以把 y 改成收益率、相对价格等。
    """
    # 特征矩阵（去掉时间，只用数值特征）
    feature_cols = ["open", "high", "low", "close", "volume"]
    data = df[feature_cols].values.astype(float)

    # 目标只用 close
    close = df["close"].values.astype(float)

    n_total = len(df)
    n_samples = n_total - input_window - output_window + 1
    if n_samples <= 0:
        raise ValueError(
            f"数据太少，无法构造窗口：总长度={n_total}, "
            f"需要至少 {input_window + output_window} 条。"
        )

    X_list = []
    y_list = []
    time_index = []  # 可选：记录每个样本对应的“预测起点时间”

    for start in range(n_samples):
        # 输入窗口：[start, start + input_window)
        end_input = start + input_window
        # 输出窗口：[end_input, end_input + output_window)
        end_output = end_input + output_window

        x_window = data[start:end_input, :]           # (30, n_features)
        y_window = close[end_input:end_output]        # (5,)

        X_list.append(x_window)
        y_list.append(y_window)
        # 记录这个样本的“预测起点”（未来窗口第一天的时间）
        time_index.append(df["open_time"].iloc[end_input])

    X = np.stack(X_list, axis=0)  # (n_samples, 30, n_features)
    y = np.stack(y_list, axis=0)  # (n_samples, 5)

    return X, y, np.array(time_index)


if __name__ == "__main__":
    csv_path = "data_raw/BTCUSDT_binance_1d.csv"

    df = load_btcusdt_csv(csv_path)
    print("原始数据行数:", len(df))
    print(df.head())

    X, y, t = build_windows(df)

    print("X 形状:", X.shape)   # (n_samples, 30, 5)
    print("y 形状:", y.shape)   # (n_samples, 5)
    print("time_index 示例:", t[:5])

    # 你也可以把结果保存起来，方便后续训练直接加载
    np.savez("data_proc/btc_windows_30_5.npz", X=X, y=y, time_index=t)
    print("已保存到 data_proc/btc_windows_30_5.npz")