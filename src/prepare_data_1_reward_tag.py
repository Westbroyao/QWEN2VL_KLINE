import numpy as np

# 读取 npz
data = np.load("data_proc/windows_30_5_multi.npz", allow_pickle=True)
X = data["X"]              # (n_samples, 30, 5)
y = data["y"]              # (n_samples, 5)
time_index = data["time_index"]

print("X shape:", X.shape)
print("y shape:", y.shape)

# 参数
H = y.shape[1]        # 未来天数，这里应该是 5
eps = 0.005           # 阈值，比如 0.2% 平均收益      !!!!!可以改!!!!!

# 从 X 中取出历史窗口最后一天的收盘价 C_t
# X[..., 3] 是 close 特征（[open, high, low, close, volume]）
C_t = X[:, -1, 3]     # (n_samples,)

# 未来 5 天的价格 y: shape (n_samples, 5)
C_future = y          # (n_samples, H)

# 计算未来每一天相对 C_t 的收益率
# r_i_k = (C_{t+k} - C_t) / C_t
returns = (C_future - C_t[:, None]) / C_t[:, None]   # (n_samples, H)

# 未来5天平均收益率
r_avg = returns.mean(axis=1)   # (n_samples,)

# 按阈值离散成标签
labels_int = np.zeros_like(r_avg, dtype=int)
labels_int[r_avg > eps] = 1
labels_int[r_avg < -eps] = -1

# 也可以准备字符串标签，方便做可视化 / 写入 JSON
labels_str = np.empty_like(labels_int, dtype=object)
labels_str[labels_int == 1] = "up"
labels_str[labels_int == 0] = "flat"
labels_str[labels_int == -1] = "down"

print("标签分布：", {v: (labels_str == v).sum() for v in ["up", "flat", "down"]})

# 如果你希望保存新的 npz：
np.savez("data_proc/windows_30_5_multi_with_labels.npz",
         X=X,
         y=y,
         time_index=time_index,
         labels_int=labels_int,
         labels_str=labels_str)
print("已保存到 data_proc/windows_30_5_multi_with_labels.npz")