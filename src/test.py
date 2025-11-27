import numpy as np

data = np.load("./data_raw/btc_windows_30_5_with_labels.npz", allow_pickle=True)  # 默认是只读模式

# 看看里面有哪些键
print(data.files)   # 例如: ['X', 'y', 'time_index']

# 分别取出
X = data["X"]
y = data["y"]
time_index = data["time_index"]
data["labels_int"]
data.files