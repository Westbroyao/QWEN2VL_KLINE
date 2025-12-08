import numpy as np
import pandas as pd
import random
from typing import Tuple

# ================== 工具函数 ==================

def load_npz_data(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]              # (N, 30, 5)
    y = data["labels_str"]     # (N,)
    time_index = data["time_index"]  # (N,)
    return X, y, time_index


import numpy as np
import random

def balance_classes(X, y, t, sample_counts=None):
    """
    对 (X, y, t) 做下采样，使每个标签样本数达到指定数量，
    并保持三者一一对应。

    参数
    ----
    X : np.ndarray
        特征，shape = (N, ...)
    y : np.ndarray
        标签，shape = (N,)
    t : np.ndarray
        时间戳或其它附加信息，shape = (N,)
    sample_counts : array-like or None
        - 如果为 None：行为与原来相同，取各类中最少的样本数。
        - 如果为长度为 num_classes 的数组：
          按照 np.unique(y) 的顺序依次指定每个标签要采样的数量。

    返回
    ----
    X_res, y_res, t_res : 下采样后的 X, y, t
    """

    y = np.asarray(y)
    t = np.asarray(t)

    unique_labels = np.unique(y)
    label_indices = {label: np.where(y == label)[0] for label in unique_labels}

    # 决定每个标签目标采样量
    if sample_counts is None:
        # 原逻辑：用最少的那一类做下采样
        min_samples = min(len(idx) for idx in label_indices.values())
        print(f"未指定 sample_counts，使用最少标签样本数: {min_samples}")
        target_counts = {label: min_samples for label in unique_labels}
    else:
        sample_counts = np.asarray(sample_counts, dtype=int)
        if sample_counts.shape[0] != len(unique_labels):
            raise ValueError(
                f"sample_counts 长度为 {sample_counts.shape[0]}，"
                f"但当前共有 {len(unique_labels)} 个标签: {unique_labels}"
            )
        target_counts = {
            label: int(n)
            for label, n in zip(unique_labels, sample_counts)
        }
        print("标签顺序及目标采样量：")
        for label in unique_labels:
            print(f"  label={label}, target={target_counts[label]}, available={len(label_indices[label])}")

    X_res, y_res, t_res = [], [], []

    for label, indices in label_indices.items():
        n_target = target_counts[label]
        n_avail = len(indices)

        if n_target > n_avail:
            raise ValueError(
                f"标签 {label} 只有 {n_avail} 个样本，但你要求采样 {n_target} 个；"
                f"如需过采样请改成有放回采样（random.choices 或 np.random.choice(replace=True))."
            )

        sampled_indices = random.sample(list(indices), n_target)
        X_res.append(X[sampled_indices])
        y_res.append(y[sampled_indices])
        t_res.append(t[sampled_indices])

    X_res = np.concatenate(X_res, axis=0)
    y_res = np.concatenate(y_res, axis=0)
    t_res = np.concatenate(t_res, axis=0)

    # 可选：最后整体打乱一下，避免同一类挤在一起
    perm = np.random.permutation(len(y_res))
    X_res = X_res[perm]
    y_res = y_res[perm]
    t_res = t_res[perm]

    return X_res, y_res, t_res



def split_data(
    X: np.ndarray,
    y: np.ndarray,
    t: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    按时间分割数据：
    - 2024-01-01 之前：train+val
    - 2025-01-01 及之后：test
    （中间 2024 年那一段你现在是直接丢掉，如果以后要用可以再细分）
    """
    t_dt = pd.to_datetime(t)

    train_val_mask = t_dt < "2024-01-01"
    test_mask = t_dt >= "2025-01-01"

    # 这部份的数据结构很有趣，我们本来是想把 X、y 和 t 打包起来的。不过因为tensor包含了指标的映射信息，所以我们不用打包，独立的写每个张量也可以恢复打包的信息。
    # 当然也可以在 dim = 1 多加一个维度表示打包信息。不过这样的 tensor 就不是“矩阵”了。这里的矩阵指：每个指标最大值不随其他指标的值变化。
    X_train_val = X[train_val_mask]
    y_train_val = y[train_val_mask]
    t_train_val = t[train_val_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]
    t_test = t[test_mask]

    return X_train_val, y_train_val, t_train_val, X_test, y_test, t_test

def shuffle_trun_dataset(
    X: np.ndarray,
    y: np.ndarray,
    time_index: np.ndarray,
    trun: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    同步打乱 (X, y, time_index)，保持三者一一对应。
    只在第 0 维（样本维）上打乱。
    """
    assert len(X) == len(y) == len(time_index), "长度不一致，无法打乱"

    perm = np.random.permutation(len(y))  # 生成一个随机排列
    perm_trun = perm[:trun]
    X_shuffled_trun = X[perm_trun]
    y_shuffled_trun = y[perm_trun]
    t_shuffled_trun = time_index[perm_trun]
    return X_shuffled_trun, y_shuffled_trun, t_shuffled_trun

def save_resampled_data(npz_path: str, X, y, t):
    np.savez(npz_path, X=X, labels_str=y, time_index=t)
    print(
        f"数据已保存至 {npz_path} | X.shape = {X.shape}, "
        f"label 分布 = {dict(zip(*np.unique(y, return_counts=True)))}"
    )

# ================== 主流程 ==================

def main():
    input_file = "data_proc/windows_30_5_multi_with_labels.npz"
    output_train_val_file = "data_proc/windows_30_5_multi_with_labels_train_val_resampling.npz"
    output_test_file = "data_proc/windows_30_5_multi_with_labels_test_resampling.npz"
    
    X, y, t = load_npz_data(input_file)

    # 1. 按时间切 train_val / test（时间和 X, y 同步切）
    X_trv, y_trv, t_trv, X_test, y_test, t_test = split_data(X, y, t)

    # 2. 只对 train+val 做下采样（X, y, t 一起动）
    X_trv_bal, y_trv_bal, t_trv_bal = balance_classes(X_trv, y_trv, t_trv, sample_counts=[95220, 21252, 87187])

    # 2.5 
    X_test_trun, y_test_trun, t_test_trun = shuffle_trun_dataset(X_test, y_test, t_test, trun = 1000)
    
    # 3. 保存：train+val（下采样后）和 test（原样）
    save_resampled_data(output_train_val_file, X_trv_bal, y_trv_bal, t_trv_bal)
    save_resampled_data(output_test_file, X_test_trun, y_test_trun, t_test_trun)


if __name__ == "__main__":
    main()