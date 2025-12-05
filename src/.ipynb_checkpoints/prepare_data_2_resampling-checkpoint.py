import numpy as np
import pandas as pd
import random
from typing import Tuple


# 加载原始数据
def load_npz_data(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]  # 输入数据
    y = data["labels_str"]  # 标签
    time_index = data["time_index"]  # 时间索引
    return X, y, time_index


# 下采样（平衡每个标签的样本数，统一为最少标签的数量）
def balance_classes(X, y):
    """
    对训练集进行下采样，使每个标签的样本数相等。
    - 将每个标签的样本数减少到最少标签的样本数
    """
    # 获取每个标签的索引
    unique_labels = np.unique(y)
    label_indices = {label: np.where(y == label)[0] for label in unique_labels}

    # 找到最少标签的样本数
    min_samples = min(len(indices) for indices in label_indices.values())
    print(f"最少标签样本数: {min_samples}")

    # 进行下采样，确保每个标签的样本数相等
    X_resampled = []
    y_resampled = []
    for label, indices in label_indices.items():
        sampled_indices = random.sample(list(indices), min_samples)  # 随机选取 min_samples 个样本
        X_resampled.append(X[sampled_indices])
        y_resampled.append(y[sampled_indices])

    # 合并所有标签的样本
    X_resampled = np.concatenate(X_resampled, axis=0)
    y_resampled = np.concatenate(y_resampled, axis=0)

    return X_resampled, y_resampled


# 划分数据集
def split_data(X: np.ndarray, y: np.ndarray, time_index: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    按时间分割数据集：2024年之前的数据作为训练集和验证集，2025年的数据作为测试集。
    """
    # 转换时间为datetime类型
    time_index = pd.to_datetime(time_index)

    # 找到分割点：2024年开始的数据和2025年数据
    train_val_mask = time_index < "2024-01-01"  # 训练集和验证集
    test_mask = time_index >= "2025-01-01"  # 测试集

    # 划分数据
    X_train_val = X[train_val_mask]
    y_train_val = y[train_val_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    

    return X_train_val, y_train_val, X_test, y_test


# 保存结果
def save_resampled_data(npz_path: str, X_resampled, y_resampled, time_index):
    np.savez(npz_path, X=X_resampled, labels_str=y_resampled, time_index=time_index)
    print(f"数据已保存至 {npz_path}")


def main():
    input_file = "data_proc/windows_30_5_multi_with_labels.npz"  # 输入文件
    output_train_val_file = "data_proc/windows_30_5_multi_with_labels_train_val_resampling.npz"  # 输出训练集和验证集文件
    output_test_file = "data_proc/windows_30_5_multi_with_labels_test_resampling.npz"  # 输出测试集文件
    
    X, y, time_index = load_npz_data(input_file)

    # 划分数据集：训练集/验证集/测试集
    X_train_val, y_train_val, X_test, y_test = split_data(X, y, time_index)

    # 下采样（平衡标签数量）
    X_train_val_resampled, y_train_val_resampled = balance_classes(X_train_val, y_train_val)

    # 保存结果
    save_resampled_data(output_train_val_file, X_train_val_resampled, y_train_val_resampled, time_index)
    save_resampled_data(output_test_file, X_test, y_test, time_index)

if __name__ == "__main__":
    main()