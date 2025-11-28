#!/usr/bin/env bash
set -euo pipefail

# 在项目根目录运行这个脚本
# 创建本项目用到的所有目录（如果已存在则忽略）

mkdir -p data_raw
mkdir -p data_proc
mkdir -p data_images/kline_windows
mkdir -p data_train
mkdir -p experiments
mkdir -p results
mkdir -p outputs

echo "All directories created (or already existed)."