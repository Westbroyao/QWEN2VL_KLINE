## QWEN2VL_KLINE

用 Qwen2-VL-7B-Instruct 在 BTCUSDT 日 K 线图 上做指令微调，让大模型学会：

输入：过去 30 个交易日的 BTCUSDT 日 K 线图
输出：未来 5 日整体方向标签（up / flat / down）+ 简短中文理由 reason（JSON 格式）

这是一个从数据抓取 → 特征构造 → K 线图生成 → JSONL 构造 → QLoRA 微调 的完整示例项目，适合作为课程期末作业或后续研究的基线。

⸻

## 1. 项目结构

QWEN2VL_KLINE/
  data_raw/                         # 原始数据（Binance 日K）
    BTCUSDT_binance_1d.csv

  data_proc/                        # 预处理后的中间数据
    btc_windows_30_5.npz
    btc_windows_30_5_with_labels.npz

  data_images/
    kline_windows/                  # 按窗口生成的 K 线图 (window_00000_up.png ...)

  data_train/                       # 给大模型用的最终训练数据
    train.jsonl
    val.jsonl

  experiments/                      # （预留）不同实验配置

  results/                          # 训练日志、评估结果
    train_YYYYMMDD_HHMMSS.log

  src/
    download_btc_binance.py             # 从 Binance 拉取 BTCUSDT 日K
    prepare_data_0_sliding_windows.py   # 从CSV构造(30,5)滑动窗口并保存为npz
    prepare_data_1_reward_tag.py        # 根据未来5日平均收益生成 up/flat/down 标签
    make_plots.py                       # 从窗口数据批量生成K线图
    build_dataset_label_only.py         # 只带 label 的 JSONL（备用）
    build_dataset_label_and_reason.py   # 带 label + reason 的 JSONL（主用）
    train_qwen2vl.py                    # Qwen2-VL QLoRA 微调脚本
    eval.py                             # 评估脚本（预测 vs 真实标签/价格）
    test.py                             # 临时测试用

  run_train_qwen2vl.sh              # 一键启动训练的 bash 脚本
  .gitignore
  README.md


⸻

## 2. 环境准备

建议使用 Python 3.10/3.11。

conda create -n qwen_kline python=3.11 -y
conda activate qwen_kline

pip install --upgrade pip

# 基本依赖
pip install numpy pandas matplotlib mplfinance requests

# HF + 大模型相关
pip install "datasets>=2.20.0" "transformers>=4.44.0" accelerate peft bitsandbytes qwen-vl-utils

训练时默认使用 Qwen/Qwen2-VL-7B-Instruct，通过 4bit QLoRA + LoRA 进行参数高效微调。

⸻

## 3. 数据流程（Data Pipeline）

# 3.1 从 Binance 下载 BTCUSDT 日 K

cd src
python download_btc_binance.py

输出：
	•	data_raw/BTCUSDT_binance_1d.csv
包含列：open_time, open, high, low, close, volume, ...。

⸻

# 3.2 构造滑动窗口（30 天输入 + 5 天未来）

python prepare_data_0_sliding_windows.py

逻辑简述：
	•	对时间序列做滑窗：
	•	输入窗口：过去 30 日 K (X)，特征为 [open, high, low, close, volume]
	•	输出窗口：接下来 5 日的 close (y)
	•	保存为：
	•	data_proc/btc_windows_30_5.npz
	•	X: (n_samples, 30, 5)
	•	y: (n_samples, 5)
	•	time_index: 每个样本未来窗口第一天的时间

⸻

# 3.3 根据未来窗口生成标签（up / flat / down）

python prepare_data_1_reward_tag.py

简要规则（可在脚本中调整）：
	•	对每个样本 i，取历史最后一天收盘价 C_t 和未来 5 日价格 C_{t+1..t+5}
	•	计算平均收益：
r_{\text{avg}} = \frac{1}{5}\sum_{k=1}^5 \frac{C_{t+k}-C_t}{C_t}
	•	设阈值 \varepsilon（例如 0.2%）：
	•	label = "up" if r_{\text{avg}} > \varepsilon
	•	label = "down" if r_{\text{avg}} < -\varepsilon
	•	label = "flat" otherwise

输出：
	•	data_proc/btc_windows_30_5_with_labels.npz
在原结构基础上增加 labels_int / labels_str。

⸻

# 3.4 生成 K 线图

python make_plots.py

	•	对每个 30 日窗口生成一张蜡烛图（带成交量），使用 mplfinance。
	•	文件名格式示例：window_00000_up.png。
	•	输出目录：data_images/kline_windows/。

⸻

# 3.5 构造微调数据 JSONL

主用脚本（带 label + reason 的指令数据）：

python build_dataset_label_and_reason.py

每条样本格式（简化示例）：

{
  "messages": [
    {
      "role": "system",
      "content": "你是一名量化分析师……（略）"
    },
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "kline_windows/window_00000_up.png"},
        {"type": "text", "text": "下面的图片是BTCUSDT过去30个交易日的日K线图……"}
      ]
    },
    {
      "role": "assistant",
      "content": "{\"label\": \"up\", \"reason\": \"过去30个交易日整体呈现明显的上涨趋势，最近一周略有回落，成交量放大……在此背景下，多头力量相对占优，因此判断未来5个交易日上涨的概率较大。\"}"
    }
  ]
}

reason 由规则自动生成，基于当前窗口的：
	•	30 日总收益
	•	最近 5 日收益
	•	日收益波动率
	•	最近 5 日成交量 vs 前 20 日成交量

输出：
	•	data_train/train.jsonl
	•	data_train/val.jsonl（按 8:2 随机划分）。

⸻

## 4. 训练 Qwen2-VL（QLoRA）

# 4.1 一键脚本（推荐）

在项目根目录：

conda activate qwen_kline  # 激活环境

chmod +x run_train_qwen2vl.sh    # 第一次需要
./run_train_qwen2vl.sh

脚本做的事情：
	•	调用 python src/train_qwen2vl.py
	•	使用参数：
	•	batch_size=1，grad_accum=4（等效 batch=4）
	•	num_epochs=2（可在脚本中调整）
	•	learning_rate=1e-4
	•	日志写入 results/train_YYYYMMDD_HHMMSS.log
	•	LoRA 权重和 processor 保存到 outputs/qwen2vl_kline_lora/（由脚本中 OUTPUT_DIR 控制）。

# 4.2 训练脚本简介：src/train_qwen2vl.py
	•	模型：Qwen/Qwen2-VL-7B-Instruct
	•	量化：BitsAndBytesConfig(load_in_4bit=True, ...)（QLoRA）
	•	LoRA：对注意力/MLP 的 q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj 做 r=64 的 LoRA。
	•	数据：
	•	使用 datasets.load_dataset("json") 读取 train.jsonl / val.jsonl
	•	qwen_vl_utils.process_vision_info() + AutoProcessor 处理图像 & 文本
	•	采用 chat template，把 messages 拼成单一序列
	•	优化器 & Trainer：
	•	使用 transformers.Trainer 训练因果语言模型
	•	当前简单做法：对整段序列计算 loss（包含 system/user/assistant），后续可以进一步只在 assistant 段上打 mask。

⸻

## 5. 评估与可视化

src/eval.py（可根据实际情况扩展），常见评估思路：
	•	方向分类准确率
	•	模型生成 JSON → 解析 label_pred
	•	与真实 labels_str 对比，计算 accuracy / F1。
	•	未来价格路径误差（如果后续把 y 也纳入输出）：
	•	MSE / MAE / 日度方向预测正确率。
	•	Case Study
	•	选若干窗口，画出未来真实 5 日价格 vs 模型预测走势
	•	对比模型的 reason 是否跟走势描述一致。

建议将评估结果存到 results/，例如：

results/
  metrics.json
  curves/train_val_loss.png
  case_study/sample_00001_true_vs_pred.png


⸻

## 6. 在 RunPod 上复现实验（简要）
	1.	把代码推到 GitHub（.gitignore 中忽略：data_raw/ data_proc/ data_images/ data_train/ outputs/ results/）。
	2.	在 RunPod 创建 GPU Pod（推荐 Community Cloud，显存 ≥ 24GB），打开 Web Terminal：

cd /workspace
git clone https://github.com/<yourname>/QWEN2VL_KLINE.git
cd QWEN2VL_KLINE

python -m venv venv
source venv/bin/activate

# 安装依赖（按第 2 节）
pip install --upgrade pip
pip install numpy pandas matplotlib mplfinance requests
pip install "datasets>=2.20.0" "transformers>=4.44.0" accelerate peft bitsandbytes qwen-vl-utils

# 在 Pod 上重新生成数据
cd src
python download_btc_binance.py
python prepare_data_0_sliding_windows.py
python prepare_data_1_reward_tag.py
python make_plots.py
python build_dataset_label_and_reason.py
cd ..

# 开始训练
chmod +x run_train_qwen2vl.sh
./run_train_qwen2vl.sh


⸻

## 7. TODO / 可扩展方向
	•	在 loss 计算中只对 assistant 输出位置打 label mask；
	•	把未来 5 日价格序列也纳入 JSON 输出，做多任务（方向分类 + 回归）；
	•	引入更多特征（技术指标、order flow、不同比例窗口等）；
	•	从单一 BTCUSDT 扩展到多币种联合训练；
	•	对比 Qwen2-VL 微调前后的性能提升，写成完整报告。

⸻

## 8. 免责声明

本项目仅用于课程作业和研究用途，不构成任何投资建议。
加密货币市场波动和噪音极大，模型在历史数据上的预测能力并不意味着对真实市场未来走势有所保证。