以下所有代码在根目录 QWEN2VL_KLINE 运行

0. 拉取仓库：bash git_pull.sh
1. 安装依赖：pip install -r requirements.txt
2. 画K线图：python src/make_plots.py
3. 实验设计：修改 src/build_dataset.py 并运行 python src/build_dataset.py
4. 微调模型：bash run_train.sh  # 可以选择根据实验目的修改训练参数
5. 查看结果：bash eval.sh
6. 配置git：bash git_config.sh
7. 上传仓库：bash git_push.sh   # 根据需要设置分支


python src/prepare_data_0_sliding_windows.py
python src/prepare_data_1_reward_tag.py
python src/prepare_data_2_resampling.py