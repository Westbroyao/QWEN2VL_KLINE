以下所有代码在根目录 QWEN2VL_KLINE 运行

0. 拉取仓库：bash git_pull.sh
1. 安装依赖：pip install -r requirements.txt
2. python src/prepare_data_0_sliding_windows.py
3. python src/prepare_data_1_reward_tag.py
4. python src/prepare_data_2_resampling.py
5. 画K线图：python src/make_plots.py
6. 画测试集K线图：mkdir -p data_test/images ｜ python src/make_plots_test.py
7. 建立train_val数据集：python src/build_dataset.py
9. 建立test数据集：python src/build_dataset_test.py
10. 微调模型：bash run_train.sh  # 可以选择根据实验目的修改训练参数
11. 查看结果：bash eval.sh
12. 配置git：bash git_config.sh
13. 上传仓库：bash git_push.sh   # 根据需要设置分支