以下所有代码在根目录 QWEN2VL_KLINE 运行

0. bash git_pull.sh
1. pip install -r requirements.txt
2. python src/make_plots.py
3. 根据实验目的，修改 src/build_dataset.py 并运行 python src/build_dataset.py
4. bash run_train.sh  可以选择根据实验目的修改训练参数
5. bash eval.sh       查看结果
6. bash git_config.sh
7. bash git_push.sh   根据需要设置分支