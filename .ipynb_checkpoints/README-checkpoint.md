以下所有代码在根目录 QWEN2VL_KLINE 运行

0. 拉取仓库：bash git_pull.sh
1. 安装依赖：pip install -r requirements.txt

2. 终端上传csv： scp -i ~/.ssh/id_ed25519 -P 12097 \
  ~/data_raw_contents.tar.gz \
  root@connect.westc.gpuhub.com:~/autodl-tmp/QWEN2VL_KLINE/data_raw/

4. 解压缩：cd ~/autodl-tmp/QWEN2VL_KLINE/data_raw/
# 解压到当前目录
tar -xzf data_raw_contents.tar.gz
# 解压完可以把压缩包删掉
rm data_raw_contents.tar.gz
ls   # 看看文件是不是都在了

4. python src/prepare_data_0_sliding_windows.py
5. python src/prepare_data_1_reward_tag.py
6. python src/prepare_data_2_resampling.py
7. 画K线图：python src/make_plots.py
8. 画测试集K线图：mkdir -p data_test/images ｜ python src/make_plots_test.py
9. 建立train_val数据集：python src/build_dataset.py ｜ python src/build_dataset_classifier.py
10. 建立test数据集：python src/build_dataset_test.py | python src/build_dataset_test_classifier.py
11. 安装模型：python src/model_install.py
12. 微调模型：bash run_train.sh | bash run_train_classifier.sh # 可以选择根据实验目的修改训练参数
13. 查看结果：bash eval.sh ｜ bash eval_classifier.sh
14. 配置git：bash git_config.sh
15. 上传仓库：bash git_push.sh   # 根据需要设置分支
16. 下一轮实验：cd data_images; rm -r kline_windows; mkdir kline_windows; cd kline_windows; vi .gitignore;cd .. cd data_test; rm -r images