from modelscope import snapshot_download

model_dir = snapshot_download(
    'qwen/Qwen2-VL-7B-Instruct',
    cache_dir='/autodl-tmp/models',   # 改成你有空间的目录
    # revision='v1.0.0'                 # 版本号按页面上为准，也可以先不写
)
print(model_dir)
