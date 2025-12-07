import os
import time
import random
from datetime import datetime
from typing import Optional, List

import tushare as ts
import pandas as pd

# -------------------------- 配置项 --------------------------
TUSHARE_TOKEN = "8a89b2a27a727a462e313d56313bf60737ff2fc7bda7d7ad95b88ad8"  # ★ 实际使用建议放到环境变量
BASE_SAVE_DIR = "./data_raw/a_share_daily/"     # 数据保存目录（每只股票一个 CSV）

# 频率控制：TuShare 限 50 次/min，我们保守一点控制在 ~40 次/min 左右
BASE_SLEEP = 1.5     # 每次请求后基础 sleep 秒数（60 / 40 = 1.5）
JITTER = 0.5         # 再加 0~JITTER 的随机抖动，避免节奏太整齐

# -----------------------------------------------------------


def init_tushare() -> ts.pro_api:
    """初始化 TuShare Pro 接口"""
    ts.set_token(TUSHARE_TOKEN)
    return ts.pro_api()


def date_to_tushare_format(date_str: str) -> str:
    """把 'YYYY-MM-DD' 转成 TuShare 要求的 'YYYYMMDD' 格式"""
    return datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y%m%d")


def fetch_klines(
    pro: ts.pro_api,
    ts_code: str,
    freq: str = "D",
    start_str: str = "2010-01-01",
    end_str: Optional[str] = None,
    adj: str = "qfq",   # 注意：正式使用复权建议用 pro_bar，这里沿用你原来的写法
) -> pd.DataFrame:
    """
    从 TuShare Pro 抓取 A 股 K 线数据（适配最新接口）

    :param pro: 已初始化的 ts.pro_api 实例
    :param ts_code: 股票代码，如 '600000.SH'、'000001.SZ'
    :param freq: K 线周期，支持 D/W/M/60min/30min/15min/5min/1min
    :param start_str: 起始日期，'YYYY-MM-DD'
    :param end_str: 结束日期，None 表示当前日期
    :param adj: 复权类型
    """
    # 处理日期格式
    start_date = date_to_tushare_format(start_str)
    end_date = date_to_tushare_format(end_str) if end_str else datetime.now().strftime("%Y%m%d")

    # 不同周期调用不同接口
    if freq == "D":
        df = pro.daily(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            # adj=adj  # 如果 daily 不支持 adj 参数，可以改用 pro_bar
        )
    elif freq == "W":
        df = pro.weekly(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            # adj=adj
        )
    elif freq == "M":
        df = pro.monthly(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            # adj=adj
        )
    elif freq in ["60min", "30min", "15min", "5min", "1min"]:
        # 分钟线建议用 pro_bar，这里保留你原来的接口名占位
        df = pro.minute_bar(
            ts_code=ts_code,
            freq=freq,
            start_date=start_date,
            end_date=end_date,
            # adj=adj
        )
    else:
        raise ValueError(f"不支持的周期类型：{freq}")

    if df is None or df.empty:
        raise ValueError(f"未获取到 {ts_code} ({freq}) 的 K 线数据")

    return df


def klines_to_dataframe(df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """清洗 K 线数据：调整列名、转换时间格式、修正数据类型"""
    col_mapping = {
        "trade_date": "trade_date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "vol": "volume",
        "amount": "amount",
        "pct_chg": "pct_chg",
    }
    if "datetime" in df.columns:
        col_mapping["datetime"] = "trade_time"

    df = df.rename(columns=col_mapping, errors="ignore")

    if "trade_date" in df.columns:
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")
    if "trade_time" in df.columns:
        df["trade_time"] = pd.to_datetime(df["trade_time"])

    numeric_cols = ["open", "high", "low", "close", "volume", "amount", "pct_chg"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)

    sort_col = "trade_time" if "trade_time" in df.columns else "trade_date"
    df = df.sort_values(sort_col).reset_index(drop=True)

    return df


def get_all_a_share_codes(pro: ts.pro_api) -> List[str]:
    """
    获取所有 A 股 ts_code 列表，比如 '600000.SH', '000001.SZ'
    只保留在市 (list_status='L') 且交易所为上证/深证
    """
    print("获取全部 A 股列表...")
    basic = pro.stock_basic(
        exchange="",
        list_status="L",
        fields="ts_code,symbol,name,area,industry,market,list_date,exchange",
    )
    # 上证/深证 A 股
    mask = basic["ts_code"].str.endswith(("SH"))
    a_shares = basic[mask].copy()
    codes = a_shares["ts_code"].tolist()
    print(f"在市 A 股数量: {len(codes)}")
    return codes


def main():
    pro = init_tushare()

    # ===== 你可以先用一个子集测试 =====
    # 
    # ==================================

    # 全市场
    codes = get_all_a_share_codes(pro)

    freq = "D"                   # 周期
    start_date = "2015-01-01"
    end_date = None
    adj_type = "qfq"

    # 保存目录
    os.makedirs(BASE_SAVE_DIR, exist_ok=True)

    freq_map = {
        "D": "day", "W": "week", "M": "month",
        "60min": "60min", "30min": "30min",
        "15min": "15min", "5min": "5min", "1min": "1min",
    }
    freq_tag = freq_map.get(freq, freq)

    failed_codes_path = os.path.join(BASE_SAVE_DIR, "failed_codes.txt")
    if os.path.exists(failed_codes_path):
        os.remove(failed_codes_path)

    total = len(codes)
    print(f"准备下载 {total} 只股票 {freq} K 线数据，从 {start_date} 到 {end_date or 'today'} ...")

    for idx, ts_code in enumerate(codes, start=1):
        try:
            raw_df = fetch_klines(
                pro=pro,
                ts_code=ts_code,
                freq=freq,
                start_str=start_date,
                end_str=end_date,
                adj=adj_type,
            )
            cleaned_df = klines_to_dataframe(raw_df, freq=freq)

            save_filename = os.path.join(
                BASE_SAVE_DIR,
                f"{ts_code}_{freq_tag}_{adj_type}.csv",
            )
            cleaned_df.to_csv(save_filename, index=False, encoding="utf-8")
            print(f"[{idx:05d}/{total}] {ts_code} 行数={len(cleaned_df)} -> {save_filename}")

        except Exception as e:
            print(f"[{idx:05d}/{total}] {ts_code} 下载失败: {e}")
            with open(failed_codes_path, "a", encoding="utf-8") as f:
                f.write(ts_code + "\n")

        # ---- 频率控制：每次请求之后 sleep 一段时间 ----
        sleep_t = BASE_SLEEP + random.uniform(0, JITTER)
        time.sleep(sleep_t)

    print("全部下载任务结束。")
    if os.path.exists(failed_codes_path):
        print(f"部分代码下载失败，已记录在: {failed_codes_path}")


if __name__ == "__main__":
    main()