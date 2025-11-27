import requests
import pandas as pd
from datetime import datetime, timezone

BASE_URL = "https://api.binance.com/api/v3/klines"


def date_to_milliseconds(date_str: str) -> int:
    """
    把 'YYYY-MM-DD' 转成 Binance API 需要的毫秒时间戳
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    # Binance 用 UTC 时间
    dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def fetch_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1d",
    start_str: str = "2015-01-01",
    end_str: str | None = None,
    limit: int = 1000,
):
    """
    从 Binance Spot API 批量抓取 K 线数据

    :param symbol: 交易对，例如 'BTCUSDT'
    :param interval: K 线周期，例如 '1d', '1h', '4h' 等
    :param start_str: 起始日期字符串，格式 'YYYY-MM-DD'
    :param end_str: 结束日期字符串，格式 'YYYY-MM-DD'，None 表示当前时间
    :param limit: 每次请求的最大 K 线数量（Binance /api/v3/klines 上限 1000）
    :return: list[list]，每个元素是一根 K 线的数据
    """
    start_ts = date_to_milliseconds(start_str)
    if end_str is None:
        end_ts = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    else:
        end_ts = date_to_milliseconds(end_str)

    all_klines = []
    current_start = start_ts

    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "limit": limit,
        }

        # 如果你想严格截到 end_ts，也可以加上 endTime，但不是必须
        # params["endTime"] = end_ts

        resp = requests.get(BASE_URL, params=params, timeout=10)
        resp.raise_for_status()
        klines = resp.json()

        if not klines:
            # 没有更多数据了
            break

        all_klines.extend(klines)

        # 最后一根 K 线的开盘时间
        last_open_time = klines[-1][0]

        # 如果已经超过结束时间或者不足 limit，说明拿完了
        if last_open_time >= end_ts or len(klines) < limit:
            break

        # 下一个请求从最后一根K线的下一毫秒开始
        current_start = last_open_time + 1

    return all_klines


def klines_to_dataframe(klines):
    """
    把 Binance 返回的原始 K 线 list 转成 pandas DataFrame，并做一些基础清洗
    """
    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]
    df = pd.DataFrame(klines, columns=columns)

    # 时间戳转成可读日期
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    # 数值列转 float / int（方便后续计算）
    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]
    df[numeric_cols] = df[numeric_cols].astype(float)
    df["number_of_trades"] = df["number_of_trades"].astype(int)

    return df


if __name__ == "__main__":
    # 想拉最近 10 年，从 2015-01-01 开始就够用
    symbol = "BTCUSDT"
    interval = "1d"
    start_date = "2015-01-01"
    end_date = None  # None 表示拉到当前时间

    print(f"Fetching {symbol} {interval} klines from Binance...")
    klines = fetch_klines(
        symbol=symbol,
        interval=interval,
        start_str=start_date,
        end_str=end_date,
    )
    print(f"Total klines fetched: {len(klines)}")

    df = klines_to_dataframe(klines)
    print(df.head())

    # 保存为 CSV
    out_file = f"./data_raw/{symbol}_binance_{interval}.csv"
    df.to_csv(out_file, index=False)
    print(f"Saved to {out_file}")