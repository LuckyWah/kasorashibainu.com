from pathlib import Path

import numpy as np
import pandas as pd
import requests
import urllib3
import yfinance as yf
from curl_cffi import requests as curl_requests
from pandas_datareader import data as pdr


YFINANCE_SESSION = curl_requests.Session(impersonate="chrome", verify=False)
FRED_SESSION = requests.Session()
FRED_SESSION.verify = False
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

REQUIRED_FEATURES = [
    "drawdown",
    "momentum_20d",
    "momentum_60d",
    "ma_gap",
    "volatility_20d",
    "volume_spike",
    "spy_return",
    "spy_drawdown",
    "vix",
    "fed_rate",
    "treasury_10y",
    "cpi_yoy",
]


def download_yahoo(ticker, start="2010-01-01", end=None):
    ticker = ticker.upper().strip()

    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        session=YFINANCE_SESSION,
    )

    if df.empty:
        raise ValueError(f"No Yahoo Finance data found for {ticker}.")

    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(ticker, axis=1, level=1)
        except Exception:
            df.columns = df.columns.get_level_values(0)

    df.columns = [str(c).lower() for c in df.columns]

    if "close" not in df.columns:
        raise ValueError(f"No close price found for {ticker}.")

    return df


def download_fred(series_id, start, end=None):
    return pdr.DataReader(series_id, "fred", start, end, session=FRED_SESSION)


def build_adaptive_dataset(
    ticker,
    market_index="SPY",
    start_date="2010-01-01",
    end_date=None,
    output_dir="datasets",
):
    ticker = ticker.upper().strip()
    market_index = market_index.upper().strip()

    stock = download_yahoo(ticker, start_date, end_date)
    index = download_yahoo(market_index, start_date, end_date)

    vix = download_fred("VIXCLS", start_date, end_date)
    fed_rate = download_fred("FEDFUNDS", start_date, end_date)
    treasury_10y = download_fred("DGS10", start_date, end_date)
    cpi = download_fred("CPIAUCSL", start_date, end_date)

    df = stock.copy()
    df["return"] = df["close"].pct_change()
    df["rolling_max"] = df["close"].cummax()
    df["drawdown"] = (df["rolling_max"] - df["close"]) / df["rolling_max"]
    df["momentum_20d"] = df["close"].pct_change(20)
    df["momentum_60d"] = df["close"].pct_change(60)
    df["ma_50"] = df["close"].rolling(50).mean()
    df["ma_200"] = df["close"].rolling(200).mean()
    df["ma_gap"] = (df["ma_50"] - df["ma_200"]) / df["ma_200"]
    df["volatility_20d"] = df["return"].rolling(20).std() * np.sqrt(252)
    df["volume_ma_20"] = df["volume"].rolling(20).mean()
    df["volume_spike"] = df["volume"] / df["volume_ma_20"]
    df["spy_return"] = index["close"].pct_change()
    df["spy_drawdown"] = (
        index["close"].cummax() - index["close"]
    ) / index["close"].cummax()

    df = df.join(vix.rename(columns={"VIXCLS": "vix"}), how="left")
    df = df.join(fed_rate.rename(columns={"FEDFUNDS": "fed_rate"}), how="left")
    df = df.join(treasury_10y.rename(columns={"DGS10": "treasury_10y"}), how="left")
    df = df.join(cpi.rename(columns={"CPIAUCSL": "cpi"}), how="left")
    df = df.ffill()
    df["cpi_yoy"] = df["cpi"].pct_change(252)
    df["future_return_30d"] = df["close"].shift(-30) / df["close"] - 1
    df["future_return_60d"] = df["close"].shift(-60) / df["close"] - 1
    df["rebound_30d"] = (df["future_return_30d"] > 0).astype(int)
    df["future_drop_10pct_30d"] = (df["future_return_30d"] < -0.10).astype(int)
    df = df.dropna(subset=REQUIRED_FEATURES)

    if df.empty:
        raise ValueError(f"Not enough usable data for {ticker}.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{ticker}.csv"
    df.to_csv(output_file)

    return output_file
