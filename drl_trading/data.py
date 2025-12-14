from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import yfinance as yf

from .features import add_technical_features


@dataclass(frozen=True)
class MarketData:
    df: pd.DataFrame
    feature_cols: list[str]


def download_ohlcv(
    ticker: str,
    start: str = "2015-01-01",
    end: str | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Download OHLCV from Yahoo Finance.

    interval examples: "1d", "1h", "15m" (intraday is limited by Yahoo).
    """
    raw = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )
    if raw is None or raw.empty:
        raise ValueError(f"No data returned for {ticker} ({start=} {end=} {interval=})")

    # Standardize columns
    df = raw.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    df = df[[c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]].copy()
    df = df.dropna()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def build_market_data(df_ohlcv: pd.DataFrame) -> MarketData:
    df = add_technical_features(df_ohlcv)

    # Define features used by DRL agent (you can add more)
    feature_cols = [
        "ret_1",
        "log_ret_1",
        "sma_ratio_10_20",
        "ema_ratio_10_20",
        "rsi_14",
        "vol_20",
        "vol_chg_1",
    ]
    return MarketData(df=df, feature_cols=feature_cols)


def train_test_split_by_date(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be in (0,1)")

    n = len(df)
    split = int(n * train_ratio)
    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()
    return train, test
