from __future__ import annotations

import numpy as np
import pandas as pd


def _ema(x: pd.Series, span: int) -> pd.Series:
    return x.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (0..100)."""
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)

    # Wilder's smoothing (EMA with alpha=1/period)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()

    rs = roll_up / (roll_down.replace(0.0, np.nan))
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(50.0)


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds simple, self-contained technical features.

    Expected columns: open, high, low, close, volume
    Returns a new DataFrame with added feature columns.
    """
    out = df.copy()

    close = out["close"].astype(float)
    volume = out["volume"].astype(float)

    out["ret_1"] = close.pct_change().fillna(0.0)
    out["log_ret_1"] = np.log(close / close.shift(1)).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    out["sma_10"] = close.rolling(10).mean()
    out["sma_20"] = close.rolling(20).mean()
    out["sma_ratio_10_20"] = (out["sma_10"] / out["sma_20"]).replace([np.inf, -np.inf], np.nan)

    out["ema_10"] = _ema(close, 10)
    out["ema_20"] = _ema(close, 20)
    out["ema_ratio_10_20"] = (out["ema_10"] / out["ema_20"]).replace([np.inf, -np.inf], np.nan)

    out["rsi_14"] = rsi(close, 14) / 100.0  # normalize to 0..1

    # Volatility proxy: rolling std of returns
    out["vol_20"] = out["ret_1"].rolling(20).std().fillna(0.0)

    # Volume change
    out["vol_chg_1"] = volume.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    out = out.replace([np.inf, -np.inf], np.nan).fillna(method="bfill").fillna(method="ffill")
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    base = [
        "ret_1",
        "log_ret_1",
        "sma_ratio_10_20",
        "ema_ratio_10_20",
        "rsi_14",
        "vol_20",
        "vol_chg_1",
    ]
    missing = [c for c in base if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return base
