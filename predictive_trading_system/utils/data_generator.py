"""
Synthetic Data Generator
========================
Generates realistic synthetic market and trading log data for testing.
Produces OHLCV with regime switching, plus simulated trading logs.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


def generate_synthetic_ohlcv(
    n_bars: int = 5000,
    start_price: float = 100.0,
    base_volatility: float = 0.02,
    regime_switch_prob: float = 0.01,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data with regime switching.
    Includes trending, mean-reverting, and volatile regimes.
    """
    if seed is not None:
        np.random.seed(seed)

    prices = np.zeros(n_bars)
    prices[0] = start_price
    volumes = np.zeros(n_bars)

    # Regime: 0=mean-revert, 1=trend-up, 2=trend-down, 3=high-vol
    regime = 0
    regimes = np.zeros(n_bars, dtype=int)

    for i in range(1, n_bars):
        if np.random.random() < regime_switch_prob:
            regime = np.random.choice([0, 1, 2, 3])
        regimes[i] = regime

        if regime == 0:  # Mean reverting
            drift = -0.05 * (prices[i-1] - start_price) / start_price
            vol = base_volatility * 0.7
        elif regime == 1:  # Trending up
            drift = 0.0005
            vol = base_volatility
        elif regime == 2:  # Trending down
            drift = -0.0005
            vol = base_volatility
        else:  # High volatility
            drift = 0.0
            vol = base_volatility * 2.5

        ret = drift + vol * np.random.randn()
        prices[i] = prices[i-1] * (1 + ret)

        base_vol = 1_000_000
        vol_multiplier = 1 + abs(ret) * 50  # volume spike on big moves
        if regime == 3:
            vol_multiplier *= 2
        volumes[i] = base_vol * vol_multiplier * (0.5 + np.random.random())

    volumes[0] = 1_000_000

    timestamps = pd.date_range("2024-01-01", periods=n_bars, freq="1min", tz="UTC")

    highs = prices * (1 + np.abs(np.random.randn(n_bars) * base_volatility * 0.3))
    lows = prices * (1 - np.abs(np.random.randn(n_bars) * base_volatility * 0.3))
    opens = prices * (1 + np.random.randn(n_bars) * base_volatility * 0.1)

    # Ensure OHLC consistency
    highs = np.maximum(highs, np.maximum(opens, prices))
    lows = np.minimum(lows, np.minimum(opens, prices))

    return pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": volumes.astype(int),
    }, index=timestamps)


def generate_synthetic_trades(
    n_trades: int = 500,
    symbols: Optional[list] = None,
    win_rate: float = 0.55,
    avg_pnl: float = 50.0,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """Generate synthetic trading log data simulating a day trader's activity."""
    if seed is not None:
        np.random.seed(seed)

    if symbols is None:
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "SPY", "QQQ"]

    trades = []
    timestamps = pd.date_range("2024-01-01", periods=n_trades, freq="2h", tz="UTC")

    for i in range(n_trades):
        is_winner = np.random.random() < win_rate
        if is_winner:
            pnl = abs(np.random.normal(avg_pnl, avg_pnl * 0.5))
        else:
            pnl = -abs(np.random.normal(avg_pnl * 0.8, avg_pnl * 0.3))

        quantity = int(np.random.choice([10, 25, 50, 100, 200, 500]))
        price = np.random.uniform(50, 500)

        trades.append({
            "timestamp": timestamps[i],
            "symbol": np.random.choice(symbols),
            "side": np.random.choice(["long", "short"], p=[0.6, 0.4]),
            "quantity": quantity,
            "price": round(price, 2),
            "pnl": round(pnl, 2),
            "commission": round(quantity * 0.005, 2),
            "strategy": np.random.choice(["momentum", "mean_reversion", "breakout", "scalp"]),
        })

    return pd.DataFrame(trades)
