"""
Data Loading Module for Price Prediction

Handles loading and preprocessing of price data from various sources.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union
from datetime import datetime, timedelta


class DataLoader:
    """
    Data loader for price prediction models.
    
    Supports loading from CSV, generating synthetic data for testing,
    and fetching from APIs (yfinance, ccxt).
    """
    
    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        
    def load_csv(
        self,
        filepath: str,
        date_col: str = "timestamp",
        parse_dates: bool = True,
    ) -> pd.DataFrame:
        """
        Load price data from CSV file.
        
        Args:
            filepath: Path to CSV file
            date_col: Name of date/timestamp column
            parse_dates: Whether to parse dates
            
        Returns:
            DataFrame with OHLCV data
        """
        df = pd.read_csv(filepath)
        
        if parse_dates and date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
            df.sort_index(inplace=True)
        
        self.data = df
        return df
    
    def generate_synthetic_data(
        self,
        n_samples: int = 5000,
        start_price: float = 100.0,
        volatility: float = 0.02,
        trend: float = 0.0001,
        seed: Optional[int] = 42,
    ) -> pd.DataFrame:
        """
        Generate synthetic OHLCV data for testing.
        
        Creates realistic price data with trends, mean reversion,
        and volume patterns.
        
        Args:
            n_samples: Number of data points to generate
            start_price: Starting price
            volatility: Price volatility (standard deviation of returns)
            trend: Long-term trend (positive = uptrend)
            seed: Random seed for reproducibility
            
        Returns:
            DataFrame with synthetic OHLCV data
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate timestamps (hourly data)
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(hours=n_samples),
            periods=n_samples,
            freq="h"
        )
        
        # Generate returns with some autocorrelation and mean reversion
        returns = np.zeros(n_samples)
        returns[0] = np.random.normal(trend, volatility)
        
        for i in range(1, n_samples):
            # Add momentum component
            momentum = 0.1 * returns[i-1]
            # Add mean reversion
            mean_reversion = -0.05 * np.sum(returns[max(0, i-20):i]) if i > 0 else 0
            # Add random component
            noise = np.random.normal(0, volatility)
            # Add trend
            returns[i] = trend + momentum + mean_reversion + noise
        
        # Generate close prices
        close = start_price * np.exp(np.cumsum(returns))
        
        # Generate OHLV from close
        high = close * (1 + np.abs(np.random.normal(0, volatility/2, n_samples)))
        low = close * (1 - np.abs(np.random.normal(0, volatility/2, n_samples)))
        open_price = np.roll(close, 1)
        open_price[0] = start_price
        
        # Ensure high >= max(open, close) and low <= min(open, close)
        high = np.maximum(high, np.maximum(open_price, close))
        low = np.minimum(low, np.minimum(open_price, close))
        
        # Generate volume with patterns
        base_volume = 1000000
        volume = base_volume * (
            1 + 0.5 * np.abs(returns) / volatility  # Higher volume on bigger moves
            + 0.3 * np.random.exponential(1, n_samples)  # Random spikes
        )
        
        # Add time-of-day pattern to volume
        hour_effect = 1 + 0.3 * np.sin(2 * np.pi * np.arange(n_samples) / 24)
        volume = volume * hour_effect
        
        df = pd.DataFrame({
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume.astype(int),
        }, index=timestamps)
        
        self.data = df
        return df
    
    def fetch_yfinance(
        self,
        symbol: str = "BTC-USD",
        interval: str = "1h",
        period: str = "60d",
    ) -> pd.DataFrame:
        """
        Fetch data from Yahoo Finance.
        
        Args:
            symbol: Ticker symbol
            interval: Data interval (1m, 5m, 15m, 1h, 1d)
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, max)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance not installed. Run: pip install yfinance")
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        # Standardize column names
        df.columns = df.columns.str.lower()
        df = df[["open", "high", "low", "close", "volume"]]
        
        self.data = df
        return df
    
    def fetch_ccxt(
        self,
        exchange: str = "binance",
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch data from cryptocurrency exchange using ccxt.
        
        Args:
            exchange: Exchange name (binance, coinbase, kraken, etc.)
            symbol: Trading pair
            timeframe: Candle timeframe
            limit: Number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            import ccxt
        except ImportError:
            raise ImportError("ccxt not installed. Run: pip install ccxt")
        
        exchange_class = getattr(ccxt, exchange)()
        ohlcv = exchange_class.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        df = pd.DataFrame(
            ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        
        self.data = df
        return df
    
    def train_test_split(
        self,
        df: Optional[pd.DataFrame] = None,
        train_ratio: float = 0.8,
        validation_ratio: float = 0.1,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Uses time-based splitting to prevent data leakage.
        
        Args:
            df: DataFrame to split (uses self.data if None)
            train_ratio: Proportion of data for training
            validation_ratio: Proportion of data for validation
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if df is None:
            df = self.data
        
        if df is None:
            raise ValueError("No data available. Load data first.")
        
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + validation_ratio))
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        return train_df, val_df, test_df
    
    def get_data(self) -> Optional[pd.DataFrame]:
        """Return loaded data."""
        return self.data
