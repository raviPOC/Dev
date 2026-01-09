"""
Feature Engineering Module for Price Prediction

Creates technical indicators and features optimized for short-term price prediction.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple


class FeatureEngineer:
    """
    Feature engineering for short-term price prediction.
    
    Creates technical indicators, price transformations, and lagged features
    optimized for 1-hour prediction horizons.
    """
    
    def __init__(
        self,
        price_col: str = "close",
        volume_col: str = "volume",
        high_col: str = "high",
        low_col: str = "low",
        open_col: str = "open",
    ):
        self.price_col = price_col
        self.volume_col = volume_col
        self.high_col = high_col
        self.low_col = low_col
        self.open_col = open_col
        self.feature_names: List[str] = []
        
    def create_all_features(
        self,
        df: pd.DataFrame,
        include_target: bool = True,
        target_horizon: int = 1,
    ) -> pd.DataFrame:
        """
        Create all features for price prediction.
        
        Args:
            df: DataFrame with OHLCV data
            include_target: Whether to create target variable
            target_horizon: Number of periods ahead to predict
            
        Returns:
            DataFrame with all features
        """
        df = df.copy()
        
        # Price-based features
        df = self._add_price_features(df)
        
        # Technical indicators
        df = self._add_momentum_indicators(df)
        df = self._add_volatility_indicators(df)
        df = self._add_trend_indicators(df)
        df = self._add_volume_indicators(df)
        
        # Lagged features
        df = self._add_lagged_features(df)
        
        # Time-based features (if datetime index)
        df = self._add_time_features(df)
        
        # Target variable
        if include_target:
            df = self._add_target(df, target_horizon)
        
        # Store feature names
        self.feature_names = [
            col for col in df.columns 
            if col not in [self.price_col, self.volume_col, self.high_col, 
                          self.low_col, self.open_col, "target", "target_direction"]
        ]
        
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic price-derived features."""
        price = df[self.price_col]
        
        # Returns
        df["return_1"] = price.pct_change(1)
        df["return_5"] = price.pct_change(5)
        df["return_10"] = price.pct_change(10)
        df["return_20"] = price.pct_change(20)
        
        # Log returns (more stable for ML)
        df["log_return_1"] = np.log(price / price.shift(1))
        df["log_return_5"] = np.log(price / price.shift(5))
        
        # Price ratios
        df["high_low_ratio"] = df[self.high_col] / df[self.low_col]
        df["close_open_ratio"] = df[self.price_col] / df[self.open_col]
        
        # Candle body and wick
        df["body"] = df[self.price_col] - df[self.open_col]
        df["upper_wick"] = df[self.high_col] - df[[self.price_col, self.open_col]].max(axis=1)
        df["lower_wick"] = df[[self.price_col, self.open_col]].min(axis=1) - df[self.low_col]
        
        return df
    
    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based technical indicators."""
        price = df[self.price_col]
        high = df[self.high_col]
        low = df[self.low_col]
        
        # RSI (Relative Strength Index)
        for period in [7, 14, 21]:
            df[f"rsi_{period}"] = self._calculate_rsi(price, period)
        
        # Stochastic Oscillator
        for period in [14, 21]:
            df[f"stoch_k_{period}"] = self._calculate_stochastic(high, low, price, period)
            df[f"stoch_d_{period}"] = df[f"stoch_k_{period}"].rolling(3).mean()
        
        # Rate of Change (ROC)
        for period in [5, 10, 20]:
            df[f"roc_{period}"] = ((price - price.shift(period)) / price.shift(period)) * 100
        
        # MACD
        exp1 = price.ewm(span=12, adjust=False).mean()
        exp2 = price.ewm(span=26, adjust=False).mean()
        df["macd"] = exp1 - exp2
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]
        
        # Williams %R
        for period in [14, 21]:
            highest_high = high.rolling(period).max()
            lowest_low = low.rolling(period).min()
            df[f"williams_r_{period}"] = -100 * (highest_high - price) / (highest_high - lowest_low)
        
        return df
    
    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based indicators."""
        price = df[self.price_col]
        high = df[self.high_col]
        low = df[self.low_col]
        
        # Standard deviation of returns
        returns = price.pct_change()
        for period in [5, 10, 20]:
            df[f"volatility_{period}"] = returns.rolling(period).std()
        
        # Average True Range (ATR)
        for period in [7, 14, 21]:
            df[f"atr_{period}"] = self._calculate_atr(high, low, price, period)
            df[f"atr_pct_{period}"] = df[f"atr_{period}"] / price * 100
        
        # Bollinger Bands
        for period in [20]:
            sma = price.rolling(period).mean()
            std = price.rolling(period).std()
            df[f"bb_upper_{period}"] = sma + (std * 2)
            df[f"bb_lower_{period}"] = sma - (std * 2)
            df[f"bb_width_{period}"] = (df[f"bb_upper_{period}"] - df[f"bb_lower_{period}"]) / sma
            df[f"bb_position_{period}"] = (price - df[f"bb_lower_{period}"]) / (df[f"bb_upper_{period}"] - df[f"bb_lower_{period}"])
        
        # Keltner Channels
        ema_20 = price.ewm(span=20, adjust=False).mean()
        atr_10 = self._calculate_atr(high, low, price, 10)
        df["keltner_upper"] = ema_20 + (atr_10 * 2)
        df["keltner_lower"] = ema_20 - (atr_10 * 2)
        df["keltner_position"] = (price - df["keltner_lower"]) / (df["keltner_upper"] - df["keltner_lower"])
        
        return df
    
    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend-following indicators."""
        price = df[self.price_col]
        
        # Simple Moving Averages
        for period in [5, 10, 20, 50]:
            df[f"sma_{period}"] = price.rolling(period).mean()
            df[f"price_sma_{period}_ratio"] = price / df[f"sma_{period}"]
        
        # Exponential Moving Averages
        for period in [5, 10, 20, 50]:
            df[f"ema_{period}"] = price.ewm(span=period, adjust=False).mean()
            df[f"price_ema_{period}_ratio"] = price / df[f"ema_{period}"]
        
        # Moving Average Crossovers
        df["sma_5_10_cross"] = df["sma_5"] - df["sma_10"]
        df["sma_10_20_cross"] = df["sma_10"] - df["sma_20"]
        df["ema_5_10_cross"] = df["ema_5"] - df["ema_10"]
        
        # ADX (Average Directional Index)
        df["adx_14"] = self._calculate_adx(df, 14)
        
        # Linear regression slope
        for period in [10, 20]:
            df[f"linreg_slope_{period}"] = self._calculate_linreg_slope(price, period)
        
        return df
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators."""
        if self.volume_col not in df.columns:
            return df
            
        price = df[self.price_col]
        volume = df[self.volume_col]
        
        # Volume Moving Averages
        for period in [5, 10, 20]:
            df[f"volume_sma_{period}"] = volume.rolling(period).mean()
            df[f"volume_ratio_{period}"] = volume / df[f"volume_sma_{period}"]
        
        # On-Balance Volume (OBV)
        obv = (np.sign(price.diff()) * volume).fillna(0).cumsum()
        df["obv"] = obv
        df["obv_sma_10"] = obv.rolling(10).mean()
        df["obv_momentum"] = obv - df["obv_sma_10"]
        
        # Volume Weighted Average Price (VWAP) - simplified
        df["vwap"] = (price * volume).rolling(20).sum() / volume.rolling(20).sum()
        df["price_vwap_ratio"] = price / df["vwap"]
        
        # Money Flow Index (MFI)
        df["mfi_14"] = self._calculate_mfi(df, 14)
        
        # Accumulation/Distribution
        clv = ((price - df[self.low_col]) - (df[self.high_col] - price)) / (df[self.high_col] - df[self.low_col])
        df["ad_line"] = (clv * volume).fillna(0).cumsum()
        
        return df
    
    def _add_lagged_features(self, df: pd.DataFrame, max_lag: int = 10) -> pd.DataFrame:
        """Add lagged versions of key features."""
        # Lagged returns
        for lag in [1, 2, 3, 5]:
            df[f"return_lag_{lag}"] = df["return_1"].shift(lag)
            
        # Lagged RSI
        for lag in [1, 2, 3]:
            df[f"rsi_14_lag_{lag}"] = df["rsi_14"].shift(lag)
            
        # Lagged volatility
        for lag in [1, 2]:
            df[f"volatility_10_lag_{lag}"] = df["volatility_10"].shift(lag)
            
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features if datetime index available."""
        if not isinstance(df.index, pd.DatetimeIndex):
            return df
            
        df["hour"] = df.index.hour
        df["day_of_week"] = df.index.dayofweek
        df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
        
        # Cyclical encoding for hour
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        
        # Cyclical encoding for day of week
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        
        return df
    
    def _add_target(self, df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
        """Add target variable for prediction."""
        price = df[self.price_col]
        
        # Future return (regression target)
        df["target"] = price.shift(-horizon) / price - 1
        
        # Direction (classification target)
        df["target_direction"] = (df["target"] > 0).astype(int)
        
        return df
    
    # Helper calculation methods
    
    def _calculate_rsi(self, price: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = price.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_stochastic(
        self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """Calculate Stochastic Oscillator %K."""
        lowest_low = low.rolling(period).min()
        highest_high = high.rolling(period).max()
        return 100 * (close - lowest_low) / (highest_high - lowest_low)
    
    def _calculate_atr(
        self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """Calculate Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index."""
        high = df[self.high_col]
        low = df[self.low_col]
        close = df[self.price_col]
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        atr = self._calculate_atr(high, low, close, period)
        
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        return dx.rolling(period).mean()
    
    def _calculate_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index."""
        typical_price = (df[self.high_col] + df[self.low_col] + df[self.price_col]) / 3
        money_flow = typical_price * df[self.volume_col]
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0)
        
        positive_mf = positive_flow.rolling(period).sum()
        negative_mf = negative_flow.rolling(period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        return mfi
    
    def _calculate_linreg_slope(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate linear regression slope."""
        def slope(arr):
            if len(arr) < period:
                return np.nan
            x = np.arange(len(arr))
            return np.polyfit(x, arr, 1)[0]
        
        return series.rolling(period).apply(slope, raw=True)
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return self.feature_names.copy()
    
    def prepare_sequences(
        self,
        df: pd.DataFrame,
        sequence_length: int = 60,
        target_col: str = "target",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM/sequence models.
        
        Args:
            df: DataFrame with features
            sequence_length: Number of time steps in each sequence
            target_col: Target column name
            
        Returns:
            Tuple of (X sequences, y targets)
        """
        feature_cols = [col for col in self.feature_names if col in df.columns]
        
        # Drop rows with NaN values
        df_clean = df[feature_cols + [target_col]].dropna()
        
        X_data = df_clean[feature_cols].values
        y_data = df_clean[target_col].values
        
        X_sequences = []
        y_sequences = []
        
        for i in range(sequence_length, len(X_data)):
            X_sequences.append(X_data[i-sequence_length:i])
            y_sequences.append(y_data[i])
        
        return np.array(X_sequences), np.array(y_sequences)
