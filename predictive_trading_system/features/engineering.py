"""
Feature Engineering Pipeline
=============================
Computes technical indicators, volatility measures, volume features, and
behavioral features from trading logs. All features are computed without
lookahead bias — only past data is used at each point.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config.settings import FeatureConfig

logger = logging.getLogger(__name__)


class TechnicalFeatures:
    """Standard technical analysis features computed from OHLCV data."""

    def __init__(self, config: FeatureConfig):
        self.config = config

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._price_features(df)
        df = self._momentum_features(df)
        df = self._volatility_features(df)
        df = self._volume_features(df)
        df = self._pattern_features(df)
        return df

    def _price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]

        df["returns"] = close.pct_change()
        df["log_returns"] = np.log(close / close.shift(1))

        for w in self.config.ma_windows:
            df[f"sma_{w}"] = close.rolling(w).mean()
            df[f"ema_{w}"] = close.ewm(span=w, adjust=False).mean()
            df[f"price_vs_sma_{w}"] = (close - df[f"sma_{w}"]) / df[f"sma_{w}"]

        for i, fast in enumerate(self.config.ma_windows[:-1]):
            slow = self.config.ma_windows[i + 1]
            df[f"ma_cross_{fast}_{slow}"] = (
                df[f"sma_{fast}"] - df[f"sma_{slow}"]
            ) / df[f"sma_{slow}"]

        if "high" in df.columns and "low" in df.columns:
            for w in [5, 10, 20, 50]:
                df[f"high_{w}"] = df["high"].rolling(w).max()
                df[f"low_{w}"] = df["low"].rolling(w).min()
                channel_width = df[f"high_{w}"] - df[f"low_{w}"]
                safe_width = channel_width.replace(0, np.nan)
                df[f"price_position_{w}"] = (close - df[f"low_{w}"]) / safe_width

        return df

    def _momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(self.config.rsi_period).mean()
        avg_loss = loss.rolling(self.config.rsi_period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))

        fast_ema = close.ewm(span=self.config.macd_fast, adjust=False).mean()
        slow_ema = close.ewm(span=self.config.macd_slow, adjust=False).mean()
        df["macd"] = fast_ema - slow_ema
        df["macd_signal"] = df["macd"].ewm(span=self.config.macd_signal, adjust=False).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]

        for period in [5, 10, 20]:
            df[f"roc_{period}"] = close.pct_change(period)

        if "high" in df.columns and "low" in df.columns:
            low_14 = df["low"].rolling(14).min()
            high_14 = df["high"].rolling(14).max()
            range_14 = (high_14 - low_14).replace(0, np.nan)
            df["stoch_k"] = 100 * (close - low_14) / range_14
            df["stoch_d"] = df["stoch_k"].rolling(3).mean()

            high_14_wr = df["high"].rolling(14).max()
            range_14_wr = (high_14_wr - df["low"].rolling(14).min()).replace(0, np.nan)
            df["williams_r"] = -100 * (high_14_wr - close) / range_14_wr

        return df

    def _volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]

        for w in [5, 10, 20, 60]:
            df[f"realized_vol_{w}"] = df["log_returns"].rolling(w).std() * np.sqrt(252)

        if "high" in df.columns and "low" in df.columns:
            hl = np.log(df["high"] / df["low"])
            co = np.log(close / df["open"]) if "open" in df.columns else pd.Series(0, index=df.index)
            for w in [10, 20]:
                df[f"parkinson_vol_{w}"] = np.sqrt(
                    (1 / (4 * np.log(2))) * (hl ** 2).rolling(w).mean()
                ) * np.sqrt(252)

            tr = pd.concat([
                df["high"] - df["low"],
                (df["high"] - close.shift(1)).abs(),
                (df["low"] - close.shift(1)).abs(),
            ], axis=1).max(axis=1)
            df["atr"] = tr.rolling(self.config.atr_period).mean()
            df["atr_pct"] = df["atr"] / close

        sma = close.rolling(self.config.bb_period).mean()
        std = close.rolling(self.config.bb_period).std()
        df["bb_upper"] = sma + self.config.bb_std * std
        df["bb_lower"] = sma - self.config.bb_std * std
        bb_range = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
        df["bb_position"] = (close - df["bb_lower"]) / bb_range
        df["bb_width"] = bb_range / sma

        return df

    def _volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "volume" not in df.columns:
            return df

        volume = df["volume"]
        close = df["close"]

        for w in self.config.volume_ma_windows:
            vol_ma = volume.rolling(w).mean()
            df[f"volume_ma_{w}"] = vol_ma
            df[f"volume_ratio_{w}"] = volume / vol_ma.replace(0, np.nan)

        direction = np.sign(close.diff())
        df["obv"] = (volume * direction).cumsum()

        df["vwap"] = (close * volume).cumsum() / volume.cumsum().replace(0, np.nan)
        df["vwap_deviation"] = (close - df["vwap"]) / df["vwap"].replace(0, np.nan)

        df["volume_price_trend"] = (volume * close.pct_change()).cumsum()

        return df

    def _pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]

        for w in [5, 10, 20]:
            df[f"return_{w}d"] = close.pct_change(w)

        for lag in range(1, 6):
            df[f"return_lag_{lag}"] = df["returns"].shift(lag)

        df["return_skew_20"] = df["returns"].rolling(20).skew()
        df["return_kurtosis_20"] = df["returns"].rolling(20).kurt()

        return df


class BehavioralFeatures:
    """
    Features derived from your personal trading log.
    Captures patterns in YOUR trading behavior to feed into the prediction model.
    """

    @staticmethod
    def compute(trades_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
        if trades_df.empty:
            return market_df

        features = market_df.copy()

        if "timestamp" in trades_df.columns and "pnl" in trades_df.columns:
            trade_ts = trades_df.copy()
            if "is_winner" not in trade_ts.columns:
                trade_ts["is_winner"] = trade_ts["pnl"] > 0
            trade_ts = trade_ts.set_index("timestamp")

            for w in [10, 20, 50]:
                win_rate = trade_ts["is_winner"].rolling(f"{w}D").mean()
                win_rate_resampled = win_rate.resample(
                    market_df.index.freq or "1min"
                ).ffill()
                col_name = f"trader_win_rate_{w}d"
                features[col_name] = win_rate_resampled.reindex(features.index, method="ffill")

            if "pnl" in trade_ts.columns:
                cummax = trade_ts["pnl"].cumsum().cummax()
                drawdown = trade_ts["pnl"].cumsum() - cummax
                dd_resampled = drawdown.resample(
                    market_df.index.freq or "1min"
                ).ffill()
                features["trader_drawdown"] = dd_resampled.reindex(features.index, method="ffill")

            if "quantity" in trades_df.columns:
                avg_size = trade_ts["quantity"].rolling("5D").mean()
                size_resampled = avg_size.resample(
                    market_df.index.freq or "1min"
                ).ffill()
                features["trader_avg_size_5d"] = size_resampled.reindex(features.index, method="ffill")

        return features


class FeatureSelector:
    """Reduces feature dimensionality based on importance and correlation."""

    def __init__(self, config: FeatureConfig):
        self.config = config
        self.selected_features: Optional[List[str]] = None

    def fit_select(
        self,
        df: pd.DataFrame,
        target_col: str = "returns",
        method: str = "correlation"
    ) -> List[str]:
        feature_cols = [
            c for c in df.columns
            if c not in ["open", "high", "low", "close", "volume", target_col]
            and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
        ]

        if method == "correlation":
            correlations = df[feature_cols].corrwith(df[target_col]).abs()
            correlations = correlations.dropna().sort_values(ascending=False)
            selected = correlations.head(self.config.max_features).index.tolist()
        else:
            selected = feature_cols[:self.config.max_features]

        corr_matrix = df[selected].corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.95)]
        selected = [f for f in selected if f not in to_drop]

        self.selected_features = selected
        logger.info(f"Selected {len(selected)} features from {len(feature_cols)} candidates")
        return selected

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.selected_features is None:
            raise RuntimeError("Must call fit_select before transform")
        available = [f for f in self.selected_features if f in df.columns]
        return df[available]


class FeatureEngine:
    """Main feature engineering pipeline. Orchestrates all feature computation."""

    def __init__(self, config: FeatureConfig):
        self.config = config
        self.technical = TechnicalFeatures(config)
        self.selector = FeatureSelector(config)

    def build_features(
        self,
        market_data: pd.DataFrame,
        trading_logs: Optional[pd.DataFrame] = None,
        fit_selector: bool = True,
    ) -> Tuple[pd.DataFrame, List[str]]:
        logger.info("Computing technical features...")
        df = self.technical.compute_all(market_data)

        if trading_logs is not None and not trading_logs.empty:
            logger.info("Computing behavioral features...")
            df = BehavioralFeatures.compute(trading_logs, df)

        initial_nans = df.isnull().sum().sum()
        df = df.ffill().bfill()
        remaining_nans = df.isnull().sum().sum()
        if remaining_nans > 0:
            df = df.dropna(axis=1, how="all")
            df = df.fillna(0)

        logger.info(f"Total features computed: {len(df.columns)}")

        if fit_selector:
            if "returns" not in df.columns:
                df["returns"] = df["close"].pct_change()
            selected = self.selector.fit_select(df, target_col="returns")
        else:
            selected = self.selector.selected_features or list(df.columns)

        return df, selected
