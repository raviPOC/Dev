"""
Data Ingestion Layer
====================
Handles loading, validating, and normalizing trading data from multiple sources.
Supports personal trading logs, OHLCV market data, and custom formats.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union

import numpy as np
import pandas as pd

from ..config.settings import DataConfig

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates incoming data for integrity and flags anomalies."""

    REQUIRED_OHLCV_COLS = {"open", "high", "low", "close", "volume"}

    @staticmethod
    def validate_ohlcv(df: pd.DataFrame) -> Dict[str, Any]:
        issues = []

        missing_cols = DataValidator.REQUIRED_OHLCV_COLS - set(df.columns.str.lower())
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")

        if df.isnull().any().any():
            null_counts = df.isnull().sum()
            null_cols = null_counts[null_counts > 0].to_dict()
            issues.append(f"Null values found: {null_cols}")

        if "high" in df.columns and "low" in df.columns:
            violations = (df["high"] < df["low"]).sum()
            if violations > 0:
                issues.append(f"High < Low in {violations} rows")

        if "close" in df.columns:
            pct_change = df["close"].pct_change().abs()
            extreme_moves = (pct_change > 0.5).sum()  # >50% single-bar moves
            if extreme_moves > 0:
                issues.append(f"{extreme_moves} bars with >50% price change (possible data error)")

        if "volume" in df.columns:
            neg_volume = (df["volume"] < 0).sum()
            if neg_volume > 0:
                issues.append(f"Negative volume in {neg_volume} rows")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "row_count": len(df),
            "date_range": (df.index.min(), df.index.max()) if isinstance(df.index, pd.DatetimeIndex) else None,
        }


class TradingLogParser:
    """
    Parses personal trading logs into a standardized format.
    Handles common export formats from brokers and trading platforms.
    """

    STANDARD_COLUMNS = [
        "timestamp", "symbol", "side", "quantity", "price",
        "pnl", "commission", "strategy", "notes"
    ]

    @staticmethod
    def parse(filepath: Union[str, Path], format_hint: Optional[str] = None) -> pd.DataFrame:
        filepath = Path(filepath)
        suffix = filepath.suffix.lower()

        if suffix == ".csv" or format_hint == "csv":
            df = pd.read_csv(filepath)
        elif suffix == ".json" or format_hint == "json":
            df = pd.read_json(filepath)
        elif suffix == ".parquet" or format_hint == "parquet":
            df = pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

        df = TradingLogParser._normalize_columns(df)
        df = TradingLogParser._parse_timestamps(df)
        df = TradingLogParser._compute_derived_fields(df)

        logger.info(f"Parsed {len(df)} trades from {filepath}")
        return df

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        col_aliases = {
            "date": "timestamp", "time": "timestamp", "datetime": "timestamp",
            "trade_date": "timestamp", "exec_time": "timestamp",
            "ticker": "symbol", "instrument": "symbol", "asset": "symbol",
            "direction": "side", "action": "side", "type": "side",
            "qty": "quantity", "size": "quantity", "amount": "quantity", "shares": "quantity",
            "fill_price": "price", "avg_price": "price", "exec_price": "price",
            "profit": "pnl", "profit_loss": "pnl", "realized_pnl": "pnl", "pl": "pnl",
            "fee": "commission", "fees": "commission",
        }
        df = df.rename(columns={k: v for k, v in col_aliases.items() if k in df.columns})
        return df

    @staticmethod
    def _parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], infer_datetime_format=True, utc=True)
            df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    @staticmethod
    def _compute_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
        if "side" in df.columns:
            df["side"] = df["side"].str.lower().str.strip()
            df["side"] = df["side"].replace({
                "buy": "long", "b": "long", "long": "long",
                "sell": "short", "s": "short", "short": "short",
                "sell_short": "short", "cover": "long", "buy_to_cover": "long",
            })

        if "pnl" in df.columns and "quantity" in df.columns:
            mask = df["quantity"] != 0
            df.loc[mask, "pnl_per_unit"] = df.loc[mask, "pnl"] / df.loc[mask, "quantity"]

        if "pnl" in df.columns:
            df["cumulative_pnl"] = df["pnl"].cumsum()
            df["is_winner"] = df["pnl"] > 0

        if "timestamp" in df.columns:
            df["hour_of_day"] = df["timestamp"].dt.hour
            df["day_of_week"] = df["timestamp"].dt.dayofweek
            df["is_market_open_hour"] = df["hour_of_day"].between(9, 10)
            df["is_market_close_hour"] = df["hour_of_day"].between(15, 16)

        return df


class MarketDataLoader:
    """Loads and normalizes OHLCV market data."""

    def __init__(self, config: DataConfig):
        self.config = config
        self.validator = DataValidator()

    def load(self, filepath: Union[str, Path]) -> pd.DataFrame:
        filepath = Path(filepath)
        suffix = filepath.suffix.lower()

        if suffix == ".csv":
            df = pd.read_csv(filepath, parse_dates=True, index_col=0)
        elif suffix == ".parquet":
            df = pd.read_parquet(filepath)
            if not isinstance(df.index, pd.DatetimeIndex) and "timestamp" in df.columns:
                df = df.set_index("timestamp")
        elif suffix == ".json":
            df = pd.read_json(filepath)
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
        else:
            raise ValueError(f"Unsupported format: {suffix}")

        df.columns = df.columns.str.lower().str.strip()
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index()

        validation = self.validator.validate_ohlcv(df)
        if not validation["valid"]:
            for issue in validation["issues"]:
                logger.warning(f"Data issue: {issue}")

        df = self._clean(df)

        logger.info(
            f"Loaded {len(df)} bars from {filepath} "
            f"({validation['date_range'][0]} to {validation['date_range'][1]})"
        )
        return df

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = ["open", "high", "low", "close", "volume"]
        existing = [c for c in numeric_cols if c in df.columns]
        df[existing] = df[existing].ffill().bfill()

        if "volume" in df.columns:
            df["volume"] = df["volume"].clip(lower=0)

        df = df[~df.index.duplicated(keep="first")]

        return df
