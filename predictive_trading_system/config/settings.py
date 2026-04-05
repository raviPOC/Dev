"""
Central configuration for the Predictive Trading System.
All tunable hyperparameters and system settings live here.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class TimeHorizon(Enum):
    SHORT = "short"       # 1-60 minutes
    MEDIUM = "medium"     # 1 hour - 1 day
    LONG = "long"         # 1 day - 1 week


class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    UNKNOWN = "unknown"


@dataclass
class DataConfig:
    lookback_window: int = 252          # trading days of history to use
    min_data_points: int = 100          # minimum rows before training
    train_test_split: float = 0.8
    purge_gap: int = 5                  # rows to purge between train/test to avoid leakage
    resample_freq: str = "1min"         # base data frequency
    supported_formats: List[str] = field(default_factory=lambda: ["csv", "json", "parquet"])


@dataclass
class FeatureConfig:
    ma_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100, 200])
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    volume_ma_windows: List[int] = field(default_factory=lambda: [5, 10, 20])
    max_features: int = 100             # feature selection cap
    feature_importance_threshold: float = 0.01


@dataclass
class ModelConfig:
    # LSTM / Temporal models
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2
    lstm_sequence_length: int = 60

    # TCN
    tcn_num_channels: List[int] = field(default_factory=lambda: [64, 64, 64, 64])
    tcn_kernel_size: int = 3
    tcn_dropout: float = 0.2

    # XGBoost
    xgb_n_estimators: int = 500
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.01
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-3
    max_epochs: int = 100
    early_stopping_patience: int = 10
    weight_decay: float = 1e-5


@dataclass
class MetaLearnerConfig:
    regime_lookback: int = 60           # bars to determine current regime
    weight_update_frequency: int = 10   # how often to re-weight ensemble (in bars)
    min_model_weight: float = 0.05      # floor weight — no model fully zeroed out
    performance_window: int = 100       # recent predictions to evaluate
    regime_change_threshold: float = 0.3
    hmm_n_states: int = 4


@dataclass
class FeedbackConfig:
    retrain_error_threshold: float = 0.15   # MAE threshold to trigger retraining
    retrain_min_interval: int = 1000         # min bars between retrains
    performance_log_path: str = "logs/performance.csv"
    prediction_log_path: str = "logs/predictions.csv"
    feature_drift_threshold: float = 0.2


@dataclass
class SystemConfig:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    meta: MetaLearnerConfig = field(default_factory=MetaLearnerConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    horizons: List[TimeHorizon] = field(
        default_factory=lambda: [TimeHorizon.SHORT, TimeHorizon.MEDIUM, TimeHorizon.LONG]
    )
    device: str = "cpu"  # "cpu" or "cuda"
    random_seed: int = 42
    log_level: str = "INFO"


def get_default_config() -> SystemConfig:
    return SystemConfig()
