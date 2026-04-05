"""
Core Pipeline
=============
Orchestrates the full prediction flow:
  Data Ingestion → Feature Engineering → Model Training → Prediction → Feedback

This is the main entry point for running the system.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from ..config.settings import SystemConfig, get_default_config, TimeHorizon
from ..data.ingestion import MarketDataLoader, TradingLogParser
from ..features.engineering import FeatureEngine
from ..models.base import BasePredictor
from ..models.lstm_model import LSTMPredictor
from ..models.tcn_model import TCNPredictor
from ..models.xgboost_model import XGBoostPredictor
from ..meta.meta_learner import MetaLearner
from ..core.feedback import FeedbackLoop

logger = logging.getLogger(__name__)


class SequenceBuilder:
    """Converts flat feature DataFrames into sequences for temporal models."""

    @staticmethod
    def build(
        features: np.ndarray,
        targets: np.ndarray,
        sequence_length: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if len(features) <= sequence_length:
            raise ValueError(
                f"Not enough data ({len(features)}) for sequence length {sequence_length}"
            )

        X, y = [], []
        for i in range(sequence_length, len(features)):
            X.append(features[i - sequence_length: i])
            y.append(targets[i])

        return np.array(X), np.array(y)


class WalkForwardSplitter:
    """
    Walk-forward cross-validation with purging.
    Prevents lookahead bias by always training on past, testing on future,
    with a gap between train and test sets.
    """

    def __init__(self, n_splits: int = 5, train_ratio: float = 0.7, purge_gap: int = 5):
        self.n_splits = n_splits
        self.train_ratio = train_ratio
        self.purge_gap = purge_gap

    def split(self, n_samples: int):
        fold_size = n_samples // (self.n_splits + 1)

        for i in range(self.n_splits):
            train_end = fold_size * (i + 1)
            test_start = train_end + self.purge_gap
            test_end = min(test_start + fold_size, n_samples)

            if test_start >= n_samples or test_end <= test_start:
                continue

            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            yield train_idx, test_idx


class PredictionPipeline:
    """
    Main orchestrator for the entire prediction system.

    Usage:
        config = get_default_config()
        pipeline = PredictionPipeline(config)
        pipeline.load_data("market_data.csv", "trading_logs.csv")
        pipeline.train()
        predictions = pipeline.predict()
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or get_default_config()
        self.feature_engine = FeatureEngine(self.config.features)
        self.meta_learner = MetaLearner(self.config.meta)
        self.feedback = FeedbackLoop(self.config.feedback)
        self.models: Dict[str, BasePredictor] = {}
        self.market_data: Optional[pd.DataFrame] = None
        self.trading_logs: Optional[pd.DataFrame] = None
        self.feature_df: Optional[pd.DataFrame] = None
        self.selected_features: Optional[List[str]] = None
        self._is_trained = False

    def load_data(
        self,
        market_data_path: str,
        trading_logs_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        loader = MarketDataLoader(self.config.data)
        self.market_data = loader.load(market_data_path)

        if trading_logs_path and os.path.exists(trading_logs_path):
            self.trading_logs = TradingLogParser.parse(trading_logs_path)
            logger.info(f"Loaded {len(self.trading_logs)} trades from logs")
        else:
            self.trading_logs = pd.DataFrame()

        return {
            "market_bars": len(self.market_data),
            "trades_loaded": len(self.trading_logs) if self.trading_logs is not None else 0,
            "date_range": (
                str(self.market_data.index.min()),
                str(self.market_data.index.max()),
            ),
        }

    def engineer_features(self) -> Dict[str, Any]:
        if self.market_data is None:
            raise RuntimeError("Must load data before engineering features")

        self.feature_df, self.selected_features = self.feature_engine.build_features(
            self.market_data,
            self.trading_logs,
            fit_selector=True,
        )

        return {
            "total_features": len(self.feature_df.columns),
            "selected_features": len(self.selected_features),
            "feature_names": self.selected_features[:20],
        }

    def _build_models(self, n_features: int) -> None:
        seq_len = self.config.model.lstm_sequence_length

        self.models = {
            "tcn": TCNPredictor(
                input_size=n_features,
                config=self.config.model,
                device=self.config.device,
            ),
            "lstm": LSTMPredictor(
                input_size=n_features,
                config=self.config.model,
                device=self.config.device,
            ),
            "xgboost": XGBoostPredictor(
                config=self.config.model,
                feature_names=self.selected_features,
            ),
        }

        self.meta_learner.initialize_weights(list(self.models.keys()))

    def _prepare_targets(self, horizons: Optional[Dict[str, int]] = None) -> Dict[str, np.ndarray]:
        if horizons is None:
            horizons = {
                "short": 5,      # 5-bar forward return
                "medium": 20,    # 20-bar forward return
                "long": 60,      # 60-bar forward return
            }

        targets = {}
        close = self.feature_df["close"]
        for name, period in horizons.items():
            fwd_return = close.pct_change(period).shift(-period)
            targets[name] = fwd_return.values

        return targets

    def train(
        self,
        target_horizon: str = "medium",
        target_periods: int = 20,
    ) -> Dict[str, Any]:
        if self.feature_df is None:
            self.engineer_features()

        targets = self._prepare_targets()
        target = targets[target_horizon]

        feature_matrix = self.feature_df[self.selected_features].values

        valid_mask = ~np.isnan(target)
        valid_mask[:max(self.config.model.lstm_sequence_length, 200)] = False
        valid_mask[-target_periods:] = False

        feature_matrix = feature_matrix[valid_mask]
        target = target[valid_mask]

        n_features = feature_matrix.shape[1]
        self._build_models(n_features)

        # Train/val split
        split_idx = int(len(feature_matrix) * self.config.data.train_test_split)
        purge = self.config.data.purge_gap

        X_train_flat = feature_matrix[:split_idx]
        y_train_flat = target[:split_idx]
        X_val_flat = feature_matrix[split_idx + purge:]
        y_val_flat = target[split_idx + purge:]

        # Build sequences for temporal models
        seq_len = self.config.model.lstm_sequence_length
        X_train_seq, y_train_seq = SequenceBuilder.build(X_train_flat, y_train_flat, seq_len)
        X_val_seq, y_val_seq = SequenceBuilder.build(X_val_flat, y_val_flat, seq_len)

        results = {}

        # Train temporal models (LSTM, TCN) on sequences
        for name in ["lstm", "tcn"]:
            logger.info(f"Training {name}...")
            result = self.models[name].fit(X_train_seq, y_train_seq, X_val_seq, y_val_seq)
            results[name] = result
            logger.info(f"{name} training result: {result}")

        # Train XGBoost on flat features
        logger.info("Training XGBoost...")
        result = self.models["xgboost"].fit(X_train_flat, y_train_flat, X_val_flat, y_val_flat)
        results["xgboost"] = result
        logger.info(f"XGBoost training result: {result}")

        # Fit regime detector
        if "returns" in self.feature_df.columns and "realized_vol_20" in self.feature_df.columns:
            returns = self.feature_df["returns"].dropna().values
            vol = self.feature_df["realized_vol_20"].dropna().values
            min_len = min(len(returns), len(vol))
            self.meta_learner.fit_regime_detector(returns[:min_len], vol[:min_len])

        # Evaluate on validation
        eval_results = {}
        for name, model in self.models.items():
            if name in ["lstm", "tcn"]:
                eval_results[name] = model.evaluate(X_val_seq, y_val_seq)
            else:
                eval_results[name] = model.evaluate(X_val_flat, y_val_flat)

        self._is_trained = True

        return {
            "training_results": results,
            "validation_metrics": eval_results,
            "data_shape": {
                "train_samples": len(y_train_flat),
                "val_samples": len(y_val_flat),
                "n_features": n_features,
            },
        }

    def predict(self, latest_features: Optional[np.ndarray] = None) -> Dict[str, Any]:
        if not self._is_trained:
            raise RuntimeError("Must train before predicting")

        if latest_features is None:
            feature_matrix = self.feature_df[self.selected_features].values
            latest_features = feature_matrix[-self.config.model.lstm_sequence_length:]

        predictions = {}

        # Sequence input for temporal models
        seq_input = latest_features.reshape(1, *latest_features.shape)

        for name, model in self.models.items():
            if name in ["lstm", "tcn"]:
                preds, conf = model.predict(seq_input)
            else:
                flat_input = latest_features[-1:].reshape(1, -1)
                preds, conf = model.predict(flat_input)

            predictions[name] = (preds, conf)

        combined_pred, combined_conf = self.meta_learner.combine_predictions(predictions)

        return {
            "ensemble_prediction": float(combined_pred[0]),
            "ensemble_confidence": float(combined_conf[0]),
            "individual_predictions": {
                name: {
                    "prediction": float(p[0]),
                    "confidence": float(c[0]),
                    "weight": self.meta_learner.model_weights.get(name, 0),
                }
                for name, (p, c) in predictions.items()
            },
            "regime": self.meta_learner.current_regime.value,
            "diagnostics": self.meta_learner.get_diagnostics(),
        }

    def run_backtest(
        self,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Walk-forward backtest: train on expanding window, predict next bar.
        Honest evaluation with no lookahead.
        """
        if self.feature_df is None:
            self.engineer_features()

        feature_matrix = self.feature_df[self.selected_features].values
        close = self.feature_df["close"].values
        forward_returns = np.concatenate([np.diff(close) / close[:-1], [0]])

        seq_len = self.config.model.lstm_sequence_length
        min_train = 500

        start = start_idx or (min_train + seq_len)
        end = end_idx or len(feature_matrix) - 1

        results = []

        for i in range(start, end):
            if (i - start) % 100 == 0:
                logger.info(f"Backtest progress: {i - start}/{end - start}")

            train_features = feature_matrix[:i]
            train_targets = forward_returns[:i]

            current_seq = feature_matrix[i - seq_len: i]
            actual_return = forward_returns[i]

            try:
                # Retrain periodically
                if (i - start) % self.config.meta.weight_update_frequency == 0:
                    X_seq, y_seq = SequenceBuilder.build(train_features, train_targets, seq_len)
                    split = int(len(X_seq) * 0.8)

                    for name in ["lstm", "tcn"]:
                        self.models[name].fit(
                            X_seq[:split], y_seq[:split],
                            X_seq[split:], y_seq[split:]
                        )

                    self.models["xgboost"].fit(
                        train_features[seq_len:split + seq_len],
                        train_targets[seq_len:split + seq_len],
                        train_features[split + seq_len:],
                        train_targets[split + seq_len:],
                    )

                pred_result = self.predict(current_seq)

                results.append({
                    "bar_index": i,
                    "actual_return": actual_return,
                    "predicted_return": pred_result["ensemble_prediction"],
                    "confidence": pred_result["ensemble_confidence"],
                    "regime": pred_result["regime"],
                    **{
                        f"{name}_pred": pred_result["individual_predictions"][name]["prediction"]
                        for name in self.models
                    },
                })

            except Exception as e:
                logger.warning(f"Backtest error at bar {i}: {e}")
                continue

        return pd.DataFrame(results)
