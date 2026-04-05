"""
Feedback Loop
=============
Monitors prediction performance in real-time, detects degradation,
and triggers model retraining when necessary.
Logs all predictions with context for post-hoc analysis.
"""

import logging
import os
from collections import deque
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd

from ..config.settings import FeedbackConfig

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Rolling performance monitor that detects model degradation.
    Tracks MAE, directional accuracy, and information coefficient
    over a sliding window.
    """

    def __init__(self, window_size: int = 200):
        self.window_size = window_size
        self._predictions = deque(maxlen=window_size)
        self._actuals = deque(maxlen=window_size)
        self._timestamps = deque(maxlen=window_size)
        self._confidences = deque(maxlen=window_size)

    def record(
        self,
        prediction: float,
        actual: float,
        confidence: float,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> None:
        self._predictions.append(prediction)
        self._actuals.append(actual)
        self._confidences.append(confidence)
        self._timestamps.append(timestamp or pd.Timestamp.now(tz="UTC"))

    def get_metrics(self) -> Dict[str, float]:
        if len(self._predictions) < 10:
            return {"status": "insufficient_data", "n_samples": len(self._predictions)}

        preds = np.array(self._predictions)
        actuals = np.array(self._actuals)
        confs = np.array(self._confidences)

        errors = preds - actuals
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))

        direction_correct = np.sign(preds) == np.sign(actuals)
        dir_acc = np.mean(direction_correct)

        ic = np.corrcoef(preds, actuals)[0, 1] if len(preds) > 1 else 0.0
        if np.isnan(ic):
            ic = 0.0

        # Confidence calibration: are high-confidence predictions more accurate?
        if len(confs) > 20:
            high_conf_mask = confs > np.median(confs)
            low_conf_mask = ~high_conf_mask
            high_conf_mae = np.mean(np.abs(errors[high_conf_mask])) if high_conf_mask.sum() > 0 else mae
            low_conf_mae = np.mean(np.abs(errors[low_conf_mask])) if low_conf_mask.sum() > 0 else mae
            calibration = high_conf_mae / (low_conf_mae + 1e-10)
        else:
            calibration = 1.0

        # Recent performance trend (is it getting worse?)
        half = len(errors) // 2
        if half > 5:
            recent_mae = np.mean(np.abs(errors[half:]))
            older_mae = np.mean(np.abs(errors[:half]))
            degradation_ratio = recent_mae / (older_mae + 1e-10)
        else:
            degradation_ratio = 1.0

        return {
            "mae": float(mae),
            "rmse": float(rmse),
            "directional_accuracy": float(dir_acc),
            "information_coefficient": float(ic),
            "confidence_calibration": float(calibration),
            "degradation_ratio": float(degradation_ratio),
            "n_samples": len(preds),
        }

    @property
    def is_degrading(self) -> bool:
        metrics = self.get_metrics()
        if "status" in metrics:
            return False
        return metrics.get("degradation_ratio", 1.0) > 1.5


class PredictionLogger:
    """Logs every prediction with full context for analysis and debugging."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        self._buffer: List[Dict] = []
        self._flush_interval = 100

        os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else ".", exist_ok=True)

    def log(
        self,
        prediction: float,
        confidence: float,
        actual: Optional[float] = None,
        regime: str = "unknown",
        model_weights: Optional[Dict[str, float]] = None,
        feature_snapshot: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        entry = {
            "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
            "prediction": prediction,
            "confidence": confidence,
            "actual": actual,
            "regime": regime,
        }

        if model_weights:
            for name, weight in model_weights.items():
                entry[f"weight_{name}"] = weight

        if feature_snapshot:
            for name, value in list(feature_snapshot.items())[:10]:
                entry[f"feat_{name}"] = value

        if metadata:
            entry.update(metadata)

        self._buffer.append(entry)

        if len(self._buffer) >= self._flush_interval:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return

        df = pd.DataFrame(self._buffer)

        if os.path.exists(self.log_path):
            df.to_csv(self.log_path, mode="a", header=False, index=False)
        else:
            df.to_csv(self.log_path, index=False)

        self._buffer.clear()

    def load_history(self) -> pd.DataFrame:
        if os.path.exists(self.log_path):
            return pd.read_csv(self.log_path)
        return pd.DataFrame()


class FeedbackLoop:
    """
    Coordinates performance monitoring, logging, and retraining decisions.
    This is the self-learning component — it decides when the system
    needs to adapt and how.
    """

    def __init__(self, config: FeedbackConfig):
        self.config = config
        self.monitor = PerformanceMonitor()
        self.pred_logger = PredictionLogger(config.prediction_log_path)
        self._bars_since_retrain = 0
        self._retrain_count = 0

    def on_prediction(
        self,
        prediction: float,
        confidence: float,
        regime: str = "unknown",
        model_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.pred_logger.log(
            prediction=prediction,
            confidence=confidence,
            regime=regime,
            model_weights=model_weights,
        )

    def on_actual(self, prediction: float, actual: float, confidence: float) -> Dict[str, Any]:
        self.monitor.record(prediction, actual, confidence)
        self._bars_since_retrain += 1

        metrics = self.monitor.get_metrics()
        should_retrain = self._should_retrain(metrics)

        if should_retrain:
            self._retrain_count += 1
            self._bars_since_retrain = 0
            logger.info(
                f"Retraining triggered (#{self._retrain_count}). "
                f"MAE={metrics.get('mae', 'N/A'):.6f}, "
                f"Degradation={metrics.get('degradation_ratio', 'N/A'):.2f}"
            )

        return {
            "metrics": metrics,
            "should_retrain": should_retrain,
            "bars_since_retrain": self._bars_since_retrain,
            "total_retrains": self._retrain_count,
        }

    def _should_retrain(self, metrics: Dict[str, Any]) -> bool:
        if "status" in metrics:
            return False

        if self._bars_since_retrain < self.config.retrain_min_interval:
            return False

        mae = metrics.get("mae", 0)
        if mae > self.config.retrain_error_threshold:
            return True

        degradation = metrics.get("degradation_ratio", 1.0)
        if degradation > 2.0:
            return True

        return False

    def get_status(self) -> Dict[str, Any]:
        return {
            "performance_metrics": self.monitor.get_metrics(),
            "is_degrading": self.monitor.is_degrading,
            "bars_since_retrain": self._bars_since_retrain,
            "total_retrains": self._retrain_count,
        }

    def close(self) -> None:
        self.pred_logger.flush()
