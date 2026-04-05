"""
Base model interface that all prediction models must implement.
Ensures consistent API across LSTM, TCN, XGBoost, and other models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd


class BasePredictor(ABC):
    """Abstract base class for all prediction models in the ensemble."""

    def __init__(self, name: str, horizon: str):
        self.name = name
        self.horizon = horizon
        self.is_fitted = False
        self._training_history: list = []

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Train the model. Returns training metrics."""
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions.
        Returns:
            predictions: array of predicted values
            confidence: array of confidence scores [0, 1]
        """
        ...

    @abstractmethod
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Return feature importance scores if available."""
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Serialize model to disk."""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from disk."""
        ...

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        predictions, confidence = self.predict(X)
        mae = np.mean(np.abs(predictions - y_true))
        mse = np.mean((predictions - y_true) ** 2)
        rmse = np.sqrt(mse)

        ss_res = np.sum((y_true - predictions) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        direction_pred = np.sign(predictions)
        direction_actual = np.sign(y_true)
        directional_accuracy = np.mean(direction_pred == direction_actual)

        # Sharpe-like metric: mean prediction accuracy / std of errors
        errors = predictions - y_true
        ic = np.corrcoef(predictions, y_true)[0, 1] if len(predictions) > 1 else 0.0

        return {
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "r2": float(r2),
            "directional_accuracy": float(directional_accuracy),
            "information_coefficient": float(ic) if not np.isnan(ic) else 0.0,
            "mean_confidence": float(np.mean(confidence)),
        }
