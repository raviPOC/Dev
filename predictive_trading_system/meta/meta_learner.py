"""
Meta-Learner
============
Dynamically weights ensemble models based on:
1. Recent prediction accuracy per model
2. Current market regime
3. Model confidence scores

This is the "brain" that makes the system adaptive. It tracks which models
perform best under which conditions and adjusts accordingly.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config.settings import MetaLearnerConfig, MarketRegime
from ..models.base import BasePredictor
from ..models.regime_detector import RegimeDetector

logger = logging.getLogger(__name__)


class ModelPerformanceTracker:
    """Tracks rolling performance metrics for each model."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._errors: Dict[str, List[float]] = defaultdict(list)
        self._directional_hits: Dict[str, List[bool]] = defaultdict(list)
        self._regime_errors: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

    def record(
        self,
        model_name: str,
        prediction: float,
        actual: float,
        regime: MarketRegime,
    ) -> None:
        error = abs(prediction - actual)
        self._errors[model_name].append(error)
        if len(self._errors[model_name]) > self.window_size:
            self._errors[model_name] = self._errors[model_name][-self.window_size:]

        direction_correct = (prediction > 0) == (actual > 0)
        self._directional_hits[model_name].append(direction_correct)
        if len(self._directional_hits[model_name]) > self.window_size:
            self._directional_hits[model_name] = self._directional_hits[model_name][-self.window_size:]

        regime_key = regime.value
        self._regime_errors[model_name][regime_key].append(error)
        if len(self._regime_errors[model_name][regime_key]) > self.window_size:
            self._regime_errors[model_name][regime_key] = \
                self._regime_errors[model_name][regime_key][-self.window_size:]

    def get_recent_mae(self, model_name: str) -> float:
        errors = self._errors.get(model_name, [])
        return np.mean(errors) if errors else float("inf")

    def get_directional_accuracy(self, model_name: str) -> float:
        hits = self._directional_hits.get(model_name, [])
        return np.mean(hits) if hits else 0.5

    def get_regime_mae(self, model_name: str, regime: MarketRegime) -> float:
        regime_errors = self._regime_errors.get(model_name, {}).get(regime.value, [])
        return np.mean(regime_errors) if regime_errors else float("inf")


class MetaLearner:
    """
    Adaptive ensemble combiner.
    Weights models based on recent performance, regime context, and model confidence.
    """

    def __init__(self, config: MetaLearnerConfig):
        self.config = config
        self.regime_detector = RegimeDetector(config)
        self.tracker = ModelPerformanceTracker(window_size=config.performance_window)
        self.model_weights: Dict[str, float] = {}
        self.current_regime: MarketRegime = MarketRegime.UNKNOWN
        self._update_counter = 0

    def initialize_weights(self, model_names: List[str]) -> None:
        n = len(model_names)
        for name in model_names:
            self.model_weights[name] = 1.0 / n
        logger.info(f"Initialized equal weights for {n} models")

    def fit_regime_detector(self, returns: np.ndarray, volatility: np.ndarray) -> None:
        self.regime_detector.fit(returns, volatility)

    def update(
        self,
        predictions: Dict[str, Tuple[float, float]],
        actual: float,
        returns_window: np.ndarray,
        volatility_window: np.ndarray,
    ) -> None:
        """
        Called after each prediction round to update model weights.

        Args:
            predictions: {model_name: (prediction, confidence)}
            actual: the realized value
            returns_window: recent returns for regime detection
            volatility_window: recent volatility for regime detection
        """
        self.current_regime = self.regime_detector.detect(returns_window, volatility_window)

        for model_name, (pred, conf) in predictions.items():
            self.tracker.record(model_name, pred, actual, self.current_regime)

        self._update_counter += 1
        if self._update_counter % self.config.weight_update_frequency == 0:
            self._recompute_weights()

    def combine_predictions(
        self,
        predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Combine model predictions using adaptive weights.

        Args:
            predictions: {model_name: (predictions_array, confidence_array)}

        Returns:
            combined_predictions, combined_confidence
        """
        if not predictions:
            raise ValueError("No predictions to combine")

        first_key = next(iter(predictions))
        n_samples = len(predictions[first_key][0])

        weighted_preds = np.zeros(n_samples)
        weighted_conf = np.zeros(n_samples)
        total_weight = 0.0

        for model_name, (preds, conf) in predictions.items():
            w = self.model_weights.get(model_name, 1.0 / len(predictions))

            # Weight by both meta-learner weight AND model's own confidence
            effective_weight = w * np.mean(conf)
            weighted_preds += effective_weight * preds
            weighted_conf += effective_weight * conf
            total_weight += effective_weight

        if total_weight > 0:
            weighted_preds /= total_weight
            weighted_conf /= total_weight
        else:
            # Equal weighting fallback
            for preds, conf in predictions.values():
                weighted_preds += preds / len(predictions)
                weighted_conf += conf / len(predictions)

        return weighted_preds, weighted_conf

    def _recompute_weights(self) -> None:
        if not self.model_weights:
            return

        raw_weights = {}
        for model_name in self.model_weights:
            mae = self.tracker.get_recent_mae(model_name)
            dir_acc = self.tracker.get_directional_accuracy(model_name)
            regime_mae = self.tracker.get_regime_mae(model_name, self.current_regime)

            if mae == float("inf") or regime_mae == float("inf"):
                raw_weights[model_name] = 1.0 / len(self.model_weights)
                continue

            # Composite score: lower MAE is better, higher directional accuracy is better
            # Regime-specific MAE is weighted more heavily
            inverse_mae = 1.0 / (mae + 1e-10)
            inverse_regime_mae = 1.0 / (regime_mae + 1e-10)
            score = 0.3 * inverse_mae + 0.3 * inverse_regime_mae + 0.4 * dir_acc
            raw_weights[model_name] = max(score, 1e-10)

        total = sum(raw_weights.values())
        for model_name in raw_weights:
            weight = raw_weights[model_name] / total
            weight = max(weight, self.config.min_model_weight)
            self.model_weights[model_name] = weight

        # Renormalize after floor
        total = sum(self.model_weights.values())
        for model_name in self.model_weights:
            self.model_weights[model_name] /= total

        logger.debug(f"Updated weights (regime={self.current_regime.value}): {self.model_weights}")

    def get_diagnostics(self) -> Dict:
        return {
            "current_regime": self.current_regime.value,
            "model_weights": dict(self.model_weights),
            "model_performance": {
                name: {
                    "recent_mae": self.tracker.get_recent_mae(name),
                    "directional_accuracy": self.tracker.get_directional_accuracy(name),
                }
                for name in self.model_weights
            },
        }
