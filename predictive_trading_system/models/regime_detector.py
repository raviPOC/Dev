"""
Market Regime Detection
=======================
Uses Hidden Markov Models and volatility clustering to identify
the current market regime (trending, mean-reverting, high/low vol).
Regime state feeds into the meta-learner for ensemble weight adjustment.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..config.settings import MetaLearnerConfig, MarketRegime

logger = logging.getLogger(__name__)

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logger.warning("hmmlearn not available. Using volatility-based regime detection fallback.")


class RegimeDetector:
    """Detects market regime using HMM or heuristic fallback."""

    def __init__(self, config: MetaLearnerConfig):
        self.config = config
        self.hmm_model = None
        self.regime_mapping: Dict[int, MarketRegime] = {}
        self.is_fitted = False

    def fit(self, returns: np.ndarray, volatility: np.ndarray) -> None:
        features = np.column_stack([returns, volatility])
        features = features[~np.isnan(features).any(axis=1)]

        if len(features) < 50:
            logger.warning("Not enough data for regime detection. Using heuristic.")
            self.is_fitted = False
            return

        if HMM_AVAILABLE:
            self._fit_hmm(features)
        else:
            self._fit_heuristic(returns, volatility)

    def _fit_hmm(self, features: np.ndarray) -> None:
        self.hmm_model = GaussianHMM(
            n_components=self.config.hmm_n_states,
            covariance_type="full",
            n_iter=200,
            random_state=42,
        )
        self.hmm_model.fit(features)

        states = self.hmm_model.predict(features)
        self.regime_mapping = self._map_states_to_regimes(features, states)
        self.is_fitted = True
        logger.info(f"HMM regime detector fitted with {self.config.hmm_n_states} states")

    def _fit_heuristic(self, returns: np.ndarray, volatility: np.ndarray) -> None:
        self._vol_median = np.nanmedian(volatility)
        self._ret_thresholds = (
            np.nanpercentile(returns, 25),
            np.nanpercentile(returns, 75),
        )
        self.is_fitted = True

    def _map_states_to_regimes(
        self, features: np.ndarray, states: np.ndarray
    ) -> Dict[int, MarketRegime]:
        mapping = {}
        for state in range(self.config.hmm_n_states):
            mask = states == state
            if mask.sum() == 0:
                mapping[state] = MarketRegime.UNKNOWN
                continue

            mean_return = features[mask, 0].mean()
            mean_vol = features[mask, 1].mean()

            overall_vol = features[:, 1].mean()
            overall_vol = max(overall_vol, 1e-10)
            vol_ratio = mean_vol / overall_vol

            if vol_ratio > 1.5:
                mapping[state] = MarketRegime.HIGH_VOLATILITY
            elif vol_ratio < 0.5:
                mapping[state] = MarketRegime.LOW_VOLATILITY
            elif mean_return > 0.001:
                mapping[state] = MarketRegime.TRENDING_UP
            elif mean_return < -0.001:
                mapping[state] = MarketRegime.TRENDING_DOWN
            else:
                mapping[state] = MarketRegime.MEAN_REVERTING

        return mapping

    def detect(self, returns: np.ndarray, volatility: np.ndarray) -> MarketRegime:
        if not self.is_fitted:
            return MarketRegime.UNKNOWN

        if HMM_AVAILABLE and self.hmm_model is not None:
            return self._detect_hmm(returns, volatility)
        return self._detect_heuristic(returns, volatility)

    def _detect_hmm(self, returns: np.ndarray, volatility: np.ndarray) -> MarketRegime:
        features = np.column_stack([returns, volatility])
        features = features[~np.isnan(features).any(axis=1)]

        if len(features) == 0:
            return MarketRegime.UNKNOWN

        states = self.hmm_model.predict(features)
        current_state = states[-1]
        return self.regime_mapping.get(current_state, MarketRegime.UNKNOWN)

    def _detect_heuristic(self, returns: np.ndarray, volatility: np.ndarray) -> MarketRegime:
        recent_vol = np.nanmean(volatility[-20:]) if len(volatility) >= 20 else np.nanmean(volatility)
        recent_ret = np.nanmean(returns[-20:]) if len(returns) >= 20 else np.nanmean(returns)

        if recent_vol > self._vol_median * 1.5:
            return MarketRegime.HIGH_VOLATILITY
        if recent_vol < self._vol_median * 0.5:
            return MarketRegime.LOW_VOLATILITY
        if recent_ret > self._ret_thresholds[1]:
            return MarketRegime.TRENDING_UP
        if recent_ret < self._ret_thresholds[0]:
            return MarketRegime.TRENDING_DOWN
        return MarketRegime.MEAN_REVERTING

    def get_regime_probabilities(self, returns: np.ndarray, volatility: np.ndarray) -> Dict[MarketRegime, float]:
        """Get probability distribution over regimes (HMM only)."""
        if not HMM_AVAILABLE or self.hmm_model is None:
            regime = self.detect(returns, volatility)
            return {regime: 1.0}

        features = np.column_stack([returns, volatility])
        features = features[~np.isnan(features).any(axis=1)]

        if len(features) == 0:
            return {MarketRegime.UNKNOWN: 1.0}

        posteriors = self.hmm_model.predict_proba(features)
        last_probs = posteriors[-1]

        result = {}
        for state, prob in enumerate(last_probs):
            regime = self.regime_mapping.get(state, MarketRegime.UNKNOWN)
            result[regime] = result.get(regime, 0) + prob

        return result
