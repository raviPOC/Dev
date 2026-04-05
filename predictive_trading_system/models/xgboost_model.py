"""
XGBoost / Gradient Boosted Trees Model
=======================================
Robust tabular model for engineered features.
Works well for all horizons as a strong baseline.
Provides native feature importance.
"""

import logging
import pickle
from typing import Dict, Any, Optional, Tuple

import numpy as np

from .base import BasePredictor
from ..config.settings import ModelConfig

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("XGBoost not available. Using simple decision stump fallback.")


class XGBoostPredictor(BasePredictor):
    def __init__(self, config: ModelConfig, feature_names: Optional[list] = None):
        super().__init__(name="xgboost", horizon="all")
        self.config = config
        self.feature_names = feature_names
        self.model = None
        self._fallback_weights = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        if not XGB_AVAILABLE:
            return self._fit_fallback(X_train, y_train)

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)
        params = {
            "objective": "reg:squarederror",
            "max_depth": self.config.xgb_max_depth,
            "learning_rate": self.config.xgb_learning_rate,
            "subsample": self.config.xgb_subsample,
            "colsample_bytree": self.config.xgb_colsample_bytree,
            "min_child_weight": 5,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "tree_method": "hist",
            "verbosity": 0,
        }

        evals = [(dtrain, "train")]
        if X_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)
            evals.append((dval, "val"))

        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.config.xgb_n_estimators,
            evals=evals,
            early_stopping_rounds=self.config.early_stopping_patience,
            verbose_eval=False,
        )

        self.is_fitted = True
        result = {"best_iteration": self.model.best_iteration}
        if X_val is not None:
            val_preds = self.model.predict(dval)
            result["val_mae"] = float(np.mean(np.abs(val_preds - y_val)))
        return result

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not XGB_AVAILABLE or self.model is None:
            return self._predict_fallback(X)

        dtest = xgb.DMatrix(X, feature_names=self.feature_names)
        predictions = self.model.predict(dtest)

        # Approximate confidence via prediction margin consistency across trees
        # Use the standard deviation of tree predictions as uncertainty
        leaf_preds = []
        for i in range(0, min(self.model.best_iteration + 1, 50), 5):
            p = self.model.predict(dtest, iteration_range=(0, max(i + 1, 1)))
            leaf_preds.append(p)

        if len(leaf_preds) > 1:
            leaf_preds = np.array(leaf_preds)
            uncertainty = np.std(leaf_preds, axis=0)
            max_unc = np.max(uncertainty) if np.max(uncertainty) > 0 else 1.0
            confidence = 1.0 - (uncertainty / max_unc)
        else:
            confidence = np.full(len(predictions), 0.5)

        return predictions, confidence

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if self.model is None:
            return None
        importance = self.model.get_score(importance_type="gain")
        total = sum(importance.values()) or 1.0
        return {k: v / total for k, v in sorted(importance.items(), key=lambda x: -x[1])}

    def save(self, path: str) -> None:
        if self.model is not None:
            self.model.save_model(path)
        else:
            with open(path, "wb") as f:
                pickle.dump({"weights": self._fallback_weights, "feature_names": self.feature_names}, f)

    def load(self, path: str) -> None:
        if XGB_AVAILABLE:
            self.model = xgb.Booster()
            self.model.load_model(path)
            self.is_fitted = True
        else:
            with open(path, "rb") as f:
                data = pickle.load(f)
                self._fallback_weights = data["weights"]
                self.is_fitted = True

    def _fit_fallback(self, X_train, y_train):
        X_b = np.column_stack([X_train, np.ones(len(X_train))])
        self._fallback_weights, _, _, _ = np.linalg.lstsq(X_b, y_train, rcond=None)
        self.is_fitted = True
        return {"method": "linear_fallback"}

    def _predict_fallback(self, X):
        if self._fallback_weights is None:
            return np.zeros(len(X)), np.full(len(X), 0.5)
        X_b = np.column_stack([X, np.ones(len(X))])
        preds = X_b @ self._fallback_weights
        return preds, np.full(len(preds), 0.5)
