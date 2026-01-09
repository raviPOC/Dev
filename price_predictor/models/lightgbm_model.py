"""
LightGBM Model for Price Prediction

Fast gradient boosting model optimized for large-scale price prediction.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from sklearn.preprocessing import StandardScaler

from .base import BasePredictor


class LightGBMPredictor(BasePredictor):
    """
    LightGBM-based price predictor.
    
    Fast and memory-efficient gradient boosting optimized for
    high-frequency trading scenarios.
    
    Args:
        n_estimators: Number of boosting iterations
        max_depth: Maximum tree depth (-1 for no limit)
        learning_rate: Boosting learning rate
        num_leaves: Maximum number of leaves per tree
        subsample: Subsample ratio of training data
        colsample_bytree: Subsample ratio of columns
        reg_alpha: L1 regularization
        reg_lambda: L2 regularization
        min_child_samples: Minimum samples in leaf
        early_stopping_rounds: Early stopping patience
    """
    
    def __init__(
        self,
        n_estimators: int = 1000,
        max_depth: int = -1,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        min_child_samples: int = 20,
        early_stopping_rounds: int = 50,
        n_jobs: int = -1,
        random_state: int = 42,
        verbose: int = -1,
    ):
        super().__init__(name="LightGBMPredictor")
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_child_samples = min_child_samples
        self.early_stopping_rounds = early_stopping_rounds
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        
        self.scaler = StandardScaler()
        self.feature_importance_: Optional[pd.DataFrame] = None
        
    def fit(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None,
        verbose: bool = True,
        **kwargs
    ) -> "LightGBMPredictor":
        """
        Train the LightGBM model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (for early stopping)
            y_val: Validation targets
            verbose: Whether to print training progress
            
        Returns:
            self
        """
        import lightgbm as lgb
        
        # Store feature names if DataFrame
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            X_train = X_train.values
        
        if isinstance(X_val, pd.DataFrame):
            X_val = X_val.values
            
        # Handle NaN values
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        if X_val is not None:
            X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
        
        # Create callbacks
        callbacks = []
        if verbose:
            callbacks.append(lgb.log_evaluation(period=100))
        if X_val is not None:
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds))
        
        # Create model
        self.model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            min_child_samples=self.min_child_samples,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        
        # Fit model
        eval_set = [(X_train_scaled, y_train)]
        if X_val is not None:
            eval_set.append((X_val_scaled, y_val))
        
        self.model.fit(
            X_train_scaled,
            y_train,
            eval_set=eval_set,
            callbacks=callbacks if callbacks else None,
        )
        
        # Get best iteration
        if hasattr(self.model, "best_iteration_"):
            self.best_iteration = self.model.best_iteration_
        
        # Calculate feature importance
        self._calculate_feature_importance()
        
        self.is_fitted = True
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Get probability estimates for direction.
        
        Uses sigmoid on predictions as proxy for probability.
        """
        predictions = self.predict(X)
        probs = 1 / (1 + np.exp(-predictions * 100))
        return np.column_stack([1 - probs, probs])
    
    def _calculate_feature_importance(self) -> None:
        """Calculate and store feature importance."""
        importance = self.model.feature_importances_
        
        if self.feature_names:
            self.feature_importance_ = pd.DataFrame({
                "feature": self.feature_names,
                "importance": importance,
            }).sort_values("importance", ascending=False)
        else:
            self.feature_importance_ = pd.DataFrame({
                "feature": [f"feature_{i}" for i in range(len(importance))],
                "importance": importance,
            }).sort_values("importance", ascending=False)
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get top feature importances.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature names and importance scores
        """
        if self.feature_importance_ is None:
            raise ValueError("Model not fitted yet.")
        
        return self.feature_importance_.head(top_n)
    
    def plot_feature_importance(self, top_n: int = 20, figsize: tuple = (10, 8)):
        """Plot feature importance."""
        import matplotlib.pyplot as plt
        
        importance_df = self.get_feature_importance(top_n)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(
            importance_df["feature"][::-1],
            importance_df["importance"][::-1],
        )
        ax.set_xlabel("Importance")
        ax.set_title(f"Top {top_n} Feature Importances - {self.name}")
        plt.tight_layout()
        
        return fig
    
    def get_leaf_indices(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Get leaf indices for each sample.
        
        Useful for feature engineering and model stacking.
        
        Args:
            X: Input features
            
        Returns:
            Array of leaf indices
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        X = np.nan_to_num(X, nan=0.0)
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict(X_scaled, pred_leaf=True)
