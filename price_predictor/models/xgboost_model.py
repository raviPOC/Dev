"""
XGBoost Model for Price Prediction

Gradient boosting model optimized for tabular price prediction.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from sklearn.preprocessing import StandardScaler

from .base import BasePredictor


class XGBoostPredictor(BasePredictor):
    """
    XGBoost-based price predictor.
    
    Optimized for tabular data with engineered features.
    Includes built-in feature importance analysis.
    
    Args:
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Boosting learning rate
        subsample: Subsample ratio of training data
        colsample_bytree: Subsample ratio of columns
        reg_alpha: L1 regularization
        reg_lambda: L2 regularization
        early_stopping_rounds: Early stopping patience
        objective: Learning objective (reg:squarederror, reg:absoluteerror)
    """
    
    def __init__(
        self,
        n_estimators: int = 1000,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        early_stopping_rounds: int = 50,
        objective: str = "reg:squarederror",
        n_jobs: int = -1,
        random_state: int = 42,
    ):
        super().__init__(name="XGBoostPredictor")
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.early_stopping_rounds = early_stopping_rounds
        self.objective = objective
        self.n_jobs = n_jobs
        self.random_state = random_state
        
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
    ) -> "XGBoostPredictor":
        """
        Train the XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (for early stopping)
            y_val: Validation targets
            verbose: Whether to print training progress
            
        Returns:
            self
        """
        import xgboost as xgb
        
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
        
        # Create model
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            objective=self.objective,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            tree_method="hist",  # Fast histogram-based method
        )
        
        # Fit model
        eval_set = [(X_train_scaled, y_train)]
        if X_val is not None:
            eval_set.append((X_val_scaled, y_val))
        
        self.model.fit(
            X_train_scaled,
            y_train,
            eval_set=eval_set,
            verbose=verbose if verbose else 0,
        )
        
        # Get best iteration if early stopping was used
        if hasattr(self.model, "best_iteration"):
            self.best_iteration = self.model.best_iteration
        
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
    
    def optimize_hyperparameters(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_val: Union[np.ndarray, pd.DataFrame],
        y_val: Union[np.ndarray, pd.Series],
        n_trials: int = 50,
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            n_trials: Number of optimization trials
            
        Returns:
            Best hyperparameters
        """
        try:
            import optuna
        except ImportError:
            raise ImportError("optuna not installed. Run: pip install optuna")
        
        import xgboost as xgb
        
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(X_val, pd.DataFrame):
            X_val = X_val.values
            
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
            }
            
            model = xgb.XGBRegressor(**params, tree_method="hist", n_jobs=-1)
            model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                verbose=False,
            )
            
            predictions = model.predict(X_val_scaled)
            mse = np.mean((predictions - np.array(y_val)) ** 2)
            return mse
        
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        return study.best_params
