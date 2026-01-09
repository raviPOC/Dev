"""
Base Model Class for Price Prediction

Abstract base class defining the interface for all prediction models.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
import joblib
from pathlib import Path


class BasePredictor(ABC):
    """
    Abstract base class for price prediction models.
    
    All prediction models should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self, name: str = "BasePredictor"):
        self.name = name
        self.model: Any = None
        self.is_fitted: bool = False
        self.feature_names: list = []
        self.scaler: Any = None
        self.metrics: Dict[str, float] = {}
        
    @abstractmethod
    def fit(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None,
        **kwargs
    ) -> "BasePredictor":
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            **kwargs: Additional model-specific parameters
            
        Returns:
            self
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predicted values
        """
        pass
    
    def evaluate(
        self,
        X_test: Union[np.ndarray, pd.DataFrame],
        y_test: Union[np.ndarray, pd.Series],
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import (
            mean_squared_error,
            mean_absolute_error,
            r2_score,
            accuracy_score,
        )
        
        y_pred = self.predict(X_test)
        
        # Regression metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Direction accuracy (classification-style metric)
        y_test_arr = np.array(y_test)
        direction_actual = (y_test_arr > 0).astype(int)
        direction_pred = (y_pred > 0).astype(int)
        direction_accuracy = accuracy_score(direction_actual, direction_pred)
        
        # Calculate profit factor (if predictions were used for trading)
        correct_direction = direction_actual == direction_pred
        profit_factor = self._calculate_profit_factor(y_test_arr, y_pred)
        
        self.metrics = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "direction_accuracy": direction_accuracy,
            "profit_factor": profit_factor,
        }
        
        return self.metrics
    
    def _calculate_profit_factor(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """Calculate profit factor from predictions."""
        # Assume we trade in the direction of prediction
        positions = np.sign(y_pred)
        returns = positions * y_true
        
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        
        if losses == 0:
            return float("inf") if gains > 0 else 0.0
        
        return gains / losses
    
    def save(self, filepath: str) -> None:
        """
        Save model to disk.
        
        Args:
            filepath: Path to save the model
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            "name": self.name,
            "model": self.model,
            "is_fitted": self.is_fitted,
            "feature_names": self.feature_names,
            "scaler": self.scaler,
            "metrics": self.metrics,
        }
        
        joblib.dump(save_dict, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> "BasePredictor":
        """
        Load model from disk.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            self
        """
        save_dict = joblib.load(filepath)
        
        self.name = save_dict["name"]
        self.model = save_dict["model"]
        self.is_fitted = save_dict["is_fitted"]
        self.feature_names = save_dict["feature_names"]
        self.scaler = save_dict["scaler"]
        self.metrics = save_dict["metrics"]
        
        print(f"Model loaded from {filepath}")
        return self
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        if hasattr(self.model, "get_params"):
            return self.model.get_params()
        return {}
    
    def __repr__(self) -> str:
        return f"{self.name}(fitted={self.is_fitted})"
