"""
Ensemble Model for Price Prediction

Combines multiple models for robust price prediction.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .base import BasePredictor
from .lstm_model import LSTMPredictor
from .xgboost_model import XGBoostPredictor
from .lightgbm_model import LightGBMPredictor


class EnsemblePredictor(BasePredictor):
    """
    Ensemble predictor combining multiple models.
    
    Supports various ensemble strategies:
    - Simple averaging
    - Weighted averaging
    - Stacking with meta-learner
    
    Args:
        models: List of model instances or model configs
        ensemble_method: Method for combining predictions
            - "average": Simple average
            - "weighted": Weighted average based on validation performance
            - "stacking": Meta-learner stacking
        weights: Custom weights for weighted average (optional)
    """
    
    def __init__(
        self,
        models: Optional[List[BasePredictor]] = None,
        ensemble_method: str = "stacking",
        weights: Optional[List[float]] = None,
    ):
        super().__init__(name="EnsemblePredictor")
        
        self.ensemble_method = ensemble_method
        self.weights = weights
        
        # Initialize default models if not provided
        if models is None:
            self.models = self._create_default_models()
        else:
            self.models = models
        
        self.meta_learner = None
        self.model_weights: Optional[np.ndarray] = None
        self.model_metrics: Dict[str, Dict[str, float]] = {}
        
    def _create_default_models(self) -> List[BasePredictor]:
        """Create default set of models for ensemble."""
        return [
            XGBoostPredictor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
            ),
            LightGBMPredictor(
                n_estimators=500,
                num_leaves=31,
                learning_rate=0.05,
            ),
            XGBoostPredictor(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.1,
            ),
            LightGBMPredictor(
                n_estimators=300,
                num_leaves=63,
                learning_rate=0.1,
            ),
        ]
    
    def fit(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None,
        verbose: bool = True,
        **kwargs
    ) -> "EnsemblePredictor":
        """
        Train all ensemble models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            verbose: Whether to print training progress
            
        Returns:
            self
        """
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
        
        # Train each model
        val_predictions = []
        
        for i, model in enumerate(self.models):
            if verbose:
                print(f"\n{'='*50}")
                print(f"Training model {i+1}/{len(self.models)}: {model.name}")
                print(f"{'='*50}")
            
            # Handle LSTM differently (needs sequences)
            if isinstance(model, LSTMPredictor):
                # LSTM requires sequence data - skip for now in default ensemble
                # User can add pre-configured LSTM to ensemble
                if verbose:
                    print(f"Skipping LSTM in ensemble (requires sequence data)")
                continue
            
            model.fit(X_train, y_train, X_val, y_val, verbose=verbose)
            
            # Collect validation predictions for stacking/weighting
            if X_val is not None:
                val_pred = model.predict(X_val)
                val_predictions.append(val_pred)
                
                # Evaluate model
                metrics = model.evaluate(X_val, y_val)
                self.model_metrics[f"{model.name}_{i}"] = metrics
                
                if verbose:
                    print(f"\nValidation Metrics for {model.name}:")
                    print(f"  RMSE: {metrics['rmse']:.6f}")
                    print(f"  Direction Accuracy: {metrics['direction_accuracy']:.4f}")
        
        # Remove unfitted models (like LSTM that was skipped)
        self.models = [m for m in self.models if m.is_fitted]
        
        # Set up ensemble combination method
        if X_val is not None and len(val_predictions) > 0:
            val_predictions = np.column_stack(val_predictions)
            
            if self.ensemble_method == "weighted":
                self._calculate_weights(val_predictions, np.array(y_val))
            elif self.ensemble_method == "stacking":
                self._fit_meta_learner(val_predictions, np.array(y_val))
        
        self.is_fitted = True
        
        if verbose:
            print(f"\n{'='*50}")
            print("Ensemble training complete!")
            print(f"Number of models: {len(self.models)}")
            print(f"Ensemble method: {self.ensemble_method}")
            print(f"{'='*50}")
        
        return self
    
    def _calculate_weights(
        self,
        val_predictions: np.ndarray,
        y_val: np.ndarray
    ) -> None:
        """Calculate model weights based on validation performance."""
        # Calculate MSE for each model
        mse_scores = []
        for i in range(val_predictions.shape[1]):
            mse = np.mean((val_predictions[:, i] - y_val) ** 2)
            mse_scores.append(mse)
        
        # Convert to weights (inverse MSE)
        mse_scores = np.array(mse_scores)
        weights = 1 / (mse_scores + 1e-10)  # Add small constant to avoid division by zero
        self.model_weights = weights / weights.sum()  # Normalize
        
        print(f"Calculated model weights: {self.model_weights}")
    
    def _fit_meta_learner(
        self,
        val_predictions: np.ndarray,
        y_val: np.ndarray
    ) -> None:
        """Fit meta-learner for stacking."""
        self.meta_scaler = StandardScaler()
        val_predictions_scaled = self.meta_scaler.fit_transform(val_predictions)
        
        # Use Ridge regression as meta-learner
        self.meta_learner = Ridge(alpha=1.0)
        self.meta_learner.fit(val_predictions_scaled, y_val)
        
        print(f"Meta-learner coefficients: {self.meta_learner.coef_}")
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            X: Features to predict on
            
        Returns:
            Ensemble predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get predictions from each model
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.column_stack(predictions)
        
        # Combine predictions
        if self.ensemble_method == "average":
            return np.mean(predictions, axis=1)
        
        elif self.ensemble_method == "weighted":
            if self.model_weights is None:
                return np.mean(predictions, axis=1)
            return np.average(predictions, axis=1, weights=self.model_weights)
        
        elif self.ensemble_method == "stacking":
            if self.meta_learner is None:
                return np.mean(predictions, axis=1)
            predictions_scaled = self.meta_scaler.transform(predictions)
            return self.meta_learner.predict(predictions_scaled)
        
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
    
    def predict_individual(
        self, X: Union[np.ndarray, pd.DataFrame]
    ) -> Dict[str, np.ndarray]:
        """
        Get predictions from each individual model.
        
        Args:
            X: Features to predict on
            
        Returns:
            Dictionary of model name -> predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        results = {}
        for i, model in enumerate(self.models):
            key = f"{model.name}_{i}"
            results[key] = model.predict(X)
        
        return results
    
    def get_model_weights(self) -> Optional[Dict[str, float]]:
        """Get weights assigned to each model."""
        if self.model_weights is None:
            return None
        
        weights_dict = {}
        for i, model in enumerate(self.models):
            key = f"{model.name}_{i}"
            weights_dict[key] = self.model_weights[i]
        
        return weights_dict
    
    def get_model_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get validation metrics for each model."""
        return self.model_metrics
    
    def add_model(self, model: BasePredictor) -> None:
        """Add a model to the ensemble."""
        self.models.append(model)
        self.is_fitted = False  # Need to refit
        
    def remove_model(self, index: int) -> None:
        """Remove a model from the ensemble by index."""
        if 0 <= index < len(self.models):
            self.models.pop(index)
            self.is_fitted = False  # Need to refit


class StackingEnsemble(EnsemblePredictor):
    """
    Advanced stacking ensemble with cross-validation.
    
    Uses out-of-fold predictions for training the meta-learner
    to avoid overfitting.
    """
    
    def __init__(
        self,
        models: Optional[List[BasePredictor]] = None,
        n_folds: int = 5,
    ):
        super().__init__(models=models, ensemble_method="stacking")
        self.n_folds = n_folds
        self.name = "StackingEnsemble"
        
    def fit(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None,
        verbose: bool = True,
        **kwargs
    ) -> "StackingEnsemble":
        """
        Train stacking ensemble with cross-validation.
        
        Uses time-series aware cross-validation to generate
        out-of-fold predictions for meta-learner training.
        """
        from sklearn.model_selection import TimeSeriesSplit
        
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            X_train_arr = X_train.values
        else:
            X_train_arr = X_train
            
        y_train_arr = np.array(y_train)
        
        n_samples = len(X_train_arr)
        oof_predictions = np.zeros((n_samples, len(self.models)))
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=self.n_folds)
        
        for i, model in enumerate(self.models):
            if isinstance(model, LSTMPredictor):
                if verbose:
                    print(f"Skipping LSTM in stacking (requires sequence data)")
                continue
                
            if verbose:
                print(f"\nTraining {model.name} with {self.n_folds}-fold CV...")
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_arr)):
                X_fold_train = X_train_arr[train_idx]
                y_fold_train = y_train_arr[train_idx]
                X_fold_val = X_train_arr[val_idx]
                
                # Clone and train model
                model_clone = self._clone_model(model)
                model_clone.fit(X_fold_train, y_fold_train, verbose=False)
                
                # Out-of-fold predictions
                oof_predictions[val_idx, i] = model_clone.predict(X_fold_val)
            
            # Final model training on full data
            model.fit(X_train_arr, y_train_arr, X_val, y_val, verbose=verbose)
        
        # Remove unfitted models
        fitted_mask = [m.is_fitted for m in self.models]
        self.models = [m for m, fitted in zip(self.models, fitted_mask) if fitted]
        oof_predictions = oof_predictions[:, fitted_mask]
        
        # Remove samples with zero predictions (from early folds)
        valid_mask = np.any(oof_predictions != 0, axis=1)
        oof_valid = oof_predictions[valid_mask]
        y_valid = y_train_arr[valid_mask]
        
        # Fit meta-learner
        self._fit_meta_learner(oof_valid, y_valid)
        
        self.is_fitted = True
        
        if verbose:
            print(f"\nStacking ensemble training complete!")
            print(f"Meta-learner coefficients: {self.meta_learner.coef_}")
        
        return self
    
    def _clone_model(self, model: BasePredictor) -> BasePredictor:
        """Create a fresh copy of a model with same hyperparameters."""
        if isinstance(model, XGBoostPredictor):
            return XGBoostPredictor(
                n_estimators=model.n_estimators,
                max_depth=model.max_depth,
                learning_rate=model.learning_rate,
                subsample=model.subsample,
                colsample_bytree=model.colsample_bytree,
                reg_alpha=model.reg_alpha,
                reg_lambda=model.reg_lambda,
            )
        elif isinstance(model, LightGBMPredictor):
            return LightGBMPredictor(
                n_estimators=model.n_estimators,
                max_depth=model.max_depth,
                learning_rate=model.learning_rate,
                num_leaves=model.num_leaves,
                subsample=model.subsample,
                colsample_bytree=model.colsample_bytree,
                reg_alpha=model.reg_alpha,
                reg_lambda=model.reg_lambda,
            )
        else:
            # For other models, just return a new instance
            return type(model)()
