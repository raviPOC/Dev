"""
LSTM + Attention Model
======================
Sequence-to-one LSTM with attention mechanism for medium-term prediction.
Captures temporal dependencies in price/feature sequences.
"""

import logging
import pickle
from typing import Dict, Any, Optional, Tuple

import numpy as np

from .base import BasePredictor
from ..config.settings import ModelConfig

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. LSTM model will use fallback numpy implementation.")


if TORCH_AVAILABLE:
    class AttentionLayer(nn.Module):
        def __init__(self, hidden_size: int):
            super().__init__()
            self.attention = nn.Linear(hidden_size, 1)

        def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
            weights = torch.softmax(self.attention(lstm_output), dim=1)
            context = torch.sum(weights * lstm_output, dim=1)
            return context

    class LSTMNetwork(nn.Module):
        def __init__(self, input_size: int, config: ModelConfig):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=config.lstm_hidden_size,
                num_layers=config.lstm_num_layers,
                dropout=config.lstm_dropout if config.lstm_num_layers > 1 else 0,
                batch_first=True,
            )
            self.attention = AttentionLayer(config.lstm_hidden_size)
            self.dropout = nn.Dropout(config.lstm_dropout)
            self.fc_prediction = nn.Linear(config.lstm_hidden_size, 1)
            self.fc_confidence = nn.Linear(config.lstm_hidden_size, 1)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            lstm_out, _ = self.lstm(x)
            context = self.attention(lstm_out)
            context = self.dropout(context)
            prediction = self.fc_prediction(context)
            confidence = torch.sigmoid(self.fc_confidence(context))
            return prediction.squeeze(-1), confidence.squeeze(-1)


class LSTMPredictor(BasePredictor):
    def __init__(self, input_size: int, config: ModelConfig, device: str = "cpu"):
        super().__init__(name="lstm_attention", horizon="medium")
        self.config = config
        self.device = device
        self.input_size = input_size

        if TORCH_AVAILABLE:
            self.model = LSTMNetwork(input_size, config).to(device)
        else:
            self.model = None
            self._weights = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        if not TORCH_AVAILABLE:
            return self._fit_fallback(X_train, y_train, X_val, y_val)

        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train).to(self.device),
            torch.FloatTensor(y_train).to(self.device),
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.config.max_epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                predictions, _ = self.model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)
            history["train_loss"].append(avg_train_loss)

            if X_val is not None:
                val_loss = self._validate(X_val, y_val, criterion)
                history["val_loss"].append(val_loss)
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1

                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    self.model.load_state_dict(best_state)
                    break

        self.is_fitted = True
        return {
            "final_train_loss": history["train_loss"][-1],
            "best_val_loss": best_val_loss if X_val is not None else None,
            "epochs_trained": len(history["train_loss"]),
        }

    def _validate(self, X_val, y_val, criterion):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X_val).to(self.device)
            y_t = torch.FloatTensor(y_val).to(self.device)
            preds, _ = self.model(X_t)
            return criterion(preds, y_t).item()

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not TORCH_AVAILABLE:
            return self._predict_fallback(X)

        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            predictions, confidence = self.model(X_t)
            return predictions.cpu().numpy(), confidence.cpu().numpy()

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        return None

    def save(self, path: str) -> None:
        if TORCH_AVAILABLE and self.model is not None:
            torch.save({
                "model_state": self.model.state_dict(),
                "config": self.config,
                "input_size": self.input_size,
            }, path)
        else:
            with open(path, "wb") as f:
                pickle.dump({"weights": self._weights}, f)

    def load(self, path: str) -> None:
        if TORCH_AVAILABLE:
            checkpoint = torch.load(path, map_location=self.device)
            self.model = LSTMNetwork(checkpoint["input_size"], checkpoint["config"]).to(self.device)
            self.model.load_state_dict(checkpoint["model_state"])
            self.is_fitted = True
        else:
            with open(path, "rb") as f:
                data = pickle.load(f)
                self._weights = data["weights"]
                self.is_fitted = True

    def _fit_fallback(self, X_train, y_train, X_val, y_val):
        """Simple linear fallback when PyTorch is not available."""
        if X_train.ndim == 3:
            X_flat = X_train.reshape(X_train.shape[0], -1)
        else:
            X_flat = X_train

        X_b = np.column_stack([X_flat, np.ones(len(X_flat))])
        self._weights, _, _, _ = np.linalg.lstsq(X_b, y_train, rcond=None)
        self.is_fitted = True

        return {"final_train_loss": float(np.mean((X_b @ self._weights - y_train) ** 2))}

    def _predict_fallback(self, X):
        if X.ndim == 3:
            X_flat = X.reshape(X.shape[0], -1)
        else:
            X_flat = X
        X_b = np.column_stack([X_flat, np.ones(len(X_flat))])
        preds = X_b @ self._weights
        confidence = np.full(len(preds), 0.5)
        return preds, confidence
