"""
Temporal Convolutional Network (TCN)
====================================
Dilated causal convolutions for short-term prediction.
Fast inference, good at capturing local temporal patterns.
"""

import logging
import pickle
from typing import Dict, Any, Optional, Tuple, List

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


if TORCH_AVAILABLE:
    class CausalConv1d(nn.Module):
        """Causal (no future leakage) 1D convolution with dilation."""

        def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
            super().__init__()
            self.padding = (kernel_size - 1) * dilation
            self.conv = nn.Conv1d(
                in_channels, out_channels, kernel_size,
                padding=self.padding, dilation=dilation
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.conv(x)
            if self.padding > 0:
                out = out[:, :, :-self.padding]
            return out

    class TemporalBlock(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
            super().__init__()
            self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
            self.bn1 = nn.BatchNorm1d(out_ch)
            self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
            self.bn2 = nn.BatchNorm1d(out_ch)
            self.dropout = nn.Dropout(dropout)
            self.relu = nn.ReLU()
            self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = self.downsample(x)
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.dropout(out)
            out = self.relu(self.bn2(self.conv2(out)))
            out = self.dropout(out)
            return self.relu(out + residual)

    class TCNNetwork(nn.Module):
        def __init__(self, input_size: int, config: ModelConfig):
            super().__init__()
            channels = config.tcn_num_channels
            layers = []
            for i, out_ch in enumerate(channels):
                in_ch = input_size if i == 0 else channels[i - 1]
                dilation = 2 ** i
                layers.append(TemporalBlock(
                    in_ch, out_ch, config.tcn_kernel_size, dilation, config.tcn_dropout
                ))
            self.network = nn.Sequential(*layers)
            self.fc_prediction = nn.Linear(channels[-1], 1)
            self.fc_confidence = nn.Linear(channels[-1], 1)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            # x: (batch, seq_len, features) -> (batch, features, seq_len)
            x = x.transpose(1, 2)
            out = self.network(x)
            last = out[:, :, -1]  # take the last time step
            prediction = self.fc_prediction(last).squeeze(-1)
            confidence = torch.sigmoid(self.fc_confidence(last)).squeeze(-1)
            return prediction, confidence


class TCNPredictor(BasePredictor):
    def __init__(self, input_size: int, config: ModelConfig, device: str = "cpu"):
        super().__init__(name="tcn", horizon="short")
        self.config = config
        self.device = device
        self.input_size = input_size

        if TORCH_AVAILABLE:
            self.model = TCNNetwork(input_size, config).to(device)
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
            return self._fit_fallback(X_train, y_train)

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
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config.max_epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                preds, _ = self.model(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    X_v = torch.FloatTensor(X_val).to(self.device)
                    y_v = torch.FloatTensor(y_val).to(self.device)
                    val_preds, _ = self.model(X_v)
                    val_loss = criterion(val_preds, y_v).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.config.early_stopping_patience:
                    break

        self.is_fitted = True
        return {
            "final_train_loss": epoch_loss / max(n_batches, 1),
            "best_val_loss": best_val_loss if X_val is not None else None,
            "epochs_trained": epoch + 1,
        }

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
            torch.save({"model_state": self.model.state_dict(), "config": self.config, "input_size": self.input_size}, path)

    def load(self, path: str) -> None:
        if TORCH_AVAILABLE:
            ckpt = torch.load(path, map_location=self.device)
            self.model = TCNNetwork(ckpt["input_size"], ckpt["config"]).to(self.device)
            self.model.load_state_dict(ckpt["model_state"])
            self.is_fitted = True

    def _fit_fallback(self, X_train, y_train):
        X_flat = X_train.reshape(X_train.shape[0], -1) if X_train.ndim == 3 else X_train
        X_b = np.column_stack([X_flat, np.ones(len(X_flat))])
        self._weights, _, _, _ = np.linalg.lstsq(X_b, y_train, rcond=None)
        self.is_fitted = True
        return {"final_train_loss": float(np.mean((X_b @ self._weights - y_train) ** 2))}

    def _predict_fallback(self, X):
        X_flat = X.reshape(X.shape[0], -1) if X.ndim == 3 else X
        X_b = np.column_stack([X_flat, np.ones(len(X_flat))])
        preds = X_b @ self._weights
        return preds, np.full(len(preds), 0.5)
