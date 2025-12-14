from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class SeqDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


@dataclass
class PredictorConfig:
    window_size: int = 30
    batch_size: int = 128
    epochs: int = 10
    lr: float = 1e-3


def build_supervised_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int,
    target_col: str = "close",
) -> tuple[np.ndarray, np.ndarray]:
    """Build sequences X and next-step target y.

    Here y is the next-day log return of target_col.
    """
    x = df[feature_cols].to_numpy(dtype=np.float32)
    price = df[target_col].astype(float)
    y = np.log(price.shift(-1) / price).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)

    xs = []
    ys = []
    for t in range(window_size, len(df) - 1):
        xs.append(x[t - window_size : t])
        ys.append(y[t])

    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def train_lstm_predictor(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    cfg: PredictorConfig = PredictorConfig(),
    device: str | None = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    x_train, y_train = build_supervised_sequences(train_df, feature_cols, cfg.window_size)
    x_test, y_test = build_supervised_sequences(test_df, feature_cols, cfg.window_size)

    model = LSTMRegressor(n_features=len(feature_cols)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(SeqDataset(x_train, y_train), batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(SeqDataset(x_test, y_test), batch_size=cfg.batch_size, shuffle=False)

    for _epoch in range(cfg.epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            losses = []
            for xb, yb in test_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                losses.append(loss_fn(pred, yb).item())
        # return last epoch test loss in results; keep training silent for scripts

    return model


@torch.no_grad()
def predict_next_log_return(
    model: nn.Module,
    recent_window: np.ndarray,
    device: str | None = None,
) -> float:
    """recent_window: (T, F) float32."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    x = torch.tensor(recent_window[None, ...], dtype=torch.float32).to(device)
    y = model(x).detach().cpu().numpy().reshape(-1)[0]
    return float(y)
