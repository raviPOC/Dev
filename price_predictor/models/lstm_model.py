"""
LSTM Model for Price Prediction

Long Short-Term Memory neural network for sequence-based price prediction.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler

from .base import BasePredictor


class LSTMPredictor(BasePredictor):
    """
    LSTM-based price predictor for time series forecasting.
    
    Uses PyTorch for implementation with configurable architecture.
    
    Args:
        sequence_length: Number of time steps in input sequence
        hidden_size: LSTM hidden layer size
        num_layers: Number of LSTM layers
        dropout: Dropout rate for regularization
        learning_rate: Learning rate for optimizer
        batch_size: Training batch size
        epochs: Number of training epochs
        bidirectional: Whether to use bidirectional LSTM
        use_attention: Whether to add attention mechanism
    """
    
    def __init__(
        self,
        sequence_length: int = 60,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        bidirectional: bool = False,
        use_attention: bool = True,
        early_stopping_patience: int = 10,
    ):
        super().__init__(name="LSTMPredictor")
        
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.early_stopping_patience = early_stopping_patience
        
        self.scaler = StandardScaler()
        self.input_size: Optional[int] = None
        self.device = None
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        
    def _build_model(self, input_size: int):
        """Build PyTorch LSTM model."""
        import torch
        import torch.nn as nn
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        
        class LSTMNetwork(nn.Module):
            def __init__(
                self,
                input_size: int,
                hidden_size: int,
                num_layers: int,
                dropout: float,
                bidirectional: bool,
                use_attention: bool,
            ):
                super().__init__()
                
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.bidirectional = bidirectional
                self.use_attention = use_attention
                self.num_directions = 2 if bidirectional else 1
                
                # LSTM layers
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0,
                    bidirectional=bidirectional,
                )
                
                lstm_output_size = hidden_size * self.num_directions
                
                # Attention mechanism
                if use_attention:
                    self.attention = nn.Sequential(
                        nn.Linear(lstm_output_size, hidden_size),
                        nn.Tanh(),
                        nn.Linear(hidden_size, 1),
                    )
                
                # Output layers
                self.fc = nn.Sequential(
                    nn.Linear(lstm_output_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size // 2, 1),
                )
                
            def forward(self, x):
                # LSTM forward pass
                lstm_out, _ = self.lstm(x)
                
                if self.use_attention:
                    # Attention weights
                    attention_weights = torch.softmax(
                        self.attention(lstm_out).squeeze(-1), dim=1
                    )
                    # Weighted sum
                    context = torch.bmm(
                        attention_weights.unsqueeze(1), lstm_out
                    ).squeeze(1)
                else:
                    # Use last hidden state
                    context = lstm_out[:, -1, :]
                
                # Final prediction
                output = self.fc(context)
                return output.squeeze(-1)
        
        self.model = LSTMNetwork(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            use_attention=self.use_attention,
        ).to(self.device)
        
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True,
        **kwargs
    ) -> "LSTMPredictor":
        """
        Train the LSTM model.
        
        Args:
            X_train: Training sequences of shape (n_samples, sequence_length, n_features)
            y_train: Training targets of shape (n_samples,)
            X_val: Validation sequences
            y_val: Validation targets
            verbose: Whether to print training progress
            
        Returns:
            self
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        
        # Reshape if needed (for 2D input)
        if len(X_train.shape) == 2:
            X_train = self._create_sequences_from_2d(X_train)
            y_train = y_train[self.sequence_length:]
            if X_val is not None:
                X_val = self._create_sequences_from_2d(X_val)
                y_val = y_val[self.sequence_length:]
        
        # Scale features
        n_samples, seq_len, n_features = X_train.shape
        X_train_flat = X_train.reshape(-1, n_features)
        self.scaler.fit(X_train_flat)
        X_train_scaled = self.scaler.transform(X_train_flat).reshape(n_samples, seq_len, n_features)
        
        if X_val is not None:
            X_val_flat = X_val.reshape(-1, n_features)
            X_val_scaled = self.scaler.transform(X_val_flat).reshape(X_val.shape)
        
        # Build model
        self._build_model(n_features)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        
        if X_val is not None:
            X_val_tensor = torch.FloatTensor(X_val_scaled).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=verbose
        )
        
        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                
                self.val_losses.append(val_loss)
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    self.best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.epochs} - "
                          f"Train Loss: {train_loss:.6f} - "
                          f"Val Loss: {val_loss:.6f}")
                
                if patience_counter >= self.early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.6f}")
        
        # Restore best model
        if hasattr(self, "best_model_state"):
            self.model.load_state_dict(self.best_model_state)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input sequences of shape (n_samples, sequence_length, n_features)
               or (n_samples, n_features) for 2D input
               
        Returns:
            Predicted values
        """
        import torch
        
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Reshape if needed
        if len(X.shape) == 2:
            X = self._create_sequences_from_2d(X)
        
        # Scale features
        n_samples, seq_len, n_features = X.shape
        X_flat = X.reshape(-1, n_features)
        X_scaled = self.scaler.transform(X_flat).reshape(n_samples, seq_len, n_features)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()
        
        return predictions
    
    def _create_sequences_from_2d(self, X: np.ndarray) -> np.ndarray:
        """Create sequences from 2D array."""
        sequences = []
        for i in range(self.sequence_length, len(X)):
            sequences.append(X[i-self.sequence_length:i])
        return np.array(sequences)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability of price going up.
        
        Uses sigmoid on predictions as a proxy for probability.
        """
        predictions = self.predict(X)
        # Scale predictions to [0, 1] range
        probs = 1 / (1 + np.exp(-predictions * 100))  # Sigmoid with scaling
        return np.column_stack([1 - probs, probs])
    
    def get_attention_weights(self, X: np.ndarray) -> np.ndarray:
        """
        Get attention weights for interpretability.
        
        Only available if use_attention=True.
        """
        import torch
        
        if not self.use_attention:
            raise ValueError("Attention not enabled for this model")
        
        if len(X.shape) == 2:
            X = self._create_sequences_from_2d(X)
        
        n_samples, seq_len, n_features = X.shape
        X_flat = X.reshape(-1, n_features)
        X_scaled = self.scaler.transform(X_flat).reshape(n_samples, seq_len, n_features)
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            lstm_out, _ = self.model.lstm(X_tensor)
            attention_weights = torch.softmax(
                self.model.attention(lstm_out).squeeze(-1), dim=1
            ).cpu().numpy()
        
        return attention_weights
