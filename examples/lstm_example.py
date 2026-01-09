#!/usr/bin/env python3
"""
LSTM Model Example for Short-Term Price Prediction

This example demonstrates how to use the LSTM model specifically,
which requires sequence data preparation.
"""

import sys
sys.path.insert(0, "..")

import numpy as np
import pandas as pd

from price_predictor import (
    DataLoader,
    FeatureEngineer,
    LSTMPredictor,
)


def main():
    print("=" * 60)
    print("LSTM Price Prediction Example")
    print("=" * 60)
    
    # Load data
    print("\n[1] Loading Data...")
    loader = DataLoader()
    df = loader.generate_synthetic_data(
        n_samples=3000,
        start_price=100.0,
        volatility=0.02,
        seed=42,
    )
    print(f"   Generated {len(df)} samples")
    
    # Feature engineering
    print("\n[2] Creating Features...")
    feature_engineer = FeatureEngineer()
    df_features = feature_engineer.create_all_features(
        df,
        include_target=True,
        target_horizon=1,
    )
    
    # Clean data
    df_clean = df_features.dropna()
    print(f"   Features: {len(feature_engineer.feature_names)}")
    
    # Prepare sequences for LSTM
    print("\n[3] Preparing Sequences...")
    sequence_length = 60  # Use last 60 time steps
    
    X_sequences, y_targets = feature_engineer.prepare_sequences(
        df_clean,
        sequence_length=sequence_length,
        target_col="target",
    )
    
    print(f"   Sequence shape: {X_sequences.shape}")
    print(f"   (samples, timesteps, features)")
    
    # Split data
    n_samples = len(X_sequences)
    train_end = int(n_samples * 0.7)
    val_end = int(n_samples * 0.85)
    
    X_train = X_sequences[:train_end]
    y_train = y_targets[:train_end]
    
    X_val = X_sequences[train_end:val_end]
    y_val = y_targets[train_end:val_end]
    
    X_test = X_sequences[val_end:]
    y_test = y_targets[val_end:]
    
    print(f"\n   Train: {len(X_train)} sequences")
    print(f"   Val: {len(X_val)} sequences")
    print(f"   Test: {len(X_test)} sequences")
    
    # Train LSTM
    print("\n[4] Training LSTM Model...")
    lstm = LSTMPredictor(
        sequence_length=sequence_length,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        learning_rate=0.001,
        batch_size=32,
        epochs=50,
        bidirectional=False,
        use_attention=True,
        early_stopping_patience=10,
    )
    
    lstm.fit(X_train, y_train, X_val, y_val, verbose=True)
    
    # Evaluate
    print("\n[5] Evaluation Results:")
    print("-" * 40)
    metrics = lstm.evaluate(X_test, y_test)
    for metric, value in metrics.items():
        print(f"   {metric}: {value:.6f}")
    
    # Make predictions
    print("\n[6] Sample Predictions:")
    predictions = lstm.predict(X_test[:10])
    
    for i in range(10):
        actual = y_test[i]
        pred = predictions[i]
        direction_match = "✓" if (actual > 0) == (pred > 0) else "✗"
        print(f"   Sample {i+1}: Actual={actual:+.6f}, Pred={pred:+.6f} {direction_match}")
    
    # Attention weights
    if lstm.use_attention:
        print("\n[7] Attention Weights (sample):")
        attention = lstm.get_attention_weights(X_test[:1])
        print(f"   Shape: {attention.shape}")
        print(f"   Last 10 timesteps: {attention[0, -10:]}")
    
    print("\n" + "=" * 60)
    print("LSTM training complete!")
    print("=" * 60)
    
    return lstm, metrics


if __name__ == "__main__":
    model, metrics = main()
