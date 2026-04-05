#!/usr/bin/env python3
"""
Demo: Predictive Trading System
================================
Generates synthetic data, trains the ensemble, runs predictions,
and demonstrates the full pipeline including the feedback loop.

Usage:
    python run_demo.py
"""

import logging
import os
import sys
import tempfile

import numpy as np
import pandas as pd

from predictive_trading_system.config.settings import get_default_config
from predictive_trading_system.utils.data_generator import (
    generate_synthetic_ohlcv,
    generate_synthetic_trades,
)
from predictive_trading_system.data.ingestion import MarketDataLoader, DataValidator
from predictive_trading_system.features.engineering import FeatureEngine
from predictive_trading_system.models.xgboost_model import XGBoostPredictor
from predictive_trading_system.models.lstm_model import LSTMPredictor
from predictive_trading_system.models.tcn_model import TCNPredictor
from predictive_trading_system.models.regime_detector import RegimeDetector
from predictive_trading_system.meta.meta_learner import MetaLearner
from predictive_trading_system.core.feedback import FeedbackLoop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("demo")


def main():
    print("=" * 70)
    print("  PREDICTIVE TRADING SYSTEM — DEMO")
    print("=" * 70)

    config = get_default_config()
    config.model.max_epochs = 10  # fast training for demo
    config.model.lstm_sequence_length = 30
    config.model.xgb_n_estimators = 50

    # --- Step 1: Generate synthetic data ---
    print("\n[1/7] Generating synthetic market data...")
    market_data = generate_synthetic_ohlcv(n_bars=3000, seed=42)
    print(f"  Generated {len(market_data)} bars of OHLCV data")
    print(f"  Date range: {market_data.index[0]} to {market_data.index[-1]}")
    print(f"  Price range: ${market_data['close'].min():.2f} - ${market_data['close'].max():.2f}")

    print("\n  Generating synthetic trading logs...")
    trade_logs = generate_synthetic_trades(n_trades=200, seed=42)
    print(f"  Generated {len(trade_logs)} trades")
    print(f"  Win rate: {trade_logs['pnl'].gt(0).mean():.1%}")
    print(f"  Total PnL: ${trade_logs['pnl'].sum():.2f}")

    # --- Step 2: Validate data ---
    print("\n[2/7] Validating data...")
    validation = DataValidator.validate_ohlcv(market_data)
    print(f"  Valid: {validation['valid']}")
    if validation['issues']:
        for issue in validation['issues']:
            print(f"  Warning: {issue}")

    # --- Step 3: Engineer features ---
    print("\n[3/7] Engineering features...")
    feature_engine = FeatureEngine(config.features)
    feature_df, selected_features = feature_engine.build_features(
        market_data, trade_logs, fit_selector=True
    )
    print(f"  Total features computed: {len(feature_df.columns)}")
    print(f"  Selected features: {len(selected_features)}")
    print(f"  Top 5 features: {selected_features[:5]}")

    # --- Step 4: Prepare data for training ---
    print("\n[4/7] Preparing training data...")
    close = feature_df["close"]
    forward_returns = close.pct_change(20).shift(-20).values  # 20-bar forward return
    feature_matrix = feature_df[selected_features].values

    # Remove NaN targets
    valid_mask = ~np.isnan(forward_returns)
    valid_start = max(config.model.lstm_sequence_length + 50, 200)
    valid_mask[:valid_start] = False
    valid_mask[-20:] = False

    feature_matrix = feature_matrix[valid_mask]
    targets = forward_returns[valid_mask]

    split_idx = int(len(feature_matrix) * 0.8)
    purge = config.data.purge_gap

    X_train = feature_matrix[:split_idx]
    y_train = targets[:split_idx]
    X_val = feature_matrix[split_idx + purge:]
    y_val = targets[split_idx + purge:]

    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Feature dimensions: {X_train.shape[1]}")

    # Build sequences for temporal models
    seq_len = config.model.lstm_sequence_length

    def build_sequences(X, y, seq_len):
        Xs, ys = [], []
        for i in range(seq_len, len(X)):
            Xs.append(X[i - seq_len: i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)

    X_train_seq, y_train_seq = build_sequences(X_train, y_train, seq_len)
    X_val_seq, y_val_seq = build_sequences(X_val, y_val, seq_len)

    n_features = X_train.shape[1]

    # --- Step 5: Train models ---
    print("\n[5/7] Training models...")

    print("\n  Training XGBoost...")
    xgb_model = XGBoostPredictor(config=config.model, feature_names=selected_features)
    xgb_result = xgb_model.fit(X_train, y_train, X_val, y_val)
    print(f"  XGBoost: {xgb_result}")

    xgb_metrics = xgb_model.evaluate(X_val, y_val)
    print(f"  XGBoost val metrics: MAE={xgb_metrics['mae']:.6f}, "
          f"Dir.Acc={xgb_metrics['directional_accuracy']:.3f}, "
          f"IC={xgb_metrics['information_coefficient']:.3f}")

    importance = xgb_model.get_feature_importance()
    if importance:
        print(f"  Top 5 important features: {list(importance.keys())[:5]}")

    print("\n  Training LSTM...")
    lstm_model = LSTMPredictor(input_size=n_features, config=config.model, device=config.device)
    lstm_result = lstm_model.fit(X_train_seq, y_train_seq, X_val_seq, y_val_seq)
    print(f"  LSTM: {lstm_result}")

    lstm_metrics = lstm_model.evaluate(X_val_seq, y_val_seq)
    print(f"  LSTM val metrics: MAE={lstm_metrics['mae']:.6f}, "
          f"Dir.Acc={lstm_metrics['directional_accuracy']:.3f}, "
          f"IC={lstm_metrics['information_coefficient']:.3f}")

    print("\n  Training TCN...")
    tcn_model = TCNPredictor(input_size=n_features, config=config.model, device=config.device)
    tcn_result = tcn_model.fit(X_train_seq, y_train_seq, X_val_seq, y_val_seq)
    print(f"  TCN: {tcn_result}")

    tcn_metrics = tcn_model.evaluate(X_val_seq, y_val_seq)
    print(f"  TCN val metrics: MAE={tcn_metrics['mae']:.6f}, "
          f"Dir.Acc={tcn_metrics['directional_accuracy']:.3f}, "
          f"IC={tcn_metrics['information_coefficient']:.3f}")

    # --- Step 6: Meta-learner and ensemble ---
    print("\n[6/7] Running meta-learner ensemble...")
    meta = MetaLearner(config.meta)
    meta.initialize_weights(["xgboost", "lstm", "tcn"])

    returns_col = feature_df["returns"].dropna().values
    vol_col = feature_df.get("realized_vol_20", pd.Series(dtype=float)).dropna().values
    if len(vol_col) > 50 and len(returns_col) > 50:
        min_len = min(len(returns_col), len(vol_col))
        meta.fit_regime_detector(returns_col[:min_len], vol_col[:min_len])

    # Run ensemble on validation set
    seq_input = X_val_seq[-10:]  # last 10 sequences

    xgb_preds, xgb_conf = xgb_model.predict(X_val[-10:])
    lstm_preds, lstm_conf = lstm_model.predict(seq_input)
    tcn_preds, tcn_conf = tcn_model.predict(seq_input)

    predictions = {
        "xgboost": (xgb_preds, xgb_conf),
        "lstm": (lstm_preds, lstm_conf),
        "tcn": (tcn_preds, tcn_conf),
    }

    combined_pred, combined_conf = meta.combine_predictions(predictions)

    print(f"\n  Ensemble predictions (last 10 samples):")
    print(f"  {'Bar':>4} {'Ensemble':>10} {'XGB':>10} {'LSTM':>10} {'TCN':>10} {'Actual':>10} {'Conf':>8}")
    print(f"  {'-' * 68}")

    actual_last_10 = y_val[-10:]
    for i in range(10):
        print(
            f"  {i:>4} "
            f"{combined_pred[i]:>10.6f} "
            f"{xgb_preds[i]:>10.6f} "
            f"{lstm_preds[i]:>10.6f} "
            f"{tcn_preds[i]:>10.6f} "
            f"{actual_last_10[i]:>10.6f} "
            f"{combined_conf[i]:>8.3f}"
        )

    # --- Step 7: Feedback loop demo ---
    print("\n[7/7] Demonstrating feedback loop...")
    feedback = FeedbackLoop(config.feedback)

    for i in range(min(50, len(y_val))):
        pred = combined_pred[i % len(combined_pred)]
        actual = y_val[i]
        conf = combined_conf[i % len(combined_conf)]

        feedback.on_prediction(prediction=pred, confidence=conf, regime="unknown")
        result = feedback.on_actual(pred, actual, conf)

    status = feedback.get_status()
    print(f"  Performance metrics: {status['performance_metrics']}")
    print(f"  Is degrading: {status['is_degrading']}")
    print(f"  Total retrains triggered: {status['total_retrains']}")

    feedback.close()

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"\n  Models trained: XGBoost, LSTM, TCN")
    print(f"  Features used: {len(selected_features)}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"\n  Validation Directional Accuracy:")
    print(f"    XGBoost: {xgb_metrics['directional_accuracy']:.1%}")
    print(f"    LSTM:    {lstm_metrics['directional_accuracy']:.1%}")
    print(f"    TCN:     {tcn_metrics['directional_accuracy']:.1%}")
    print(f"\n  Meta-learner weights: {meta.model_weights}")
    print(f"  Current regime: {meta.current_regime.value}")
    print(f"\n  System is ready for live data ingestion.")
    print("=" * 70)


if __name__ == "__main__":
    main()
