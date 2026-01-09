#!/usr/bin/env python3
"""
Basic Usage Example for Short-Term Price Prediction

This example demonstrates how to:
1. Generate or load price data
2. Create features using technical indicators
3. Train multiple models (XGBoost, LightGBM, LSTM)
4. Evaluate and compare model performance
5. Make predictions

Run this script to see the full workflow in action.
"""

import sys
sys.path.insert(0, "..")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from price_predictor import (
    DataLoader,
    FeatureEngineer,
    XGBoostPredictor,
    LightGBMPredictor,
    LSTMPredictor,
    EnsemblePredictor,
)


def main():
    print("=" * 60)
    print("Short-Term Price Prediction - Basic Example")
    print("=" * 60)
    
    # =========================================================
    # Step 1: Load or Generate Data
    # =========================================================
    print("\n[1] Loading Data...")
    
    loader = DataLoader()
    
    # Generate synthetic hourly price data (for demo purposes)
    # In production, use loader.fetch_yfinance() or loader.load_csv()
    df = loader.generate_synthetic_data(
        n_samples=5000,
        start_price=100.0,
        volatility=0.015,
        trend=0.00005,
        seed=42,
    )
    
    print(f"   Generated {len(df)} hourly price samples")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")
    print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    # =========================================================
    # Step 2: Feature Engineering
    # =========================================================
    print("\n[2] Creating Features...")
    
    feature_engineer = FeatureEngineer()
    
    # Create all features including target (1 hour ahead prediction)
    df_features = feature_engineer.create_all_features(
        df,
        include_target=True,
        target_horizon=1,  # Predict 1 hour ahead
    )
    
    print(f"   Created {len(feature_engineer.feature_names)} features")
    print(f"   Sample features: {feature_engineer.feature_names[:10]}...")
    
    # =========================================================
    # Step 3: Prepare Train/Validation/Test Sets
    # =========================================================
    print("\n[3] Splitting Data...")
    
    # Drop rows with NaN values (from feature calculations)
    df_clean = df_features.dropna()
    
    # Time-based split (80% train, 10% validation, 10% test)
    train_df, val_df, test_df = loader.train_test_split(
        df_clean,
        train_ratio=0.8,
        validation_ratio=0.1,
    )
    
    # Prepare features and targets
    feature_cols = feature_engineer.feature_names
    target_col = "target"
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]
    
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    print(f"   Train: {len(X_train)} samples")
    print(f"   Validation: {len(X_val)} samples")
    print(f"   Test: {len(X_test)} samples")
    
    # =========================================================
    # Step 4: Train Individual Models
    # =========================================================
    print("\n[4] Training Models...")
    
    results = {}
    
    # --- XGBoost Model ---
    print("\n   Training XGBoost...")
    xgb_model = XGBoostPredictor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        early_stopping_rounds=30,
    )
    xgb_model.fit(X_train, y_train, X_val, y_val, verbose=False)
    results["XGBoost"] = xgb_model.evaluate(X_test, y_test)
    print(f"   XGBoost - RMSE: {results['XGBoost']['rmse']:.6f}, "
          f"Direction Accuracy: {results['XGBoost']['direction_accuracy']:.4f}")
    
    # --- LightGBM Model ---
    print("\n   Training LightGBM...")
    lgb_model = LightGBMPredictor(
        n_estimators=500,
        num_leaves=31,
        learning_rate=0.05,
        early_stopping_rounds=30,
    )
    lgb_model.fit(X_train, y_train, X_val, y_val, verbose=False)
    results["LightGBM"] = lgb_model.evaluate(X_test, y_test)
    print(f"   LightGBM - RMSE: {results['LightGBM']['rmse']:.6f}, "
          f"Direction Accuracy: {results['LightGBM']['direction_accuracy']:.4f}")
    
    # --- Ensemble Model ---
    print("\n   Training Ensemble...")
    ensemble = EnsemblePredictor(
        models=[
            XGBoostPredictor(n_estimators=300, max_depth=6, learning_rate=0.05),
            LightGBMPredictor(n_estimators=300, num_leaves=31, learning_rate=0.05),
            XGBoostPredictor(n_estimators=300, max_depth=8, learning_rate=0.1),
            LightGBMPredictor(n_estimators=300, num_leaves=63, learning_rate=0.1),
        ],
        ensemble_method="stacking",
    )
    ensemble.fit(X_train, y_train, X_val, y_val, verbose=False)
    results["Ensemble"] = ensemble.evaluate(X_test, y_test)
    print(f"   Ensemble - RMSE: {results['Ensemble']['rmse']:.6f}, "
          f"Direction Accuracy: {results['Ensemble']['direction_accuracy']:.4f}")
    
    # =========================================================
    # Step 5: Compare Results
    # =========================================================
    print("\n[5] Model Comparison")
    print("=" * 60)
    
    comparison_df = pd.DataFrame(results).T
    comparison_df = comparison_df[["rmse", "mae", "r2", "direction_accuracy", "profit_factor"]]
    print(comparison_df.to_string())
    
    # =========================================================
    # Step 6: Make Predictions
    # =========================================================
    print("\n[6] Making Predictions on Test Data...")
    
    # Use the best model (ensemble) for predictions
    predictions = ensemble.predict(X_test)
    
    # Create prediction DataFrame
    pred_df = pd.DataFrame({
        "actual_return": y_test.values,
        "predicted_return": predictions,
        "actual_direction": (y_test.values > 0).astype(int),
        "predicted_direction": (predictions > 0).astype(int),
    }, index=test_df.index)
    
    pred_df["correct"] = pred_df["actual_direction"] == pred_df["predicted_direction"]
    
    print(f"\n   Sample Predictions (last 10):")
    print(pred_df.tail(10).to_string())
    
    # =========================================================
    # Step 7: Feature Importance Analysis
    # =========================================================
    print("\n[7] Top 15 Most Important Features (XGBoost):")
    print("-" * 40)
    importance = xgb_model.get_feature_importance(15)
    for _, row in importance.iterrows():
        print(f"   {row['feature']:30} {row['importance']:.4f}")
    
    # =========================================================
    # Summary Statistics
    # =========================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    best_model = min(results, key=lambda x: results[x]["rmse"])
    print(f"Best Model (by RMSE): {best_model}")
    print(f"  - RMSE: {results[best_model]['rmse']:.6f}")
    print(f"  - Direction Accuracy: {results[best_model]['direction_accuracy']:.2%}")
    print(f"  - Profit Factor: {results[best_model]['profit_factor']:.2f}")
    
    print("\nModel training and evaluation complete!")
    print("=" * 60)
    
    return results, ensemble, pred_df


if __name__ == "__main__":
    results, model, predictions = main()
