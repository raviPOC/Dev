#!/usr/bin/env python3
"""
Real Data Example - Fetching and Predicting Crypto/Stock Prices

This example shows how to:
1. Fetch real price data from Yahoo Finance or Crypto exchanges
2. Train models on real market data
3. Make actionable predictions

Note: Requires additional packages: yfinance, ccxt
"""

import sys
sys.path.insert(0, "..")

import numpy as np
import pandas as pd
from datetime import datetime

from price_predictor import (
    DataLoader,
    FeatureEngineer,
    XGBoostPredictor,
    LightGBMPredictor,
    EnsemblePredictor,
)


def fetch_and_predict_yfinance(symbol: str = "BTC-USD"):
    """
    Fetch data from Yahoo Finance and make predictions.
    
    Args:
        symbol: Ticker symbol (e.g., BTC-USD, ETH-USD, AAPL, SPY)
    """
    print(f"\n{'='*60}")
    print(f"Predicting {symbol} - Yahoo Finance Data")
    print(f"{'='*60}")
    
    # Load data
    loader = DataLoader()
    
    try:
        df = loader.fetch_yfinance(
            symbol=symbol,
            interval="1h",
            period="60d",  # Last 60 days of hourly data
        )
        print(f"✓ Fetched {len(df)} hourly candles")
    except ImportError:
        print("✗ yfinance not installed. Install with: pip install yfinance")
        print("  Using synthetic data instead...")
        df = loader.generate_synthetic_data(n_samples=1440)  # ~60 days
    
    # Feature engineering
    feature_engineer = FeatureEngineer()
    df_features = feature_engineer.create_all_features(
        df, include_target=True, target_horizon=1
    )
    df_clean = df_features.dropna()
    
    print(f"✓ Created {len(feature_engineer.feature_names)} features")
    
    # Split data
    train_df, val_df, test_df = loader.train_test_split(
        df_clean, train_ratio=0.7, validation_ratio=0.15
    )
    
    feature_cols = feature_engineer.feature_names
    
    X_train = train_df[feature_cols]
    y_train = train_df["target"]
    X_val = val_df[feature_cols]
    y_val = val_df["target"]
    X_test = test_df[feature_cols]
    y_test = test_df["target"]
    
    # Train ensemble
    print("\n Training ensemble model...")
    ensemble = EnsemblePredictor(ensemble_method="stacking")
    ensemble.fit(X_train, y_train, X_val, y_val, verbose=False)
    
    # Evaluate
    metrics = ensemble.evaluate(X_test, y_test)
    
    print(f"\n📊 Test Results for {symbol}:")
    print(f"   RMSE: {metrics['rmse']:.6f}")
    print(f"   Direction Accuracy: {metrics['direction_accuracy']:.2%}")
    print(f"   Profit Factor: {metrics['profit_factor']:.2f}")
    
    # Latest prediction
    latest_features = df_clean[feature_cols].iloc[[-1]]
    latest_pred = ensemble.predict(latest_features)[0]
    
    current_price = df["close"].iloc[-1]
    predicted_direction = "📈 UP" if latest_pred > 0 else "📉 DOWN"
    predicted_change = abs(latest_pred) * 100
    
    print(f"\n🔮 Next Hour Prediction for {symbol}:")
    print(f"   Current Price: ${current_price:.2f}")
    print(f"   Predicted Direction: {predicted_direction}")
    print(f"   Predicted Change: {predicted_change:.3f}%")
    print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return ensemble, metrics


def fetch_and_predict_ccxt(exchange: str = "binance", symbol: str = "BTC/USDT"):
    """
    Fetch data from cryptocurrency exchange and make predictions.
    
    Args:
        exchange: Exchange name (binance, coinbase, kraken, etc.)
        symbol: Trading pair
    """
    print(f"\n{'='*60}")
    print(f"Predicting {symbol} on {exchange}")
    print(f"{'='*60}")
    
    loader = DataLoader()
    
    try:
        df = loader.fetch_ccxt(
            exchange=exchange,
            symbol=symbol,
            timeframe="1h",
            limit=1000,
        )
        print(f"✓ Fetched {len(df)} hourly candles from {exchange}")
    except ImportError:
        print("✗ ccxt not installed. Install with: pip install ccxt")
        print("  Using synthetic data instead...")
        df = loader.generate_synthetic_data(n_samples=1000)
    except Exception as e:
        print(f"✗ Error fetching from {exchange}: {e}")
        print("  Using synthetic data instead...")
        df = loader.generate_synthetic_data(n_samples=1000)
    
    # Feature engineering
    feature_engineer = FeatureEngineer()
    df_features = feature_engineer.create_all_features(
        df, include_target=True, target_horizon=1
    )
    df_clean = df_features.dropna()
    
    # Split and train
    train_df, val_df, test_df = loader.train_test_split(
        df_clean, train_ratio=0.7, validation_ratio=0.15
    )
    
    feature_cols = feature_engineer.feature_names
    
    X_train = train_df[feature_cols]
    y_train = train_df["target"]
    X_val = val_df[feature_cols]
    y_val = val_df["target"]
    X_test = test_df[feature_cols]
    y_test = test_df["target"]
    
    # Train LightGBM (fastest)
    print("\n Training LightGBM model...")
    model = LightGBMPredictor(
        n_estimators=500,
        learning_rate=0.05,
        early_stopping_rounds=30,
    )
    model.fit(X_train, y_train, X_val, y_val, verbose=False)
    
    # Evaluate
    metrics = model.evaluate(X_test, y_test)
    
    print(f"\n📊 Results:")
    print(f"   Direction Accuracy: {metrics['direction_accuracy']:.2%}")
    print(f"   Profit Factor: {metrics['profit_factor']:.2f}")
    
    return model, metrics


def backtest_strategy(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    initial_capital: float = 10000,
):
    """
    Simple backtest of trading strategy based on model predictions.
    
    Args:
        model: Trained prediction model
        X_test: Test features
        y_test: Actual returns
        initial_capital: Starting capital
    """
    print("\n📈 Backtesting Strategy...")
    
    predictions = model.predict(X_test)
    returns = y_test.values
    
    # Simple strategy: go long if pred > 0, short if pred < 0
    positions = np.sign(predictions)
    strategy_returns = positions * returns
    
    # Calculate cumulative returns
    cumulative_returns = (1 + strategy_returns).cumprod()
    buy_hold_returns = (1 + returns).cumprod()
    
    final_value = initial_capital * cumulative_returns[-1]
    buy_hold_value = initial_capital * buy_hold_returns[-1]
    
    # Calculate metrics
    sharpe_ratio = np.sqrt(252 * 24) * strategy_returns.mean() / strategy_returns.std()
    max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
    win_rate = (strategy_returns > 0).mean()
    
    print(f"\n   Initial Capital: ${initial_capital:,.2f}")
    print(f"   Final Value (Strategy): ${final_value:,.2f}")
    print(f"   Final Value (Buy & Hold): ${buy_hold_value:,.2f}")
    print(f"   Strategy Return: {(cumulative_returns[-1] - 1) * 100:.2f}%")
    print(f"   Buy & Hold Return: {(buy_hold_returns[-1] - 1) * 100:.2f}%")
    print(f"   Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"   Max Drawdown: {max_drawdown * 100:.2f}%")
    print(f"   Win Rate: {win_rate * 100:.2f}%")
    
    return {
        "final_value": final_value,
        "strategy_return": cumulative_returns[-1] - 1,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
    }


def main():
    """Run all examples."""
    print("=" * 60)
    print("Real Data Price Prediction Examples")
    print("=" * 60)
    
    # Example 1: Bitcoin from Yahoo Finance
    try:
        model1, metrics1 = fetch_and_predict_yfinance("BTC-USD")
    except Exception as e:
        print(f"Yahoo Finance example failed: {e}")
    
    # Example 2: Ethereum from Yahoo Finance
    try:
        model2, metrics2 = fetch_and_predict_yfinance("ETH-USD")
    except Exception as e:
        print(f"ETH example failed: {e}")
    
    # Example 3: Crypto exchange data
    try:
        model3, metrics3 = fetch_and_predict_ccxt("binance", "BTC/USDT")
    except Exception as e:
        print(f"CCXT example failed: {e}")
    
    print("\n" + "=" * 60)
    print("Examples Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
