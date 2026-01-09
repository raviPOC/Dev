# Short-Term Price Prediction Models

A comprehensive Python library for predicting short-term price movements (1-hour horizon) using state-of-the-art machine learning models.

## 🎯 Features

- **Multiple Model Architectures**
  - LSTM with Attention mechanism
  - XGBoost (Gradient Boosting)
  - LightGBM (Fast Gradient Boosting)
  - Ensemble models with stacking

- **Comprehensive Feature Engineering**
  - 80+ technical indicators
  - Momentum indicators (RSI, MACD, Stochastic)
  - Volatility indicators (ATR, Bollinger Bands, Keltner)
  - Volume indicators (OBV, MFI, VWAP)
  - Trend indicators (SMA, EMA, ADX)
  - Lagged features and time-based features

- **Production Ready**
  - Model saving/loading
  - Hyperparameter optimization
  - Cross-validation for time series
  - Feature importance analysis

## 📦 Installation

```bash
# Clone the repository
git clone <repository-url>
cd price-predictor

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
torch>=2.0.0
xgboost>=2.0.0
lightgbm>=4.0.0
```

Optional (for real data):
```
yfinance>=0.2.28
ccxt>=4.0.0
```

## 🚀 Quick Start

### Basic Usage

```python
from price_predictor import (
    DataLoader,
    FeatureEngineer,
    XGBoostPredictor,
    LightGBMPredictor,
    EnsemblePredictor,
)

# 1. Load data
loader = DataLoader()
df = loader.generate_synthetic_data(n_samples=5000)

# Or fetch real data:
# df = loader.fetch_yfinance("BTC-USD", interval="1h", period="60d")

# 2. Create features
feature_engineer = FeatureEngineer()
df_features = feature_engineer.create_all_features(
    df, 
    include_target=True,
    target_horizon=1  # Predict 1 hour ahead
)

# 3. Split data
df_clean = df_features.dropna()
train_df, val_df, test_df = loader.train_test_split(df_clean)

feature_cols = feature_engineer.feature_names
X_train, y_train = train_df[feature_cols], train_df["target"]
X_val, y_val = val_df[feature_cols], val_df["target"]
X_test, y_test = test_df[feature_cols], test_df["target"]

# 4. Train model
model = XGBoostPredictor(n_estimators=500, learning_rate=0.05)
model.fit(X_train, y_train, X_val, y_val)

# 5. Evaluate
metrics = model.evaluate(X_test, y_test)
print(f"Direction Accuracy: {metrics['direction_accuracy']:.2%}")

# 6. Predict
predictions = model.predict(X_test)
```

### Using LSTM

```python
from price_predictor import LSTMPredictor, FeatureEngineer

# Prepare sequences
X_sequences, y_targets = feature_engineer.prepare_sequences(
    df_clean,
    sequence_length=60,  # Use last 60 hours
    target_col="target"
)

# Train LSTM
lstm = LSTMPredictor(
    sequence_length=60,
    hidden_size=128,
    num_layers=2,
    use_attention=True,
    epochs=100
)
lstm.fit(X_train, y_train, X_val, y_val)
```

### Using Ensemble

```python
from price_predictor import EnsemblePredictor

ensemble = EnsemblePredictor(
    ensemble_method="stacking"  # or "average", "weighted"
)
ensemble.fit(X_train, y_train, X_val, y_val)

# Get individual model predictions
individual_preds = ensemble.predict_individual(X_test)
```

## 📊 Models Overview

### 1. LSTM (Long Short-Term Memory)

Best for capturing temporal patterns and trends.

```python
LSTMPredictor(
    sequence_length=60,    # Input sequence length
    hidden_size=128,       # LSTM hidden size
    num_layers=2,          # Number of LSTM layers
    dropout=0.2,           # Regularization
    use_attention=True,    # Attention mechanism
    bidirectional=False,   # Bidirectional LSTM
)
```

### 2. XGBoost

Best for tabular data with engineered features.

```python
XGBoostPredictor(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
)
```

### 3. LightGBM

Fastest training, good for large datasets.

```python
LightGBMPredictor(
    n_estimators=1000,
    num_leaves=31,
    learning_rate=0.05,
    min_child_samples=20,
)
```

### 4. Ensemble

Combines multiple models for robust predictions.

```python
EnsemblePredictor(
    models=[...],  # Optional custom models
    ensemble_method="stacking",  # "average", "weighted", "stacking"
)
```

## 📈 Feature Engineering

The `FeatureEngineer` class creates 80+ features:

| Category | Features |
|----------|----------|
| **Returns** | 1, 5, 10, 20 period returns, log returns |
| **Momentum** | RSI (7,14,21), Stochastic, MACD, Williams %R, ROC |
| **Volatility** | ATR (7,14,21), Bollinger Bands, Keltner Channels |
| **Trend** | SMA/EMA (5,10,20,50), ADX, Linear Regression Slope |
| **Volume** | OBV, MFI, VWAP, Volume ratios |
| **Time** | Hour, Day of week, Cyclical encodings |

## 🔧 API Reference

### DataLoader

```python
# Load from CSV
df = loader.load_csv("prices.csv")

# Generate synthetic data
df = loader.generate_synthetic_data(n_samples=5000)

# Fetch from Yahoo Finance
df = loader.fetch_yfinance("BTC-USD", interval="1h", period="60d")

# Fetch from crypto exchange
df = loader.fetch_ccxt("binance", "BTC/USDT", timeframe="1h")

# Split data
train, val, test = loader.train_test_split(df, train_ratio=0.8)
```

### FeatureEngineer

```python
# Create all features
df_features = feature_engineer.create_all_features(
    df,
    include_target=True,
    target_horizon=1
)

# Get feature names
features = feature_engineer.get_feature_names()

# Prepare sequences for LSTM
X_seq, y = feature_engineer.prepare_sequences(df, sequence_length=60)
```

### Model Methods

All models share these methods:

```python
# Train
model.fit(X_train, y_train, X_val, y_val)

# Predict
predictions = model.predict(X_test)

# Evaluate
metrics = model.evaluate(X_test, y_test)
# Returns: mse, rmse, mae, r2, direction_accuracy, profit_factor

# Save/Load
model.save("model.pkl")
model.load("model.pkl")

# Feature importance (XGBoost/LightGBM)
importance = model.get_feature_importance(top_n=20)
```

## 📁 Project Structure

```
price_predictor/
├── __init__.py
├── models/
│   ├── __init__.py
│   ├── base.py           # Base predictor class
│   ├── lstm_model.py     # LSTM implementation
│   ├── xgboost_model.py  # XGBoost implementation
│   ├── lightgbm_model.py # LightGBM implementation
│   └── ensemble.py       # Ensemble methods
├── utils/
│   ├── __init__.py
│   ├── features.py       # Feature engineering
│   └── data_loader.py    # Data loading utilities
└── data/
    └── __init__.py

examples/
├── basic_usage.py        # Basic workflow example
├── lstm_example.py       # LSTM-specific example
└── real_data_example.py  # Real market data example
```

## 🎓 Best Practices

### For 1-Hour Predictions

1. **Use multiple timeframes**: Include features from 5m, 15m, and 1h data
2. **Ensemble models**: Combine XGBoost + LightGBM + LSTM
3. **Recent data**: Train on last 30-60 days of hourly data
4. **Feature selection**: Use top 30-50 features by importance
5. **Regularization**: Use dropout (LSTM) and L1/L2 (boosting)

### Model Selection Guide

| Scenario | Recommended Model |
|----------|------------------|
| Fast inference needed | LightGBM |
| Best accuracy | Ensemble (stacking) |
| Capturing trends | LSTM with attention |
| Limited data | XGBoost |
| Real-time trading | LightGBM |

### Hyperparameter Optimization

```python
# XGBoost hyperparameter optimization
best_params = model.optimize_hyperparameters(
    X_train, y_train, X_val, y_val,
    n_trials=50
)
```

## ⚠️ Disclaimer

This software is for educational purposes only. Financial markets are inherently unpredictable, and past performance does not guarantee future results. Always:

- Paper trade before using real money
- Use proper risk management
- Never invest more than you can afford to lose
- Consider transaction costs and slippage

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions welcome! Please read CONTRIBUTING.md for guidelines.

## 📞 Support

- Create an issue for bug reports
- Discussions for questions and feature requests
