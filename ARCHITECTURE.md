# Predictive Trading System — Architecture Document

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PREDICTIVE TRADING SYSTEM                            │
│                                                                         │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────┐                │
│  │ Trading Log │──▶│   Feature    │──▶│   Model      │                │
│  │  Ingestion  │   │  Engineering │   │   Ensemble   │                │
│  └─────────────┘   └──────────────┘   └──────┬───────┘                │
│                                               │                         │
│                                               ▼                         │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────┐                │
│  │  Feedback   │◀──│ Meta-Learner │◀──│  Prediction  │                │
│  │    Loop     │   │  (Regime     │   │   Outputs    │                │
│  └──────┬──────┘   │   Adaptive)  │   └──────────────┘                │
│         │          └──────────────┘                                     │
│         │                                                               │
│         ▼                                                               │
│  ┌─────────────┐                                                        │
│  │ Model       │                                                        │
│  │ Retraining  │                                                        │
│  └─────────────┘                                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Data Ingestion Layer (`data/`)
- Reads trading logs (CSV, JSON, or database exports)
- Normalizes timestamps, handles missing data
- Supports multiple data sources: personal trade logs, OHLCV market data, order book data
- Validates data integrity and flags anomalies

### 2. Feature Engineering Pipeline (`features/`)
- **Price Features:** Returns, log returns, moving averages, VWAP
- **Volatility Features:** Realized vol, Garman-Klass, Parkinson, ATR
- **Momentum Features:** RSI, MACD, Stochastic, ROC, Williams %R
- **Volume Features:** OBV, volume profile, VWAP deviation
- **Microstructure Features:** Bid-ask spread, order imbalance, trade flow toxicity
- **Custom Features:** From your trading logs — win rate trends, drawdown patterns, position sizing behavior
- **Regime Features:** Hidden Markov Model states, volatility regime classification

### 3. Model Ensemble (`models/`)
Multi-horizon prediction with specialized models:

| Horizon | Model | Why |
|---|---|---|
| Short-term (1-60 min) | Temporal Convolutional Network | Fast, captures local patterns |
| Medium-term (1h-1d) | LSTM + Attention | Good sequential memory |
| Long-term (1d-1w) | Temporal Fusion Transformer | Best multi-horizon, interpretable |
| Regime Detection | Hidden Markov Model | Unsupervised regime switching |
| Tabular Features | XGBoost/LightGBM | Robust on engineered features |

### 4. Meta-Learning Layer (`meta/`)
- Tracks per-model performance across different market regimes
- Dynamically adjusts ensemble weights based on recent accuracy
- Detects regime changes and switches model allocations
- Implements a "model confidence" score that discounts predictions during regime transitions

### 5. Feedback Loop (`core/`)
- Compares predictions against actual outcomes
- Logs prediction errors with full context (features, regime, confidence)
- Triggers retraining when error metrics exceed thresholds
- Implements adaptive learning rate scheduling based on recent performance
- Tracks feature importance drift over time

### 6. Trading Log Analyzer (`data/`)
- Parses YOUR personal trading logs
- Extracts behavioral patterns: time-of-day biases, size scaling behavior, hold duration patterns
- Identifies your edge: when/where you tend to be right vs. wrong
- Feeds this behavioral data as features into the prediction models

## Data Flow

```
Raw Data (OHLCV + Your Trades)
        │
        ▼
┌───────────────────┐
│ Data Validation    │ ── Anomaly flags
│ & Normalization    │
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│ Feature Engine     │ ── 50+ engineered features
│ (Technical +       │
│  Behavioral +      │
│  Regime)           │
└───────┬───────────┘
        │
        ▼
┌───────────────────┐     ┌─────────────────┐
│ Model Ensemble     │────▶│ Meta-Learner     │
│ (5 model types)    │     │ (Weight Adj.)    │
└───────────────────┘     └────────┬────────┘
                                   │
                                   ▼
                          ┌─────────────────┐
                          │ Final Prediction │
                          │ + Confidence     │
                          │ + Horizon        │
                          └────────┬────────┘
                                   │
                          ┌────────▼────────┐
                          │ Performance      │
                          │ Tracker          │──▶ Retraining trigger
                          └─────────────────┘
```

## Key Design Decisions

1. **Multi-horizon by design** — Not one model predicting everything, but specialized models for each time horizon, combined by the meta-learner.

2. **Regime-aware** — Markets behave differently in different regimes (trending, mean-reverting, high-vol, low-vol). The system detects regimes and adapts model weights accordingly.

3. **Your edge is a feature** — Your personal trading patterns (when you're hot, when you're not, what setups you nail) become first-class features in the prediction pipeline.

4. **Anti-overfitting by default** — Walk-forward validation only. No peeking. Purged cross-validation for time series. Out-of-sample testing is mandatory.

5. **Interpretability** — Every prediction comes with feature importance scores and confidence intervals. Black box on the inside, glass box on the outside.
