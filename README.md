# Predictive Trading System

A self-learning, multi-horizon trading prediction system that ingests trading logs, engineers features, runs an ensemble of ML models, and adapts through a meta-learning feedback loop.

Built for quantitative day traders who want a systematic, adaptive prediction engine.

## What This Is (and Isn't)

**What it is:** A proper ML pipeline for predicting short-to-long term price movements using:
- 50+ engineered technical and behavioral features
- Ensemble of LSTM, TCN, and XGBoost models
- HMM-based market regime detection
- Meta-learner that adapts model weights based on recent performance and regime
- Feedback loop that detects degradation and triggers retraining

**What it isn't:** A magic box that prints money. This is a research tool and prediction framework. You still need to:
- Provide quality data
- Understand the predictions
- Manage risk yourself
- Validate on out-of-sample data before any live use

## Architecture

```
Trading Logs → Feature Engineering → Model Ensemble → Meta-Learner → Predictions
     ↑                                                      |
     └──────────── Feedback Loop (performance tracking) ─────┘
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the detailed system design.

See [RESEARCH_MODEL_COMPARISON.md](RESEARCH_MODEL_COMPARISON.md) for an honest comparison of AI models for quant trading research.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the demo with synthetic data
python run_demo.py
```

## Project Structure

```
predictive_trading_system/
├── config/
│   └── settings.py          # All hyperparameters and configuration
├── data/
│   └── ingestion.py         # Data loading, validation, trading log parsing
├── features/
│   └── engineering.py       # 50+ technical, volatility, volume, behavioral features
├── models/
│   ├── base.py              # Abstract model interface
│   ├── lstm_model.py        # LSTM + Attention (medium-term)
│   ├── tcn_model.py         # Temporal Convolutional Network (short-term)
│   ├── xgboost_model.py     # Gradient Boosted Trees (all horizons)
│   └── regime_detector.py   # HMM market regime detection
├── meta/
│   └── meta_learner.py      # Adaptive ensemble weighting
├── core/
│   ├── pipeline.py          # Main orchestrator
│   └── feedback.py          # Performance monitoring, retraining triggers
├── utils/
│   └── data_generator.py    # Synthetic data for testing
└── logs/                    # Prediction and performance logs
```

## Using Your Own Data

### Market Data (OHLCV)

Provide a CSV, JSON, or Parquet file with columns: `open`, `high`, `low`, `close`, `volume` and a datetime index.

```python
from predictive_trading_system.core.pipeline import PredictionPipeline

pipeline = PredictionPipeline()
pipeline.load_data("your_market_data.csv", "your_trading_logs.csv")
pipeline.train(target_horizon="medium", target_periods=20)
result = pipeline.predict()
print(result)
```

### Trading Logs

The system accepts various formats with common column names (case-insensitive):

| Your Column | Recognized Aliases |
|---|---|
| timestamp | date, time, datetime, trade_date, exec_time |
| symbol | ticker, instrument, asset |
| side | direction, action, type |
| quantity | qty, size, amount, shares |
| price | fill_price, avg_price, exec_price |
| pnl | profit, profit_loss, realized_pnl, pl |

## Key Features

- **Multi-horizon predictions**: Short (minutes), medium (hours-day), long (days-week)
- **Regime-aware**: Detects market regimes and adjusts model weights accordingly
- **Self-learning**: Feedback loop monitors performance and triggers retraining when accuracy degrades
- **Your trading patterns as features**: Win rate trends, drawdown state, and position sizing behavior feed into the model
- **Anti-overfitting**: Walk-forward validation with purging. No peeking at future data.
- **Graceful degradation**: Works without PyTorch (linear fallback), without XGBoost (numpy fallback), without hmmlearn (heuristic regime detection)

## Configuration

All hyperparameters are in `predictive_trading_system/config/settings.py`. Key settings:

```python
from predictive_trading_system.config.settings import SystemConfig

config = SystemConfig()
config.model.lstm_hidden_size = 256       # bigger LSTM
config.model.xgb_n_estimators = 1000      # more XGBoost trees
config.meta.performance_window = 200      # longer evaluation window
config.feedback.retrain_error_threshold = 0.1  # more aggressive retraining
```

## License

MIT — Use at your own risk. No financial advice is provided.
