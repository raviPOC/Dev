# DRL Stock Price Prediction

This project implements Deep Reinforcement Learning (DRL) techniques for stock trading and price prediction. It uses `gymnasium` to create a custom stock trading environment and `stable-baselines3` to train PPO agents.

## Project Structure

- `data/`: Stores downloaded stock data.
- `models/`: Stores trained DRL models.
- `src/`: Source code.
  - `data_download.py`: Script to download stock data using `yfinance`.
  - `env.py`: Custom OpenAI Gym environment for stock trading.
  - `train.py`: Script to train the DRL agent.
  - `test.py`: Script to test/evaluate the trained agent.
- `requirements.txt`: Python dependencies.

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Download Data
Download historical stock data for a specific ticker:
```bash
python3 src/data_download.py --ticker AAPL --start 2018-01-01 --end 2023-01-01
```

### 2. Train Agent
Train a PPO agent on the downloaded data:
```bash
python3 src/train.py --data data/AAPL_2018-01-01_2023-01-01.csv --timesteps 10000
```
The trained model will be saved in the `models/` directory.

### 3. Evaluate Agent
Test the trained agent and visualize performance:
```bash
python3 src/test.py --data data/AAPL_2018-01-01_2023-01-01.csv --model models/ppo_stock_trader
```
This will output the final net worth and save a performance plot to `performance.png`.

## Environment Details

The custom environment (`StockTradingEnv`) simulates a simplified trading account.
- **Action Space**: Discrete(3) - Sell (0), Hold (1), Buy (2).
- **Observation Space**: [Open, High, Low, Close, Volume, Balance, Net Worth, Shares Held].
- **Reward**: Change in Net Worth (Profit).

## Future Work

- Add more technical indicators (RSI, MACD, etc.) to the observation space.
- Implement more advanced reward functions (Sharpe ratio, etc.).
- Support multiple stocks/portfolio management.
- Tune hyperparameters for better performance.
