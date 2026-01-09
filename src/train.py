import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import argparse
from env import StockTradingEnv

def train_agent(data_path, timesteps=10000, output_dir="models"):
    """
    Trains a PPO agent on the stock trading environment.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load data
    df = pd.read_csv(data_path)
    
    # Sort by date just in case
    if 'Date' in df.columns:
        df = df.sort_values('Date')
    
    # Filter numerical columns for the environment
    # We expect columns like Open, High, Low, Close, Volume
    # We might need to handle the 'Date' column or other non-numeric columns
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    # Ensure these columns exist
    missing_cols = [c for c in numeric_cols if c not in df.columns]
    if missing_cols:
        # Try finding similar columns (e.g. 'Adj Close')
        # This is a basic check.
        print(f"Warning: Missing columns {missing_cols}. Using available numeric columns.")
        numeric_df = df.select_dtypes(include=[np.number])
    else:
        numeric_df = df[numeric_cols]

    # Create environment
    env = StockTradingEnv(numeric_df)
    env = DummyVecEnv([lambda: env])

    # Initialize agent
    model = PPO("MlpPolicy", env, verbose=1)

    # Train agent
    print("Training agent...")
    model.learn(total_timesteps=timesteps)

    # Save model
    model_path = os.path.join(output_dir, "ppo_stock_trader")
    model.save(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DRL agent for stock trading")
    parser.add_argument("--data", type=str, required=True, help="Path to the training data CSV")
    parser.add_argument("--timesteps", type=int, default=10000, help="Number of training timesteps")
    parser.add_argument("--output", type=str, default="models", help="Output directory for models")
    
    args = parser.parse_args()
    
    train_agent(args.data, args.timesteps, args.output)
