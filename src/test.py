import pandas as pd
import numpy as np
from stable_baselines3 import PPO
import os
import argparse
import matplotlib.pyplot as plt
from env import StockTradingEnv

def test_agent(data_path, model_path):
    """
    Tests a trained PPO agent on the stock trading environment.
    """
    # Load data
    df = pd.read_csv(data_path)
    if 'Date' in df.columns:
        df = df.sort_values('Date')
    
    # Filter numerical columns
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [c for c in numeric_cols if c not in df.columns]
    if missing_cols:
         numeric_df = df.select_dtypes(include=[np.number])
    else:
        numeric_df = df[numeric_cols]

    # Create environment
    env = StockTradingEnv(numeric_df)
    
    # Load model
    model = PPO.load(model_path)
    
    # Run simulation
    obs, _ = env.reset()
    done = False
    
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        
    # Analyze results
    history = pd.DataFrame(env.history)
    
    print("Final Net Worth:", env.net_worth)
    print("Initial Balance:", env.initial_balance)
    print("Profit:", env.net_worth - env.initial_balance)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(history['step'], history['net_worth'], label='Net Worth')
    plt.plot(history['step'], history['price'] / history['price'].iloc[0] * env.initial_balance, label='Buy and Hold (scaled)', alpha=0.5)
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.title('Agent Performance vs Buy and Hold')
    plt.legend()
    plt.savefig('performance.png')
    print("Performance plot saved to performance.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test DRL agent for stock trading")
    parser.add_argument("--data", type=str, required=True, help="Path to the test data CSV")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model zip file")
    
    args = parser.parse_args()
    
    test_agent(args.data, args.model)
