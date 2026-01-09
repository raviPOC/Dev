import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance=10000, commission_fee=0.001):
        super(StockTradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.commission_fee = commission_fee
        
        # Action space: 0 = Sell, 1 = Hold, 2 = Buy
        self.action_space = spaces.Discrete(3)

        # Observation space: 
        # [Open, High, Low, Close, Volume, Balance, Net Worth, Shares Held]
        # We might want to add technical indicators later, but let's start simple.
        # Assuming the dataframe has OHLVC columns.
        self.observation_shape = (len(self.df.columns) + 3,) 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.observation_shape, dtype=np.float32)

        self.reset()

    def _next_observation(self):
        # Get the data for the current step
        frame = self.df.iloc[self.current_step]
        
        # Append additional state info
        obs = np.append(frame.values, [
            self.balance,
            self.net_worth,
            self.shares_held
        ])
        
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.shares_held = 0
        self.current_step = 0
        
        # Track history for rendering
        self.history = []

        return self._next_observation(), {}

    def step(self, action):
        current_price = self.df.iloc[self.current_step]['Close']
        
        # Execute action
        if action == 0: # Sell
            if self.shares_held > 0:
                # Sell all shares
                revenue = self.shares_held * current_price
                commission = revenue * self.commission_fee
                self.balance += revenue - commission
                self.shares_held = 0
        
        elif action == 2: # Buy
            # Buy as many shares as possible
            max_shares = int(self.balance / (current_price * (1 + self.commission_fee)))
            if max_shares > 0:
                cost = max_shares * current_price
                commission = cost * self.commission_fee
                self.balance -= (cost + commission)
                self.shares_held += max_shares
        
        # Update net worth
        self.net_worth = self.balance + (self.shares_held * current_price)
        
        # Record step
        self.history.append({
            'step': self.current_step,
            'price': current_price,
            'balance': self.balance,
            'net_worth': self.net_worth,
            'shares': self.shares_held,
            'action': action
        })

        self.current_step += 1
        
        done = self.current_step >= len(self.df) - 1
        terminated = done
        truncated = False
        
        reward = self.net_worth - self.initial_balance # Simple profit reward
        # Alternatively: reward = self.net_worth - previous_net_worth
        
        obs = self._next_observation()
        
        return obs, reward, terminated, truncated, {}

    def render(self, mode='human'):
        if mode == 'human':
            print(f"Step: {self.current_step}, Net Worth: {self.net_worth:.2f}, Shares: {self.shares_held}")

