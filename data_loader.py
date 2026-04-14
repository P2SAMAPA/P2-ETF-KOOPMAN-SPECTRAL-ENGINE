"""
Data loader for Koopman-Spectral engine.
Pulls from P2 shared data hub: OHLCV, log returns, volatility, macro.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def fetch_etf_data(etf_symbol, config):
    """
    Load pre-computed features from p2-etf-deepm-data hub.
    Expected columns: open, high, low, close, volume, returns, log_returns, vol
    """
    data_path = Path(f"/mnt/data/{config['data']['source']}/{etf_symbol}.parquet")
    df = pd.read_parquet(data_path)
    df['etf'] = etf_symbol
    return df


def fetch_macro_data(config):
    """Load macro features: VIX, T10Y2Y, DXY, HY/IG spreads, WTI, DTB3"""
    macro_path = Path(f"/mnt/data/{config['data']['source']}/macro.parquet")
    return pd.read_parquet(macro_path)[config['data']['macro_features']]


def create_lookback_samples(df, feature_cols, lookback_window, target_horizon=5):
    """
    Create (X, Y) pairs where:
    X: [lookback_window, features] — input trajectory
    Y: [target_horizon, features] — future trajectory to predict
    """
    data = df[feature_cols].values
    samples = []
    
    for i in range(len(data) - lookback_window - target_horizon):
        x = data[i:i + lookback_window]  # Input: t=0 to T
        y = data[i + lookback_window:i + lookback_window + target_horizon]  # Target: T+1 to T+H
        timestamp = df.index[i + lookback_window]
        samples.append((x, y, timestamp))
    
    return samples


def build_dataset(config, split='train'):
    """Build dataset for all ETFs with macro conditioning."""
    etfs = config['data']['etf_universe']
    lookback = config['data']['lookback_window']
    
    all_samples = []
    
    for etf in etfs:
        df = fetch_etf_data(etf, config)
        macro = fetch_macro_data(config)
        
        # Align and merge
        df = df.join(macro, how='left')
        
        # Feature engineering
        df['returns_lag1'] = df['log_returns'].shift(1)
        df['vol_norm'] = df['vol'] / df['vol'].rolling(21).mean()
        
        feature_cols = ['log_returns', 'returns_lag1', 'vol_norm'] + config['data']['macro_features']
        df = df.dropna()
        
        # Time split
        if split == 'train':
            df = df[df.index < config['data']['train_end']]
        else:
            df = df[df.index >= config['data']['train_end']]
        
        samples = create_lookback_samples(df, feature_cols, lookback)
        
        for x, y, ts in samples:
            all_samples.append({
                'etf': etf,
                'X': x,  # [lookback, features]
                'Y': y,  # [horizon, features]
                'timestamp': ts
            })
    
    return all_samples


if __name__ == "__main__":
    config = load_config()
    train_data = build_dataset(config, 'train')
    print(f"Loaded {len(train_data)} training samples")
    print(f"Sample shape: X={train_data[0]['X'].shape}, Y={train_data[0]['Y'].shape}")
