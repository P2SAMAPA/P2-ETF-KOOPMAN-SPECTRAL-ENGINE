"""
Data loader for Koopman-Spectral engine.
Pulls from P2 shared data hub: HF dataset P2SAMAPA/p2-etf-deepm-data
Consolidated format: data/master.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import requests
from io import BytesIO
from typing import Optional, List, Dict


class HFDataLoader:
    """
    HuggingFace dataset loader for p2-etf-deepm-data.
    Loads from data/master.parquet (consolidated format).
    """
    
    HF_DATASET_NAME = "P2SAMAPA/p2-etf-deepm-data"
    BASE_URL = f"https://huggingface.co/datasets/{HF_DATASET_NAME}/resolve/main/data"
    
    def __init__(self, use_local: bool = True, local_path: str = "data/p2-etf-deepm-data"):
        self.use_local = use_local
        self.local_path = Path(local_path)
        self._master_df = None
        self._columns = None
        
    def _load_master_from_hf(self) -> pd.DataFrame:
        """Load master.parquet from HuggingFace Hub."""
        url = f"{self.BASE_URL}/master.parquet"
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            df = pd.read_parquet(BytesIO(response.content))
            
            # Standardize date column
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            elif 'timestamp' in df.columns:
                df['date'] = pd.to_datetime(df['timestamp'])
            
            self._columns = df.columns.tolist()
            return df
            
        except Exception as e:
            print(f"Failed to load master.parquet from HF: {e}")
            return pd.DataFrame()
    
    def _load_master_local(self) -> pd.DataFrame:
        """Load from local path."""
        # Try data/master.parquet first (HF structure)
        filepath = self.local_path / "master.parquet"
        
        if not filepath.exists():
            # Try with data/ subdirectory
            filepath = self.local_path / "data" / "master.parquet"
        
        if not filepath.exists():
            return pd.DataFrame()
        
        try:
            df = pd.read_parquet(filepath)
            
            # Standardize date column
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            elif 'timestamp' in df.columns:
                df['date'] = pd.to_datetime(df['timestamp'])
            
            self._columns = df.columns.tolist()
            return df
            
        except Exception as e:
            print(f"Failed to load local master.parquet: {e}")
            return pd.DataFrame()
    
    def load_master(self) -> pd.DataFrame:
        """Load consolidated master dataset with all ETFs and features."""
        if self._master_df is not None:
            return self._master_df
        
        if self.use_local:
            df = self._load_master_local()
            if not df.empty:
                self._master_df = df
                return df
        
        # Fallback to HF Hub
        df = self._load_master_from_hf()
        self._master_df = df
        return df
    
    def get_columns(self) -> List[str]:
        """Get available columns."""
        if self._columns is None:
            _ = self.load_master()
        return self._columns or []
    
    def get_etf_data(self, symbol: str, lookback: int = 63) -> Optional[pd.DataFrame]:
        """Extract single ETF time series from master dataset."""
        master = self.load_master()
        if master.empty:
            return None
        
        # Auto-detect ETF column
        etf_col = None
        for candidate in ['symbol', 'etf', 'ticker', 'asset', 'name']:
            if candidate in master.columns:
                etf_col = candidate
                break
        
        if etf_col is None:
            print("Cannot find ETF symbol column in master.parquet")
            return None
        
        # Filter by symbol
        df = master[master[etf_col] == symbol].copy()
        if df.empty:
            return None
        
        # Sort by date
        date_col = 'date' if 'date' in df.columns else None
        if date_col is None:
            for candidate in ['timestamp', 'date', 'time', 'trading_date']:
                if candidate in df.columns:
                    date_col = candidate
                    break
        
        if date_col:
            df = df.sort_values(date_col)
        else:
            df = df.sort_values(df.columns[0])
        
        # Get last lookback rows
        if len(df) < lookback:
            return None
            
        return df.iloc[-lookback:]
    
    def get_all_etfs(self) -> List[str]:
        """Get list of all available ETFs in dataset."""
        master = self.load_master()
        if master.empty:
            return []
        
        # Find ETF column
        etf_col = None
        for candidate in ['symbol', 'etf', 'ticker', 'asset']:
            if candidate in master.columns:
                etf_col = candidate
                break
        
        if etf_col is None:
            return []
        
        return master[etf_col].unique().tolist()


# Standalone function (not importing from self)
def load_config(config_path: str = "config.yaml"):
    """Load engine configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_dataset(config, split='train'):
    """
    Build dataset for all ETFs from master.parquet.
    Returns list of samples with X (lookback) and Y (horizon).
    """
    etfs = config['data']['etf_universe']
    lookback = config['data']['lookback_window']
    target_horizon = config['data'].get('target_horizon', 5)
    
    loader = HFDataLoader(
        use_local=True, 
        local_path=config['data'].get('local_path', 'data/p2-etf-deepm-data')
    )
    master = loader.load_master()
    
    if master.empty:
        raise ValueError("Could not load master.parquet from HF or local path")
    
    available_cols = loader.get_columns()
    
    # Detect columns
    etf_col = None
    for candidate in ['symbol', 'etf', 'ticker', 'asset']:
        if candidate in available_cols:
            etf_col = candidate
            break
    
    date_col = None
    for candidate in ['date', 'timestamp', 'time']:
        if candidate in available_cols:
            date_col = candidate
            break
    
    # Find returns column
    returns_col = None
    for candidate in ['log_returns', 'returns', 'ret', 'log_ret', 'daily_return']:
        matches = [c for c in available_cols if candidate.lower() in c.lower()]
        if matches:
            returns_col = matches[0]
            break
    
    if returns_col is None:
        raise ValueError("Could not find returns column in master.parquet")
    
    # Find macro columns
    macro_cols = [c for c in available_cols if any(m.lower() in c.lower() 
                  for m in config['data']['macro_features'])]
    
    all_samples = []
    
    for etf in etfs:
        etf_data = master[master[etf_col] == etf].copy()
        
        if len(etf_data) < lookback + target_horizon + 5:
            continue
        
        # Sort by date
        if date_col:
            etf_data = etf_data.sort_values(date_col)
        
        # Get feature columns
        feature_cols = [returns_col]
        
        # Add volatility if available
        vol_matches = [c for c in available_cols if 'vol' in c.lower()]
        if vol_matches:
            feature_cols.append(vol_matches[0])
        
        # Add macro
        feature_cols.extend(macro_cols)
        
        # Check which columns actually exist
        existing_cols = [c for c in feature_cols if c in etf_data.columns]
        
        if len(existing_cols) < 2:  # Need at least returns + one other
            continue
        
        # Create samples
        data = etf_data[existing_cols].values
        
        for i in range(len(data) - lookback - target_horizon):
            x = data[i:i + lookback]
            y = data[i + lookback:i + lookback + target_horizon]
            
            # Check for NaNs
            if np.isnan(x).any() or np.isnan(y).any():
                continue
            
            timestamp = etf_data[date_col].iloc[i + lookback] if date_col else i
            
            all_samples.append({
                'etf': etf,
                'X': x,
                'Y': y,
                'timestamp': timestamp,
                'features': existing_cols
            })
    
    # Time split
    if split == 'train':
        # Use first 80% for training
        split_idx = int(len(all_samples) * 0.8)
        return all_samples[:split_idx]
    else:
        split_idx = int(len(all_samples) * 0.8)
        return all_samples[split_idx:]


if __name__ == "__main__":
    # Test loading
    try:
        config = load_config()
        print(f"Loading data from HF: {config['data']['hf_dataset']}")
        
        loader = HFDataLoader(
            use_local=True,
            local_path=config['data'].get('local_path', 'data/p2-etf-deepm-data')
        )
        master = loader.load_master()
        
        if master.empty:
            print("ERROR: Could not load master.parquet")
            print("Checked:")
            print(f"  - {loader.local_path}/master.parquet")
            print(f"  - {loader.local_path}/data/master.parquet")
            print(f"  - HF Hub: {loader.BASE_URL}/master.parquet")
        else:
            print(f"Loaded master.parquet: {len(master)} rows")
            print(f"Columns: {loader.get_columns()[:10]}...")
            print(f"ETFs found: {len(loader.get_all_etfs())}")
            
            # Test single ETF
            test_etf = config['data']['etf_universe'][0]
            test_data = loader.get_etf_data(test_etf, lookback=63)
            if test_data is not None:
                print(f"Sample data for {test_etf}: {len(test_data)} rows")
            else:
                print(f"WARNING: No data found for {test_etf}")
            
            # Build full dataset
            try:
                train_data = build_dataset(config, 'train')
                print(f"Built training dataset: {len(train_data)} samples")
            except Exception as e:
                print(f"Error building dataset: {e}")
                
    except FileNotFoundError:
        print("config.yaml not found, using defaults")
        loader = HFDataLoader(use_local=True)
        master = loader.load_master()
        print(f"Rows: {len(master)}, ETFs: {len(loader.get_all_etfs()) if not master.empty else 0}")
