"""
Data loader for Koopman-Spectral engine.
Handles wide-format parquet: columns = ['Date', 'ETF1_Open', 'ETF1_High', ..., 'ETF2_Open', ...]
Source: HF dataset P2SAMAPA/p2-etf-deepm-data/data/master.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import requests
from io import BytesIO
from typing import Optional, List, Dict, Tuple
import re


class HFDataLoader:
    """
    HuggingFace dataset loader for p2-etf-deepm-data.
    Handles wide-format parquet with ETF-specific columns and macro indicators.
    """
    
    HF_DATASET_NAME = "P2SAMAPA/p2-etf-deepm-data"
    BASE_URL = f"https://huggingface.co/datasets/{HF_DATASET_NAME}/resolve/main/data"
    
    def __init__(self, use_local: bool = True, local_path: str = "data/p2-etf-deepm-data"):
        self.use_local = use_local
        self.local_path = Path(local_path)
        self._master_df = None
        self._columns = None
        self._etf_columns_map = None
        self._date_col = None
        
    def _load_master_from_hf(self) -> pd.DataFrame:
        """Load master.parquet from HuggingFace Hub."""
        url = f"{self.BASE_URL}/master.parquet"
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            df = pd.read_parquet(BytesIO(response.content))
            self._process_columns(df)
            return df
        except Exception as e:
            print(f"Failed to load master.parquet from HF: {e}")
            return pd.DataFrame()
    
    def _load_master_local(self) -> pd.DataFrame:
        """Load from local path."""
        filepath = self.local_path / "master.parquet"
        if not filepath.exists():
            filepath = self.local_path / "data" / "master.parquet"
        if not filepath.exists():
            return pd.DataFrame()
        try:
            df = pd.read_parquet(filepath)
            self._process_columns(df)
            return df
        except Exception as e:
            print(f"Failed to load local master.parquet: {e}")
            return pd.DataFrame()
    
    def _process_columns(self, df: pd.DataFrame):
        """Parse wide-format columns to extract ETF names."""
        self._columns = df.columns.tolist()
        # Find date column
        date_cols = [c for c in self._columns if c.lower() in ['date', 'timestamp', 'time']]
        self._date_col = date_cols[0] if date_cols else self._columns[0]
        
        # Parse ETF names from column names like "ETF_Open", "ETF_High", etc.
        self._etf_columns_map = {}
        pattern = re.compile(r'^(.*?)(?:_Open|_High|_Low|_Close|_Volume|_AdjClose|_Returns|_LogReturns)$')
        for col in self._columns:
            match = pattern.match(col)
            if match:
                etf = match.group(1)
                if etf not in self._etf_columns_map:
                    self._etf_columns_map[etf] = []
                self._etf_columns_map[etf].append(col)
        print(f"Detected ETFs: {list(self._etf_columns_map.keys())[:10]}...")
        print(f"Total ETFs: {len(self._etf_columns_map)}")
    
    def load_master(self) -> pd.DataFrame:
        """Load consolidated master dataset."""
        if self._master_df is not None:
            return self._master_df
        if self.use_local:
            df = self._load_master_local()
            if not df.empty:
                self._master_df = df
                return df
        df = self._load_master_from_hf()
        self._master_df = df
        return df
    
    def get_columns(self) -> List[str]:
        """Get available columns."""
        if self._columns is None:
            _ = self.load_master()
        return self._columns or []
    
    def get_all_etfs(self) -> List[str]:
        """Get list of all available ETFs."""
        if self._etf_columns_map is None:
            _ = self.load_master()
        return list(self._etf_columns_map.keys()) if self._etf_columns_map else []
    
    def get_etf_data(self, symbol: str, lookback: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Extract time series for a single ETF from wide-format dataset.
        If lookback is None, returns full history. Otherwise returns last lookback days.
        """
        master = self.load_master()
        if master.empty:
            return None
        if symbol not in self._etf_columns_map:
            print(f"ETF {symbol} not found in columns")
            return None
        
        etf_cols = self._etf_columns_map[symbol]
        df = pd.DataFrame()
        df['date'] = pd.to_datetime(master[self._date_col])
        
        # Map to standard OHLCV names
        col_mapping = {}
        for col in etf_cols:
            col_lower = col.lower()
            if '_open' in col_lower:
                col_mapping['open'] = col
            elif '_high' in col_lower:
                col_mapping['high'] = col
            elif '_low' in col_lower:
                col_mapping['low'] = col
            elif '_close' in col_lower:
                col_mapping['close'] = col
            elif '_volume' in col_lower:
                col_mapping['volume'] = col
        
        for std_name, orig_col in col_mapping.items():
            df[std_name] = master[orig_col]
        
        # Calculate returns if not present
        if 'close' in df.columns:
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        df['symbol'] = symbol
        df = df.sort_values('date').dropna().reset_index(drop=True)
        
        if lookback is not None and len(df) > lookback:
            return df.iloc[-lookback:]
        return df
    
    def get_macro_data(self, macro_cols: List[str]) -> pd.DataFrame:
        """Extract macro indicator columns from master."""
        master = self.load_master()
        if master.empty:
            return pd.DataFrame()
        available = [c for c in macro_cols if c in master.columns]
        missing = set(macro_cols) - set(available)
        if missing:
            print(f"Warning: Missing macro columns: {missing}")
        if not available:
            print("No macro columns found. Returning empty DataFrame.")
            return pd.DataFrame()
        df = pd.DataFrame()
        df['date'] = pd.to_datetime(master[self._date_col])
        for col in available:
            df[col] = master[col]
        df = df.sort_values('date').dropna().reset_index(drop=True)
        return df


def load_config(config_path: str = "config.yaml"):
    """Load engine configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_dataset(config, split='train'):
    """
    Build dataset from wide-format master.parquet (legacy format).
    Returns list of dicts.
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
        raise ValueError("Could not load master.parquet")
    
    all_samples = []
    
    for etf in etfs:
        df = loader.get_etf_data(etf, lookback=None)  # full history
        if df is None or len(df) < lookback + target_horizon:
            continue
        
        # Use log_returns if available, else returns
        if 'log_returns' in df.columns:
            returns_col = 'log_returns'
        elif 'returns' in df.columns:
            returns_col = 'returns'
        else:
            continue
        
        # Create features
        features = [returns_col]
        
        # Add volatility if we can calculate
        if returns_col in df.columns:
            df['vol'] = df[returns_col].rolling(5).std()
            features.append('vol')
        
        # Get feature matrix
        feature_df = df[features].dropna()
        
        if len(feature_df) < lookback + target_horizon:
            continue
        
        data = feature_df.values
        
        # Create samples
        for i in range(len(data) - lookback - target_horizon):
            x = data[i:i + lookback]
            y = data[i + lookback:i + lookback + target_horizon, 0]  # only returns target
            if np.isnan(x).any() or np.isnan(y).any():
                continue
            
            all_samples.append({
                'etf': etf,
                'X': x,
                'Y': y,
                'timestamp': df['date'].iloc[i + lookback],
                'features': features
            })
    
    if len(all_samples) == 0:
        raise ValueError("No training samples generated")
    
    # Time split
    split_idx = int(len(all_samples) * 0.8)
    if split == 'train':
        return all_samples[:split_idx]
    else:
        return all_samples[split_idx:]


def build_dataset_tensors(config, split='train') -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
    """
    Build dataset as PyTorch tensors for training using all macro features.
    Returns:
        X_tensor: [N, lookback, features] (returns, vol, macro1, macro2, ...)
        y_tensor: [N, target_horizon] (future returns)
        feature_names: list
    """
    import torch
    
    lookback = config['data']['lookback_window']
    target_horizon = config['data'].get('target_horizon', 5)
    etf_universe = config['data']['etf_universe']
    macro_cols = config['data']['macro_features']
    
    loader = HFDataLoader(
        use_local=True,
        local_path=config['data'].get('local_path', 'data/p2-etf-deepm-data')
    )
    master = loader.load_master()
    if master.empty:
        raise ValueError("Could not load master.parquet")
    
    # Load macro data
    macro_df = loader.get_macro_data(macro_cols)
    if macro_df.empty:
        print("Warning: No macro data found. Using zeros for macro features.")
        macro_df = pd.DataFrame({'date': master[loader._date_col]})
        for col in macro_cols:
            macro_df[col] = 0.0
        macro_df = macro_df.dropna().reset_index(drop=True)
    
    all_X = []
    all_y = []
    
    for etf in etf_universe:
        df = loader.get_etf_data(etf, lookback=None)  # full history
        if df is None or len(df) < lookback + target_horizon:
            continue
        
        if 'log_returns' in df.columns:
            returns = df['log_returns'].values
        elif 'returns' in df.columns:
            returns = df['returns'].values
        else:
            continue
        
        # Compute rolling volatility
        vol = pd.Series(returns).rolling(5).std().values
        valid = ~np.isnan(vol)
        returns = returns[valid]
        vol = vol[valid]
        if len(returns) < lookback + target_horizon:
            continue
        
        # Align macro data by date
        etf_dates = df['date'].iloc[valid].reset_index(drop=True)
        merged = pd.DataFrame({'date': etf_dates, 'returns': returns, 'vol': vol})
        merged = merged.merge(macro_df, on='date', how='left')
        for col in macro_cols:
            merged[col] = merged[col].fillna(method='ffill').fillna(0.0)
        
        returns = merged['returns'].values
        vol = merged['vol'].values
        macro_vals = merged[macro_cols].values
        
        for i in range(len(returns) - lookback - target_horizon + 1):
            ret_window = returns[i:i+lookback].reshape(-1, 1)
            vol_window = vol[i:i+lookback].reshape(-1, 1)
            macro_window = macro_vals[i:i+lookback, :]
            X_seq = np.hstack([ret_window, vol_window, macro_window])
            y_seq = returns[i+lookback:i+lookback+target_horizon]
            all_X.append(X_seq)
            all_y.append(y_seq)
    
    if len(all_X) == 0:
        return None, None, None
    
    split_idx = int(len(all_X) * 0.8)
    if split == 'train':
        X = all_X[:split_idx]
        y = all_y[:split_idx]
    else:
        X = all_X[split_idx:]
        y = all_y[split_idx:]
    
    X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y), dtype=torch.float32)
    feature_names = ['returns', 'volatility'] + macro_cols
    return X_tensor, y_tensor, feature_names


if __name__ == "__main__":
    try:
        config = load_config()
        print(f"Loading data from: {config['data'].get('local_path', 'data/p2-etf-deepm-data')}")
        
        loader = HFDataLoader(
            use_local=True,
            local_path=config['data'].get('local_path', 'data/p2-etf-deepm-data')
        )
        master = loader.load_master()
        
        if master.empty:
            print("ERROR: Could not load master.parquet")
        else:
            print(f"Loaded: {len(master)} rows, {len(loader.get_columns())} columns")
            print(f"ETFs detected: {len(loader.get_all_etfs())}")
            print(f"Sample ETFs: {loader.get_all_etfs()[:10]}")
            
            # Test single ETF full data
            test_etf = config['data']['etf_universe'][0]
            test_data = loader.get_etf_data(test_etf, lookback=None)
            if test_data is not None:
                print(f"\nFull data for {test_etf}: {len(test_data)} rows")
                print(test_data[['date', 'close', 'log_returns']].head())
            else:
                print(f"\nWARNING: No data for {test_etf}")
                print(f"Available: {loader.get_all_etfs()[:10]}...")
            
            # Test tensor builder with macros
            X_train, y_train, feat_names = build_dataset_tensors(config, 'train')
            if X_train is not None:
                print(f"\nTrain samples: {X_train.shape[0]}, features: {X_train.shape[-1]} -> {feat_names}")
            else:
                print("\nNo tensor data generated.")
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
