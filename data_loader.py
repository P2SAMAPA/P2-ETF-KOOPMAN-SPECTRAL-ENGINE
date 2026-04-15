"""
Data loader for Koopman-Spectral engine.
Handles wide-format parquet + macro indicators.
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
        self._columns = df.columns.tolist()
        date_cols = [c for c in self._columns if c.lower() in ['date', 'timestamp', 'time']]
        self._date_col = date_cols[0] if date_cols else self._columns[0]
        
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
        if self._columns is None:
            _ = self.load_master()
        return self._columns or []
    
    def get_all_etfs(self) -> List[str]:
        if self._etf_columns_map is None:
            _ = self.load_master()
        return list(self._etf_columns_map.keys()) if self._etf_columns_map else []
    
    def get_etf_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Return full time series for one ETF (date, close, returns, log_returns)."""
        master = self.load_master()
        if master.empty:
            return None
        if symbol not in self._etf_columns_map:
            print(f"ETF {symbol} not found in columns")
            return None
        
        etf_cols = self._etf_columns_map[symbol]
        df = pd.DataFrame()
        df['date'] = pd.to_datetime(master[self._date_col])
        
        col_mapping = {}
        for col in etf_cols:
            col_lower = col.lower()
            if '_close' in col_lower:
                col_mapping['close'] = col
        if 'close' not in col_mapping:
            return None
        
        df['close'] = master[col_mapping['close']]
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['symbol'] = symbol
        df = df.sort_values('date').dropna().reset_index(drop=True)
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
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_dataset_tensors(config, split='train') -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
    """
    Build dataset with ETF returns + volatility + all macro features.
    Returns:
        X: [N, lookback, features] (features = returns, vol, macro1, macro2, ...)
        y: [N, target_horizon] (future returns)
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
        # Create dummy macro with zeros (same date range as master)
        macro_df = pd.DataFrame({'date': master[loader._date_col]})
        for col in macro_cols:
            macro_df[col] = 0.0
        macro_df = macro_df.dropna().reset_index(drop=True)
    
    all_X = []
    all_y = []
    
    for etf in etf_universe:
        df = loader.get_etf_data(etf)
        if df is None or len(df) < lookback + target_horizon:
            continue
        
        # Use log returns
        returns = df['log_returns'].values
        # Compute volatility (5-day rolling)
        vol = pd.Series(returns).rolling(5).std().values
        # Remove initial NaNs from vol
        valid = ~np.isnan(vol)
        returns = returns[valid]
        vol = vol[valid]
        if len(returns) < lookback + target_horizon:
            continue
        
        # Align macro data by date
        etf_dates = df['date'].iloc[valid].reset_index(drop=True)
        merged = pd.DataFrame({'date': etf_dates, 'returns': returns, 'vol': vol})
        merged = merged.merge(macro_df, on='date', how='left')
        # Forward fill missing macro values
        for col in macro_cols:
            merged[col] = merged[col].fillna(method='ffill').fillna(0.0)
        
        returns = merged['returns'].values
        vol = merged['vol'].values
        macro_vals = merged[macro_cols].values  # shape [T, M]
        
        # Create sliding windows
        for i in range(len(returns) - lookback - target_horizon + 1):
            ret_window = returns[i:i+lookback].reshape(-1, 1)
            vol_window = vol[i:i+lookback].reshape(-1, 1)
            macro_window = macro_vals[i:i+lookback, :]  # [lookback, M]
            X_seq = np.hstack([ret_window, vol_window, macro_window])  # [lookback, 2+M]
            y_seq = returns[i+lookback:i+lookback+target_horizon]
            all_X.append(X_seq)
            all_y.append(y_seq)
    
    if len(all_X) == 0:
        return None, None, None
    
    # Split chronologically
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
    config = load_config()
    print("Testing data loader with all macro features...")
    X_train, y_train, feat_names = build_dataset_tensors(config, 'train')
    if X_train is not None:
        print(f"Train samples: {X_train.shape[0]}, features: {X_train.shape[-1]} -> {feat_names}")
        print(f"Example X[0, -1, :]: {X_train[0, -1, :]}")
    else:
        print("No data generated.")
