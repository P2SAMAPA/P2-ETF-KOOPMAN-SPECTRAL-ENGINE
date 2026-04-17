"""
Data loader for Koopman-Spectral engine.
Source: HF dataset P2SAMAPA/p2-etf-deepm-data/data/master.parquet

Fixes vs original:
  1. build_dataset_tensors() aligns macro data per-ETF by date
     (was a shared pre-computed block — the main cause of identical predictions).
  2. StandardScaler fit on training data, saved/loaded alongside checkpoint.
  3. Returns etf_to_idx and idx_tensor for ETF embedding table.
  4. SPY excluded from training universe (benchmark only).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import pickle
import requests
from io import BytesIO
from typing import Optional, List, Tuple
import re


# ---------------------------------------------------------------------------
# HFDataLoader
# ---------------------------------------------------------------------------

class HFDataLoader:

    HF_DATASET_NAME = "P2SAMAPA/p2-etf-deepm-data"
    BASE_URL = f"https://huggingface.co/datasets/{HF_DATASET_NAME}/resolve/main/data"

    def __init__(self, use_local: bool = True,
                 local_path: str = "data/p2-etf-deepm-data"):
        self.use_local   = use_local
        self.local_path  = Path(local_path)
        self._master_df  = None
        self._columns    = None
        self._etf_columns_map = None
        self._date_col   = None

    # ---- internal loaders ------------------------------------------------

    def _load_master_from_hf(self) -> pd.DataFrame:
        url = f"{self.BASE_URL}/master.parquet"
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            df = pd.read_parquet(BytesIO(r.content))
            self._process_columns(df)
            return df
        except Exception as e:
            print(f"Failed to load master.parquet from HF: {e}")
            return pd.DataFrame()

    def _load_master_local(self) -> pd.DataFrame:
        for fp in [self.local_path / "master.parquet",
                   self.local_path / "data" / "master.parquet"]:
            if fp.exists():
                try:
                    df = pd.read_parquet(fp)
                    self._process_columns(df)
                    return df
                except Exception as e:
                    print(f"Failed to load {fp}: {e}")
        return pd.DataFrame()

    def _process_columns(self, df: pd.DataFrame):
        self._columns = df.columns.tolist()
        date_cols = [c for c in self._columns if c.lower() in ('date', 'timestamp', 'time')]
        self._date_col = date_cols[0] if date_cols else self._columns[0]
        self._etf_columns_map = {}
        pat = re.compile(r'^(.*?)(?:_Open|_High|_Low|_Close|_Volume|_AdjClose|_Returns|_LogReturns)$')
        for col in self._columns:
            m = pat.match(col)
            if m:
                self._etf_columns_map.setdefault(m.group(1), []).append(col)
        print(f"Detected {len(self._etf_columns_map)} ETFs")

    # ---- public API -------------------------------------------------------

    def load_master(self) -> pd.DataFrame:
        if self._master_df is not None:
            return self._master_df
        df = self._load_master_local() if self.use_local else pd.DataFrame()
        if df.empty:
            df = self._load_master_from_hf()
        self._master_df = df
        return df

    def get_columns(self) -> List[str]:
        if self._columns is None:
            self.load_master()
        return self._columns or []

    def get_all_etfs(self) -> List[str]:
        if self._etf_columns_map is None:
            self.load_master()
        return list(self._etf_columns_map.keys()) if self._etf_columns_map else []

    def get_etf_data(self, symbol: str,
                     lookback: Optional[int] = None) -> Optional[pd.DataFrame]:
        master = self.load_master()
        if master.empty or symbol not in (self._etf_columns_map or {}):
            return None

        etf_cols = self._etf_columns_map[symbol]
        df = pd.DataFrame({'date': pd.to_datetime(master[self._date_col])})

        col_map = {}
        for col in etf_cols:
            cl = col.lower()
            if   '_open'   in cl: col_map['open']   = col
            elif '_high'   in cl: col_map['high']   = col
            elif '_low'    in cl: col_map['low']    = col
            elif '_close'  in cl: col_map['close']  = col
            elif '_volume' in cl: col_map['volume'] = col

        for name, orig in col_map.items():
            df[name] = master[orig].values

        if 'close' in df.columns:
            df['returns']     = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        df['symbol'] = symbol
        df = df.sort_values('date').dropna(subset=['log_returns']).reset_index(drop=True)

        if lookback and len(df) > lookback:
            return df.iloc[-lookback:]
        return df

    def get_macro_data(self, macro_cols: List[str]) -> pd.DataFrame:
        master = self.load_master()
        if master.empty:
            return pd.DataFrame()
        available = [c for c in macro_cols if c in master.columns]
        missing   = set(macro_cols) - set(available)
        if missing:
            print(f"Warning: missing macro columns: {missing}")
        if not available:
            return pd.DataFrame()

        df = pd.DataFrame({'date': pd.to_datetime(master[self._date_col])})
        for col in available:
            df[col] = master[col].values
        df = df.sort_values('date').reset_index(drop=True)
        for col in available:
            df[col] = df[col].ffill().bfill().fillna(0.0)
        return df


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Scaler helpers
# ---------------------------------------------------------------------------

def save_scaler(scaler, path: str = "koopman_scaler.pkl"):
    with open(path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved → {path}")


def load_scaler(path: str = "koopman_scaler.pkl"):
    p = Path(path)
    if not p.exists():
        print(f"No scaler found at {path} — skipping normalisation.")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def apply_scaler(X_np: np.ndarray, scaler) -> np.ndarray:
    """X_np: [..., lookback, features] — scaler was fit on [N*lookback, features]."""
    if scaler is None:
        return X_np
    shape = X_np.shape
    flat  = X_np.reshape(-1, shape[-1])
    return scaler.transform(flat).reshape(shape)


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset_tensors(config, split: str = 'train',
                          scaler=None, fit_scaler: bool = False):
    """
    Returns: X_tensor, y_tensor, idx_tensor, feature_names, etf_to_idx, scaler

    Key fix: macro data is now merged with each ETF's own date index
    instead of using a single shared macro block for all tickers.
    """
    import torch
    from sklearn.preprocessing import StandardScaler as SK_StandardScaler

    lookback       = config['data']['lookback_window']
    target_horizon = config['data'].get('target_horizon', 5)
    etf_universe   = config['data']['etf_universe']   # SPY already removed in config
    macro_cols     = config['data']['macro_features']

    loader = HFDataLoader(
        use_local=True,
        local_path=config['data'].get('local_path', 'data/p2-etf-deepm-data'),
    )
    master = loader.load_master()
    if master.empty:
        raise ValueError("Could not load master.parquet")

    etf_to_idx = {etf: i for i, etf in enumerate(etf_universe)}

    # Load full macro series — we merge per-ETF by date below
    macro_df = loader.get_macro_data(macro_cols)
    if macro_df.empty:
        macro_df = pd.DataFrame({'date': pd.to_datetime(master[loader._date_col])})
        for col in macro_cols:
            macro_df[col] = 0.0

    all_X, all_y, all_idx = [], [], []

    for etf in etf_universe:
        df = loader.get_etf_data(etf, lookback=None)
        if df is None or len(df) < lookback + target_horizon:
            continue

        ret_col = 'log_returns' if 'log_returns' in df.columns else 'returns'
        returns = df[ret_col].values
        vol     = pd.Series(returns).rolling(5).std().values
        valid   = ~np.isnan(vol)
        returns, vol = returns[valid], vol[valid]
        dates   = df['date'].values[valid]

        if len(returns) < lookback + target_horizon:
            continue

        # FIX: align macro by this ETF's own dates
        merged = pd.DataFrame({'date': pd.to_datetime(dates),
                                'returns': returns, 'vol': vol})
        merged = merged.merge(macro_df, on='date', how='left')
        for col in macro_cols:
            if col in merged.columns:
                merged[col] = merged[col].ffill().bfill().fillna(0.0)
            else:
                merged[col] = 0.0

        ret_arr   = merged['returns'].values
        vol_arr   = merged['vol'].values
        mac_arr   = merged[macro_cols].values
        etf_i     = etf_to_idx[etf]

        for i in range(len(ret_arr) - lookback - target_horizon + 1):
            X_seq = np.hstack([ret_arr[i:i+lookback].reshape(-1,1),
                               vol_arr[i:i+lookback].reshape(-1,1),
                               mac_arr[i:i+lookback]])
            y_seq = ret_arr[i+lookback: i+lookback+target_horizon]
            if np.isnan(X_seq).any() or np.isnan(y_seq).any():
                continue
            all_X.append(X_seq)
            all_y.append(y_seq)
            all_idx.append(etf_i)

    if not all_X:
        return None, None, None, None, etf_to_idx, scaler

    split_idx = int(len(all_X) * 0.8)
    sl = slice(None, split_idx) if split == 'train' else slice(split_idx, None)
    X = np.array(all_X[sl])
    y = np.array(all_y[sl])
    idx_arr = np.array(all_idx[sl])

    # FIX: normalise features
    if fit_scaler:
        scaler = SK_StandardScaler()
        N, L, F = X.shape
        scaler.fit(X.reshape(-1, F))
        print("StandardScaler fitted on training data.")

    X = apply_scaler(X, scaler)

    X_tensor   = torch.tensor(X,       dtype=torch.float32)
    y_tensor   = torch.tensor(y,       dtype=torch.float32)
    idx_tensor = torch.tensor(idx_arr, dtype=torch.long)

    feat_names = ['returns', 'volatility'] + macro_cols
    return X_tensor, y_tensor, idx_tensor, feat_names, etf_to_idx, scaler


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        config = load_config()
        result = build_dataset_tensors(config, 'train', fit_scaler=True)
        X, y, idx, feat_names, etf_to_idx, scaler = result
        if X is not None:
            print(f"Train samples: {X.shape[0]}, features: {X.shape[-1]}")
            print(f"ETF universe ({len(etf_to_idx)}): {list(etf_to_idx.keys())}")
        else:
            print("No data generated.")
    except Exception as e:
        import traceback; traceback.print_exc()
