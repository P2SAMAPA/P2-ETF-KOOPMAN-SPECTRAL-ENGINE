"""
Data loader for Koopman-Spectral engine.
Handles wide-format parquet: columns = ['Date', 'ETF1_Open', 'ETF1_High', ..., 'ETF2_Open', ...]
Source: HF dataset P2SAMAPA/p2-etf-deepm-data/data/master.parquet

Fixes applied:
  1. build_dataset_tensors() now fits a StandardScaler on training data
     and saves it alongside the model checkpoint so inference is consistent.
  2. load_scaler() / apply_scaler() helpers used in generate_signals.py.
  3. ETF-index map (etf_to_idx) returned for use with the embedding table.
  4. Replaced deprecated fillna(method='ffill') with .ffill() (already
     present in original) — kept as-is; added bfill fallback for leading NaNs.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import pickle
import requests
from io import BytesIO
from typing import Optional, List, Dict, Tuple
import re


# ── HFDataLoader ──────────────────────────────────────────────────────────────

class HFDataLoader:
    """
    HuggingFace dataset loader for p2-etf-deepm-data.
    Handles wide-format parquet with ETF-specific columns and macro indicators.
    """

    HF_DATASET_NAME = "P2SAMAPA/p2-etf-deepm-data"
    BASE_URL = f"https://huggingface.co/datasets/{HF_DATASET_NAME}/resolve/main/data"

    def __init__(self, use_local: bool = True,
                 local_path: str = "data/p2-etf-deepm-data"):
        self.use_local = use_local
        self.local_path = Path(local_path)
        self._master_df = None
        self._columns = None
        self._etf_columns_map = None
        self._date_col = None

    # ── Internal loaders ──────────────────────────────────────────────────

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
        """Parse wide-format columns to extract ETF names."""
        self._columns = df.columns.tolist()
        date_cols = [c for c in self._columns
                     if c.lower() in ['date', 'timestamp', 'time']]
        self._date_col = date_cols[0] if date_cols else self._columns[0]

        self._etf_columns_map = {}
        pattern = re.compile(
            r'^(.*?)(?:_Open|_High|_Low|_Close|_Volume|_AdjClose|_Returns|_LogReturns)$'
        )
        for col in self._columns:
            m = pattern.match(col)
            if m:
                etf = m.group(1)
                self._etf_columns_map.setdefault(etf, []).append(col)

        print(f"Detected ETFs: {list(self._etf_columns_map.keys())[:10]}...")
        print(f"Total ETFs: {len(self._etf_columns_map)}")

    # ── Public API ────────────────────────────────────────────────────────

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
            _ = self.load_master()
        return self._columns or []

    def get_all_etfs(self) -> List[str]:
        if self._etf_columns_map is None:
            _ = self.load_master()
        return list(self._etf_columns_map.keys()) if self._etf_columns_map else []

    def get_etf_data(self, symbol: str,
                     lookback: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Extract time series for a single ETF from wide-format dataset.
        Returns full history if lookback is None, else last <lookback> rows.
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

        col_mapping = {}
        for col in etf_cols:
            cl = col.lower()
            if '_open'   in cl: col_mapping['open']   = col
            elif '_high' in cl: col_mapping['high']   = col
            elif '_low'  in cl: col_mapping['low']    = col
            elif '_close' in cl: col_mapping['close'] = col
            elif '_volume' in cl: col_mapping['volume'] = col

        for std_name, orig_col in col_mapping.items():
            df[std_name] = master[orig_col].values

        if 'close' in df.columns:
            df['returns']     = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        df['symbol'] = symbol
        df = df.sort_values('date').dropna(subset=['log_returns']).reset_index(drop=True)

        if lookback is not None and len(df) > lookback:
            return df.iloc[-lookback:]
        return df

    def get_macro_data(self, macro_cols: List[str]) -> pd.DataFrame:
        """Extract macro indicator columns from master."""
        master = self.load_master()
        if master.empty:
            return pd.DataFrame()

        available = [c for c in macro_cols if c in master.columns]
        missing   = set(macro_cols) - set(available)
        if missing:
            print(f"Warning: Missing macro columns: {missing}")
        if not available:
            print("No macro columns found. Returning empty DataFrame.")
            return pd.DataFrame()

        df = pd.DataFrame()
        df['date'] = pd.to_datetime(master[self._date_col])
        for col in available:
            df[col] = master[col].values
        df = df.sort_values('date').reset_index(drop=True)
        # Forward-fill then backward-fill to handle leading NaNs
        for col in available:
            df[col] = df[col].ffill().bfill().fillna(0.0)
        return df


# ── Config loader ─────────────────────────────────────────────────────────────

def load_config(config_path: str = "config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── Scaler helpers ────────────────────────────────────────────────────────────

def save_scaler(scaler, path: str = "koopman_scaler.pkl"):
    with open(path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {path}")


def load_scaler(path: str = "koopman_scaler.pkl"):
    """Returns scaler or None if file not found."""
    p = Path(path)
    if not p.exists():
        print(f"Scaler not found at {path} — using no normalisation.")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def apply_scaler(X_np: np.ndarray, scaler) -> np.ndarray:
    """
    Apply a fitted scaler to a sequence array.
    X_np: [N, lookback, features]  or  [lookback, features]
    Scaler was fit on shape [N*lookback, features].
    """
    if scaler is None:
        return X_np
    orig_shape = X_np.shape
    flat = X_np.reshape(-1, orig_shape[-1])
    flat_scaled = scaler.transform(flat)
    return flat_scaled.reshape(orig_shape)


# ── Dataset builders ──────────────────────────────────────────────────────────

def build_dataset_tensors(
    config,
    split: str = 'train',
    scaler=None,
    fit_scaler: bool = False,
) -> Tuple[Optional[object], Optional[object], List[str], dict, Optional[object]]:
    """
    Build dataset as PyTorch tensors for training, with per-ETF macro alignment
    and optional StandardScaler normalisation.

    Returns:
        X_tensor:   [N, lookback, features]
        y_tensor:   [N, target_horizon]
        feat_names: list of feature names
        etf_to_idx: dict mapping ETF symbol -> integer index
        scaler:     fitted scaler (if fit_scaler=True) else the passed-in scaler
    """
    import torch
    from sklearn.preprocessing import StandardScaler

    lookback       = config['data']['lookback_window']
    target_horizon = config['data'].get('target_horizon', 5)
    etf_universe   = config['data']['etf_universe']
    macro_cols     = config['data']['macro_features']

    loader = HFDataLoader(
        use_local=True,
        local_path=config['data'].get('local_path', 'data/p2-etf-deepm-data'),
    )

    master = loader.load_master()
    if master.empty:
        raise ValueError("Could not load master.parquet")

    # Build ETF index map (for embedding table)
    etf_to_idx = {etf: i for i, etf in enumerate(etf_universe)}

    # Load macro data once — we will align per-ETF by date
    macro_df = loader.get_macro_data(macro_cols)
    if macro_df.empty:
        print("Warning: No macro data found. Using zeros for macro features.")
        macro_df = pd.DataFrame({'date': pd.to_datetime(master[loader._date_col])})
        for col in macro_cols:
            macro_df[col] = 0.0

    all_X     = []
    all_y     = []
    all_etf_idx = []

    for etf in etf_universe:
        df = loader.get_etf_data(etf, lookback=None)
        if df is None or len(df) < lookback + target_horizon:
            continue

        returns_col = 'log_returns' if 'log_returns' in df.columns else 'returns'
        returns = df[returns_col].values
        vol     = pd.Series(returns).rolling(5).std().values

        # Drop rows where vol is NaN (first 4 rows)
        valid   = ~np.isnan(vol)
        returns = returns[valid]
        vol     = vol[valid]
        dates   = df['date'].values[valid]

        if len(returns) < lookback + target_horizon:
            continue

        # FIX: align macro by date PER ETF (was a shared array in original)
        merged = pd.DataFrame({'date': pd.to_datetime(dates),
                                'returns': returns,
                                'vol': vol})
        merged = merged.merge(macro_df, on='date', how='left')
        for col in macro_cols:
            if col in merged.columns:
                merged[col] = merged[col].ffill().bfill().fillna(0.0)
            else:
                merged[col] = 0.0

        ret_arr   = merged['returns'].values
        vol_arr   = merged['vol'].values
        macro_arr = merged[macro_cols].values

        etf_i = etf_to_idx.get(etf, 0)

        for i in range(len(ret_arr) - lookback - target_horizon + 1):
            ret_w   = ret_arr[i:i + lookback].reshape(-1, 1)
            vol_w   = vol_arr[i:i + lookback].reshape(-1, 1)
            mac_w   = macro_arr[i:i + lookback, :]
            X_seq   = np.hstack([ret_w, vol_w, mac_w])  # [lookback, 2+M]
            y_seq   = ret_arr[i + lookback: i + lookback + target_horizon]

            if np.isnan(X_seq).any() or np.isnan(y_seq).any():
                continue

            all_X.append(X_seq)
            all_y.append(y_seq)
            all_etf_idx.append(etf_i)

    if len(all_X) == 0:
        return None, None, None, etf_to_idx, scaler

    split_idx = int(len(all_X) * 0.8)
    if split == 'train':
        X = np.array(all_X[:split_idx])
        y = np.array(all_y[:split_idx])
        idx_arr = np.array(all_etf_idx[:split_idx])
    else:
        X = np.array(all_X[split_idx:])
        y = np.array(all_y[split_idx:])
        idx_arr = np.array(all_etf_idx[split_idx:])

    # FIX: Normalise features
    if fit_scaler:
        scaler = StandardScaler()
        N, L, F = X.shape
        scaler.fit(X.reshape(-1, F))
        print("StandardScaler fitted on training data.")

    X = apply_scaler(X, scaler)

    import torch
    X_tensor   = torch.tensor(X,       dtype=torch.float32)
    y_tensor   = torch.tensor(y,       dtype=torch.float32)
    idx_tensor = torch.tensor(idx_arr, dtype=torch.long)

    feature_names = ['returns', 'volatility'] + macro_cols
    return X_tensor, y_tensor, idx_tensor, feature_names, etf_to_idx, scaler


# ── Legacy builder (kept for backward compat) ─────────────────────────────────

def build_dataset(config, split='train'):
    """Legacy list-of-dicts builder (unchanged logic, kept for compatibility)."""
    etfs           = config['data']['etf_universe']
    lookback       = config['data']['lookback_window']
    target_horizon = config['data'].get('target_horizon', 5)

    loader = HFDataLoader(
        use_local=True,
        local_path=config['data'].get('local_path', 'data/p2-etf-deepm-data'),
    )
    master = loader.load_master()
    if master.empty:
        raise ValueError("Could not load master.parquet")

    all_samples = []
    for etf in etfs:
        df = loader.get_etf_data(etf, lookback=None)
        if df is None or len(df) < lookback + target_horizon:
            continue

        returns_col = 'log_returns' if 'log_returns' in df.columns else 'returns'
        if returns_col not in df.columns:
            continue

        df['vol'] = df[returns_col].rolling(5).std()
        feature_df = df[[returns_col, 'vol']].dropna()
        if len(feature_df) < lookback + target_horizon:
            continue

        data = feature_df.values
        for i in range(len(data) - lookback - target_horizon):
            x = data[i:i + lookback]
            y = data[i + lookback:i + lookback + target_horizon, 0]
            if np.isnan(x).any() or np.isnan(y).any():
                continue
            all_samples.append({
                'etf': etf,
                'X': x, 'Y': y,
                'timestamp': df['date'].iloc[i + lookback],
                'features': [returns_col, 'vol'],
            })

    if len(all_samples) == 0:
        raise ValueError("No training samples generated")

    split_idx = int(len(all_samples) * 0.8)
    return all_samples[:split_idx] if split == 'train' else all_samples[split_idx:]


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        config = load_config()
        loader = HFDataLoader(
            use_local=True,
            local_path=config['data'].get('local_path', 'data/p2-etf-deepm-data'),
        )
        master = loader.load_master()
        if master.empty:
            print("ERROR: Could not load master.parquet")
        else:
            print(f"Loaded: {len(master)} rows, {len(loader.get_columns())} columns")
            print(f"ETFs detected: {len(loader.get_all_etfs())}")

        result = build_dataset_tensors(config, 'train', fit_scaler=True)
        X_train, y_train, idx_train, feat_names, etf_to_idx, scaler = result
        if X_train is not None:
            print(f"Train samples: {X_train.shape[0]}, features: {X_train.shape[-1]}")
            print(f"Feature names: {feat_names}")
            print(f"ETF index map: {etf_to_idx}")
        else:
            print("No tensor data generated.")
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
