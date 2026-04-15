"""
Signal generation using trained Koopman model with all macro features.
"""

import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from data_loader import HFDataLoader, load_config
from koopman_model import KoopmanSpectral


def load_trained_model(config, model_path: str = "koopman_spectral_best.pt") -> Optional[KoopmanSpectral]:
    if not Path(model_path).exists():
        print(f"Model checkpoint not found: {model_path}")
        return None
    checkpoint = torch.load(model_path, map_location='cpu')
    # Ensure input_dim is set in config (should be from training)
    num_macro = len(config['data']['macro_features'])
    config['model']['input_dim'] = 2 + num_macro
    model = KoopmanSpectral(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def compute_predictability_index(eigenvalues: torch.Tensor) -> float:
    mags = torch.abs(eigenvalues)
    log_mags = torch.log(mags + 1e-8)
    variance = torch.var(log_mags)
    return float(1.0 / (1.0 + variance))


def classify_regime(eigenvalues: torch.Tensor) -> str:
    mags = torch.abs(eigenvalues)
    growth_ratio = (mags > 1.05).float().mean().item()
    decay_ratio = (mags < 0.95).float().mean().item()
    if growth_ratio > 0.2:
        return "expansion"
    elif decay_ratio > 0.3:
        return "contraction"
    else:
        return "oscillatory"


def get_latest_macro_values(loader: HFDataLoader, macro_cols: List[str]) -> Dict[str, float]:
    """Retrieve most recent macro indicator values."""
    macro_df = loader.get_macro_data(macro_cols)
    if macro_df.empty:
        return {col: 0.0 for col in macro_cols}
    latest = macro_df.iloc[-1]
    return {col: latest[col] for col in macro_cols if col in latest}


def generate_signals(config) -> Dict:
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
    
    model = load_trained_model(config)
    if model is None:
        raise RuntimeError("Trained model not found.")
    
    # Get latest macro values (for current day)
    current_macro = get_latest_macro_values(loader, macro_cols)
    
    predictions = []
    
    for etf in etf_universe:
        df = loader.get_etf_data(etf, lookback=None)  # full history
        if df is None or len(df) < lookback + 1:
            continue
        
        if 'log_returns' in df.columns:
            returns = df['log_returns'].values
        elif 'returns' in df.columns:
            returns = df['returns'].values
        else:
            continue
        
        vol = pd.Series(returns).rolling(5).std().values
        valid = ~(np.isnan(returns) | np.isnan(vol))
        returns = returns[valid]
        vol = vol[valid]
        if len(returns) < lookback:
            continue
        
        # Use most recent lookback window for returns and vol
        ret_window = returns[-lookback:].reshape(-1, 1)
        vol_window = vol[-lookback:].reshape(-1, 1)
        
        # For macro, use current values repeated across lookback
        macro_history = np.tile([current_macro.get(c, 0.0) for c in macro_cols], (lookback, 1))
        
        X_seq = np.hstack([ret_window, vol_window, macro_history])  # [lookback, 2+M]
        X_tensor = torch.tensor(X_seq, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            last_obs = X_tensor[0, -1, :]  # [features]
            z = model.encoder(last_obs.unsqueeze(0))  # FIXED: use model.encoder
            pred_return = model(z).item()
        
        pred_return_bps = pred_return * 10000  # log return to bps
        
        # Global predictability from model eigenvalues
        with torch.no_grad():
            eigs = torch.linalg.eigvals(model.K)
            predictability = compute_predictability_index(eigs)
            regime = classify_regime(eigs)
        
        predictions.append({
            'etf': etf,
            'predicted_1d_return_bps': pred_return_bps,
            'predicted_1d_return': pred_return_bps,
            'predictability_index': predictability,
            'koopman_regime': regime,
            'is_predictable': predictability >= config['signals']['predictability_threshold'],
            'data_quality': 1.0
        })
    
    # Filter and sort
    threshold = config['signals']['predictability_threshold']
    filtered = [p for p in predictions if p['predictability_index'] >= threshold]
    filtered.sort(key=lambda x: x['predicted_1d_return_bps'], reverse=True)
    top3 = filtered[:3] if filtered else predictions[:3]
    
    signals = {
        "engine": "KOOPMAN-SPECTRAL",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "signal_date": datetime.now().strftime("%Y-%m-%d"),
        "target_date": "Next NYSE Open",
        "objective": "MAXIMUM PREDICTED RETURN",
        "data_source": "HF: P2SAMAPA/p2-etf-deepm-data/data/master.parquet",
        "results_repo": "HF: P2SAMAPA/p2-etf-koopman-spectral-results",
        "primary_pick": {
            "etf": top3[0]['etf'],
            "rank": 1,
            "predicted_1d_return_bps": round(top3[0]['predicted_1d_return_bps'], 1),
            "predicted_1d_return_pct": round(top3[0]['predicted_1d_return_bps'] / 100, 3),
            "predictability_index": round(top3[0]['predictability_index'], 3),
            "regime": top3[0]['koopman_regime']
        },
        "runner_up_picks": [
            {
                "rank": i+2,
                "etf": p['etf'],
                "predicted_1d_return_bps": round(p['predicted_1d_return_bps'], 1),
                "predicted_1d_return_pct": round(p['predicted_1d_return_bps'] / 100, 3),
                "predictability_index": round(p['predictability_index'], 3),
                "regime": p['koopman_regime']
            }
            for i, p in enumerate(top3[1:3])
        ],
        "koopman_modes": {
            "regime": top3[0]['koopman_regime'],
            "predictability_index": round(top3[0]['predictability_index'], 3),
            "growth_modes": 0,
            "oscillatory_modes": 0,
            "decay_modes": 0,
            "dominant_frequency_cycles": 0.0
        },
        "all_etfs": predictions,
        "metadata": {
            "total_etfs_analyzed": len(predictions),
            "predictable_etfs": len(filtered),
            "lookback_window": lookback
        }
    }
    return signals


def main():
    print(f"Generating signals at {datetime.now()}")
    config = load_config()
    signals = generate_signals(config)
    output_dir = Path(config['signals']['output_dir'])
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"koopman_signals_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(signals, f, indent=2)
    latest_file = output_dir / "latest.json"
    with open(latest_file, 'w') as f:
        json.dump(signals, f, indent=2)
    print(f"Signals saved to {output_file}")
    print(f"Top pick: {signals['primary_pick']['etf']} with {signals['primary_pick']['predicted_1d_return_bps']:.1f} bps")


if __name__ == "__main__":
    main()
