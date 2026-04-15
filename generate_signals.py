"""
Signal generation using trained Koopman model with GRU encoder.
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
    num_macro = len(config['data']['macro_features'])
    config['model']['input_dim'] = 2 + num_macro
    model = KoopmanSpectral(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    scale_factor = checkpoint.get('scale_factor', 100.0)
    return model, scale_factor


def compute_predictability_index(eigenvalues: torch.Tensor) -> float:
    # Placeholder – not used in GRU variant, kept for compatibility
    return 0.6


def classify_regime(eigenvalues: torch.Tensor) -> str:
    return "oscillatory"


def get_latest_macro_values(loader: HFDataLoader, macro_cols: List[str]) -> Dict[str, float]:
    macro_df = loader.get_macro_data(macro_cols)
    if macro_df.empty:
        return {col: 0.0 for col in macro_cols}
    latest = macro_df.iloc[-1]
    return {col: latest[col] for col in macro_cols if col in latest}


def generate_signals(config) -> Dict:
    lookback = config['data']['lookback_window']
    etf_universe = config['data']['etf_universe']
    macro_cols = config['data']['macro_features']
    
    loader = HFDataLoader(
        use_local=True,
        local_path=config['data'].get('local_path', 'data/p2-etf-deepm-data')
    )
    master = loader.load_master()
    if master.empty:
        raise ValueError("Could not load master.parquet")
    
    model, scale_factor = load_trained_model(config)
    if model is None:
        raise RuntimeError("Trained model not found.")
    
    current_macro = get_latest_macro_values(loader, macro_cols)
    
    predictions = []
    
    for etf in etf_universe:
        df = loader.get_etf_data(etf, lookback=None)
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
        
        # Most recent lookback window
        ret_window = returns[-lookback:].reshape(-1, 1)
        vol_window = vol[-lookback:].reshape(-1, 1)
        macro_history = np.tile([current_macro.get(c, 0.0) for c in macro_cols], (lookback, 1))
        X_seq = np.hstack([ret_window, vol_window, macro_history])
        X_tensor = torch.tensor(X_seq, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            pred_scaled = model(X_tensor).item()  # scaled prediction
            # Unscale
            pred_return = pred_scaled / scale_factor
        
        pred_return_bps = pred_return * 10000
        
        # Use placeholder predictability (global value)
        predictability = 0.6
        regime = "oscillatory"
        
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
        "version": "2.0.0-GRU",
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
            "regime": "oscillatory",
            "predictability_index": 0.6,
            "growth_modes": 0,
            "oscillatory_modes": 0,
            "decay_modes": 0,
            "dominant_frequency_cycles": 0.0
        },
        "all_etfs": predictions,
        "metadata": {
            "total_etfs_analyzed": len(predictions),
            "predictable_etfs": len(filtered),
            "lookback_window": lookback,
            "model_type": "GRU-sequence-encoder"
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
