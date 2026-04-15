"""
Signal generation for Koopman-Spectral engine.
Loads trained model and produces JSON output with top ETF picks.
"""

import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import yaml

from data_loader import HFDataLoader, load_config
from koopman_model import KoopmanSpectral


def load_trained_model(config, model_path: str = "koopman_spectral_best.pt") -> Optional[KoopmanSpectral]:
    """Load the trained Koopman model from checkpoint."""
    if not Path(model_path).exists():
        print(f"Model checkpoint not found: {model_path}")
        return None
    
    checkpoint = torch.load(model_path, map_location='cpu')
    model = KoopmanSpectral(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def compute_predictability_index(eigenvalues: torch.Tensor) -> float:
    """
    Compute predictability index from Koopman eigenvalues.
    Higher when eigenvalues are clustered near unit circle (oscillatory/stable).
    """
    mags = torch.abs(eigenvalues)
    log_mags = torch.log(mags + 1e-8)
    variance = torch.var(log_mags)
    predictability = 1.0 / (1.0 + variance)
    return float(predictability)


def classify_regime(eigenvalues: torch.Tensor) -> str:
    """Classify market regime based on dominant eigenvalues."""
    mags = torch.abs(eigenvalues)
    growth_ratio = (mags > 1.05).float().mean().item()
    decay_ratio = (mags < 0.95).float().mean().item()
    
    if growth_ratio > 0.2:
        return "expansion"
    elif decay_ratio > 0.3:
        return "contraction"
    else:
        return "oscillatory"


def generate_signals(config) -> Dict:
    """
    Generate trading signals using trained Koopman model.
    Returns dict in the format expected by the Streamlit app.
    """
    lookback = config['data']['lookback_window']
    target_horizon = config['data'].get('target_horizon', 5)
    etf_universe = config['data']['etf_universe']
    
    # Load data
    loader = HFDataLoader(
        use_local=True,
        local_path=config['data'].get('local_path', 'data/p2-etf-deepm-data')
    )
    master = loader.load_master()
    if master.empty:
        raise ValueError("Could not load master.parquet")
    
    # Load trained model
    model = load_trained_model(config)
    if model is None:
        raise RuntimeError("Trained model not found. Run train.py first.")
    
    predictions = []
    
    for etf in etf_universe:
        # Get full historical data for this ETF
        df = loader.get_etf_data(etf)
        if df is None or len(df) < lookback + 1:
            continue
        
        # Use log returns
        if 'log_returns' in df.columns:
            returns = df['log_returns'].values
        elif 'returns' in df.columns:
            returns = df['returns'].values
        else:
            continue
        
        # Compute volatility
        vol = pd.Series(returns).rolling(5).std().values
        
        # Align and remove NaNs
        valid = ~(np.isnan(returns) | np.isnan(vol))
        returns = returns[valid]
        vol = vol[valid]
        if len(returns) < lookback + 1:
            continue
        
        # Take most recent lookback window
        X_seq = np.column_stack([
            returns[-lookback:],
            vol[-lookback:]
        ])  # shape [lookback, 2]
        X_tensor = torch.tensor(X_seq, dtype=torch.float32).unsqueeze(0)  # [1, lookback, 2]
        
        # Predict next return
        with torch.no_grad():
            last_obs = X_tensor[0, -1, :]  # [2]
            z_last = model.encoder(last_obs.unsqueeze(0))  # [1, obs_dim]
            pred_return = model(z_last).item()  # log return (e.g., 0.001 = 10 bps)
        
        # Convert to basis points (log return * 10000)
        pred_return_bps = pred_return * 10000
        
        # Compute predictability from model eigenvalues
        with torch.no_grad():
            eigs = torch.linalg.eigvals(model.K)
            predictability = compute_predictability_index(eigs)
            regime = classify_regime(eigs)
        
        predictions.append({
            'etf': etf,
            'predicted_1d_return_bps': pred_return_bps,
            'predicted_1d_return': pred_return_bps,   # ← for ranking table
            'predictability_index': predictability,
            'koopman_regime': regime,
            'is_predictable': predictability >= config['signals']['predictability_threshold'],
            'data_quality': 1.0  # placeholder
        })
    
    # Filter by predictability threshold and sort by predicted return
    threshold = config['signals']['predictability_threshold']
    filtered = [p for p in predictions if p['predictability_index'] >= threshold]
    filtered.sort(key=lambda x: x['predicted_1d_return_bps'], reverse=True)
    
    # Take top 3
    top3 = filtered[:3]
    if len(top3) == 0:
        predictions.sort(key=lambda x: x['predicted_1d_return_bps'], reverse=True)
        top3 = predictions[:3]
    
    # Build output structure matching the Streamlit app's expectations
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
        "all_etfs": predictions,   # contains 'predicted_1d_return' for ranking table
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
    
    try:
        signals = generate_signals(config)
        
        # Save to JSON
        output_dir = Path(config['signals']['output_dir'])
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"koopman_signals_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(signals, f, indent=2)
        
        # Also save as latest.json
        latest_file = output_dir / "latest.json"
        with open(latest_file, 'w') as f:
            json.dump(signals, f, indent=2)
        
        print(f"Signals saved to {output_file}")
        print(f"Top pick: {signals['primary_pick']['etf']} with {signals['primary_pick']['predicted_1d_return_bps']:.1f} bps")
        
        # Optionally upload to HuggingFace
        if config.get('github_actions', {}).get('upload_hf', False):
            from hf_results_uploader import upload_signals
            upload_signals(output_file)
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
