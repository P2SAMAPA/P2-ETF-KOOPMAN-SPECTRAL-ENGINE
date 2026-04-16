"""
Signal generation using trained Koopman model with LSTM sequence encoder.

Fixes applied:
  1. Macro data is aligned per-ETF by date (was a shared array — root cause
     of identical predictions).
  2. StandardScaler loaded from checkpoint and applied before inference.
  3. etf_idx passed to model.forward() so ETF embedding is used.
  4. etf_to_idx loaded from checkpoint; falls back to config order if absent.
  5. Mode counts populated from actual K eigenvalue analysis.
"""

import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from data_loader import HFDataLoader, load_config, load_scaler, apply_scaler
from koopman_model import KoopmanSpectral


# ── Model loading ─────────────────────────────────────────────────────────────

def load_trained_model(
    config,
    model_path: str = "koopman_spectral_best.pt",
) -> tuple:
    """
    Returns (model, etf_to_idx) or (None, {}) if checkpoint not found.
    Reads num_etfs and etf_to_idx from the checkpoint so the architecture
    matches what was trained.
    """
    if not Path(model_path).exists():
        print(f"Model checkpoint not found: {model_path}")
        return None, {}

    checkpoint = torch.load(model_path, map_location='cpu')

    # Rebuild config with correct dimensions
    saved_cfg   = checkpoint.get('config', config)
    num_macro   = len(saved_cfg['data']['macro_features'])
    saved_cfg['model']['input_dim'] = 2 + num_macro

    # Use num_etfs / etf_to_idx saved at training time
    etf_to_idx = checkpoint.get('etf_to_idx',
                                {e: i for i, e in enumerate(saved_cfg['data']['etf_universe'])})
    saved_cfg['model']['num_etfs']    = checkpoint.get('num_etfs', len(etf_to_idx))
    saved_cfg['model']['etf_emb_dim'] = saved_cfg['model'].get('etf_emb_dim', 16)

    model = KoopmanSpectral(saved_cfg)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, etf_to_idx


# ── Spectral helpers ──────────────────────────────────────────────────────────

def compute_predictability_index(eigenvalues: torch.Tensor) -> float:
    mags     = torch.abs(eigenvalues)
    log_mags = torch.log(mags + 1e-8)
    variance = torch.var(log_mags)
    return float(1.0 / (1.0 + variance))


def classify_regime(eigenvalues: torch.Tensor) -> str:
    mags         = torch.abs(eigenvalues)
    growth_ratio = (mags > 1.05).float().mean().item()
    decay_ratio  = (mags < 0.95).float().mean().item()
    if growth_ratio > 0.2:
        return "expansion"
    elif decay_ratio > 0.3:
        return "contraction"
    return "oscillatory"


def count_modes(eigenvalues: torch.Tensor) -> dict:
    mags = torch.abs(eigenvalues)
    imag = torch.abs(eigenvalues.imag)
    return {
        'growth':      int((mags > 1.05).sum()),
        'oscillatory': int(((mags >= 0.95) & (mags <= 1.05) & (imag > 0.01)).sum()),
        'decay':       int((mags < 0.95).sum()),
    }


# ── Main signal generation ────────────────────────────────────────────────────

def generate_signals(config) -> Dict:
    lookback       = config['data']['lookback_window']
    etf_universe   = config['data']['etf_universe']
    macro_cols     = config['data']['macro_features']

    loader = HFDataLoader(
        use_local=True,
        local_path=config['data'].get('local_path', 'data/p2-etf-deepm-data'),
    )
    master = loader.load_master()
    if master.empty:
        raise ValueError("Could not load master.parquet")

    model, etf_to_idx = load_trained_model(config)
    if model is None:
        raise RuntimeError("Trained model not found. Run train.py first.")

    # Load scaler saved during training
    scaler = load_scaler("koopman_scaler.pkl")

    # Load full macro series once (we will merge per-ETF by date below)
    macro_df = loader.get_macro_data(macro_cols)

    # Global spectral stats (same K for all ETFs — per-ticker differentiation
    # comes from the LSTM hidden state and the ETF embedding, not K itself)
    with torch.no_grad():
        eigs = model.get_eigenvalues()
    global_predictability = compute_predictability_index(eigs)
    global_regime         = classify_regime(eigs)
    global_modes          = count_modes(eigs)
    dominant_freq         = float(torch.abs(eigs.imag).max())

    predictions = []

    for etf in etf_universe:
        df = loader.get_etf_data(etf, lookback=None)
        if df is None or len(df) < lookback + 1:
            continue

        returns_col = 'log_returns' if 'log_returns' in df.columns else 'returns'
        if returns_col not in df.columns:
            continue

        returns = df[returns_col].values
        vol     = pd.Series(returns).rolling(5).std().values
        valid   = ~(np.isnan(returns) | np.isnan(vol))
        returns = returns[valid]
        vol     = vol[valid]
        dates   = df['date'].values[valid]

        if len(returns) < lookback:
            continue

        # FIX: align macro to THIS ETF's date index
        merged = pd.DataFrame({'date': pd.to_datetime(dates[-lookback:]),
                                'returns': returns[-lookback:],
                                'vol': vol[-lookback:]})

        if not macro_df.empty:
            merged = merged.merge(macro_df, on='date', how='left')
            for col in macro_cols:
                if col in merged.columns:
                    merged[col] = merged[col].ffill().bfill().fillna(0.0)
                else:
                    merged[col] = 0.0
        else:
            for col in macro_cols:
                merged[col] = 0.0

        ret_w   = merged['returns'].values.reshape(-1, 1)
        vol_w   = merged['vol'].values.reshape(-1, 1)
        mac_w   = merged[macro_cols].values
        X_seq   = np.hstack([ret_w, vol_w, mac_w])  # [lookback, 2+M]

        # FIX: apply scaler before inference
        X_seq_scaled = apply_scaler(X_seq[np.newaxis], scaler)[0]  # [lookback, F]

        X_tensor = torch.tensor(X_seq_scaled, dtype=torch.float32).unsqueeze(0)  # [1, L, F]
        etf_idx  = torch.tensor([etf_to_idx.get(etf, 0)], dtype=torch.long)

        with torch.no_grad():
            pred_return = model(X_tensor, etf_idx=etf_idx).item()

        pred_return_bps = pred_return * 10000

        predictions.append({
            'etf':                      etf,
            'predicted_1d_return_bps':  pred_return_bps,
            'predicted_1d_return':      pred_return_bps,
            'predictability_index':     global_predictability,
            'koopman_regime':           global_regime,
            'is_predictable':           global_predictability >= config['signals']['predictability_threshold'],
            'data_quality':             1.0,
        })

    if not predictions:
        raise RuntimeError("No predictions generated — check that ETF data is available.")

    # Filter and rank
    threshold = config['signals']['predictability_threshold']
    filtered  = [p for p in predictions if p['predictability_index'] >= threshold]
    filtered.sort(key=lambda x: x['predicted_1d_return_bps'], reverse=True)
    top3 = filtered[:3] if filtered else sorted(
        predictions, key=lambda x: x['predicted_1d_return_bps'], reverse=True
    )[:3]

    signals = {
        "engine":      "KOOPMAN-SPECTRAL",
        "version":     "1.1.0",
        "timestamp":   datetime.now().isoformat(),
        "signal_date": datetime.now().strftime("%Y-%m-%d"),
        "target_date": "Next NYSE Open",
        "objective":   "MAXIMUM PREDICTED RETURN",
        "data_source": "HF: P2SAMAPA/p2-etf-deepm-data/data/master.parquet",
        "results_repo": "HF: P2SAMAPA/p2-etf-koopman-spectral-results",
        "primary_pick": {
            "etf":                     top3[0]['etf'],
            "rank":                    1,
            "predicted_1d_return_bps": round(top3[0]['predicted_1d_return_bps'], 1),
            "predicted_1d_return_pct": round(top3[0]['predicted_1d_return_bps'] / 100, 3),
            "predictability_index":    round(top3[0]['predictability_index'], 3),
            "regime":                  top3[0]['koopman_regime'],
        },
        "runner_up_picks": [
            {
                "rank":                    i + 2,
                "etf":                     p['etf'],
                "predicted_1d_return_bps": round(p['predicted_1d_return_bps'], 1),
                "predicted_1d_return_pct": round(p['predicted_1d_return_bps'] / 100, 3),
                "predictability_index":    round(p['predictability_index'], 3),
                "regime":                  p['koopman_regime'],
            }
            for i, p in enumerate(top3[1:3])
        ],
        "koopman_modes": {
            "regime":                   global_regime,
            "predictability_index":     round(global_predictability, 3),
            "growth_modes":             global_modes['growth'],
            "oscillatory_modes":        global_modes['oscillatory'],
            "decay_modes":              global_modes['decay'],
            "dominant_frequency_cycles": round(dominant_freq, 4),
        },
        "all_etfs": predictions,
        "metadata": {
            "total_etfs_analyzed": len(predictions),
            "predictable_etfs":    len(filtered),
            "lookback_window":     lookback,
            "scaler_applied":      scaler is not None,
            "etf_embedding_used":  model.use_etf_emb,
        },
    }
    return signals


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print(f"Generating signals at {datetime.now()}")
    config  = load_config()
    signals = generate_signals(config)

    output_dir = Path(config['signals']['output_dir'])
    output_dir.mkdir(exist_ok=True)

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"koopman_signals_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(signals, f, indent=2)

    latest_file = output_dir / "latest.json"
    with open(latest_file, 'w') as f:
        json.dump(signals, f, indent=2)

    print(f"Signals saved to {output_file}")
    print(f"Top pick: {signals['primary_pick']['etf']} "
          f"with {signals['primary_pick']['predicted_1d_return_bps']:.1f} bps")
    print("\nAll ETF predictions (sorted by predicted return):")
    ranked = sorted(signals['all_etfs'],
                    key=lambda x: x['predicted_1d_return_bps'], reverse=True)
    for p in ranked:
        print(f"  {p['etf']:6s}  {p['predicted_1d_return_bps']:+7.1f} bps")


if __name__ == "__main__":
    main()
