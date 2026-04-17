"""
Signal generation for Koopman-Spectral engine.

Fixes vs original:
  1. Macro aligned per-ETF by date inside the loop (was shared array — root cause).
  2. Scaler loaded and applied before inference.
  3. etf_idx passed to model for ETF embedding.
  4. SPY excluded from picks (benchmark only); shown separately if data available.
  5. Mode counts from actual K eigenvalues (were all hardcoded 0).
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


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_trained_model(config, model_path: str = "koopman_spectral_best.pt"):
    if not Path(model_path).exists():
        print(f"Checkpoint not found: {model_path}")
        return None, {}

    ckpt = torch.load(model_path, map_location='cpu')
    cfg  = ckpt.get('config', config)

    num_macro = len(cfg['data']['macro_features'])
    cfg['model']['input_dim'] = 2 + num_macro

    etf_to_idx = ckpt.get('etf_to_idx',
                           {e: i for i, e in enumerate(cfg['data']['etf_universe'])})
    cfg['model']['num_etfs']    = ckpt.get('num_etfs', len(etf_to_idx))
    cfg['model']['etf_emb_dim'] = cfg['model'].get('etf_emb_dim', 16)

    model = KoopmanSpectral(cfg)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, etf_to_idx


# ---------------------------------------------------------------------------
# Spectral helpers
# ---------------------------------------------------------------------------

def compute_predictability_index(eigs: torch.Tensor) -> float:
    mags = torch.abs(eigs)
    return float(1.0 / (1.0 + torch.var(torch.log(mags + 1e-8))))


def classify_regime(eigs: torch.Tensor) -> str:
    mags = torch.abs(eigs)
    if (mags > 1.05).float().mean() > 0.2:  return "expansion"
    if (mags < 0.95).float().mean() > 0.3:  return "contraction"
    return "oscillatory"


def count_modes(eigs: torch.Tensor) -> dict:
    mags = torch.abs(eigs)
    return {
        'growth':      int((mags > 1.05).sum()),
        'oscillatory': int(((mags >= 0.95) & (mags <= 1.05)).sum()),
        'decay':       int((mags < 0.95).sum()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_signals(config) -> Dict:
    lookback     = config['data']['lookback_window']
    etf_universe = config['data']['etf_universe']   # SPY already excluded
    benchmark    = config['data'].get('benchmark', 'SPY')
    macro_cols   = config['data']['macro_features']

    loader = HFDataLoader(
        use_local=True,
        local_path=config['data'].get('local_path', 'data/p2-etf-deepm-data'),
    )
    loader.load_master()

    model, etf_to_idx = load_trained_model(config)
    if model is None:
        raise RuntimeError("No trained model found. Run train.py first.")

    scaler   = load_scaler("koopman_scaler.pkl")
    macro_df = loader.get_macro_data(macro_cols)

    # Global spectral stats from K
    eigs                  = model.get_eigenvalues()
    global_predictability = compute_predictability_index(eigs)
    global_regime         = classify_regime(eigs)
    global_modes          = count_modes(eigs)
    dominant_freq         = float(torch.abs(eigs.imag).max())

    def predict_etf(etf: str, etf_i: int) -> Optional[dict]:
        df = loader.get_etf_data(etf, lookback=None)
        if df is None or len(df) < lookback + 1:
            return None

        ret_col = 'log_returns' if 'log_returns' in df.columns else 'returns'
        if ret_col not in df.columns:
            return None

        returns = df[ret_col].values
        vol     = pd.Series(returns).rolling(5).std().values
        valid   = ~(np.isnan(returns) | np.isnan(vol))
        returns, vol = returns[valid], vol[valid]
        dates        = df['date'].values[valid]

        if len(returns) < lookback:
            return None

        # FIX: per-ETF macro alignment by date
        merged = pd.DataFrame({
            'date':    pd.to_datetime(dates[-lookback:]),
            'returns': returns[-lookback:],
            'vol':     vol[-lookback:],
        })
        if not macro_df.empty:
            merged = merged.merge(macro_df, on='date', how='left')
        for col in macro_cols:
            if col in merged.columns:
                merged[col] = merged[col].ffill().bfill().fillna(0.0)
            else:
                merged[col] = 0.0

        X_seq = np.hstack([
            merged['returns'].values.reshape(-1, 1),
            merged['vol'].values.reshape(-1, 1),
            merged[macro_cols].values,
        ])                                                  # [lookback, 2+M]

        X_scaled = apply_scaler(X_seq[np.newaxis], scaler)[0]
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)
        idx_t    = torch.tensor([etf_i], dtype=torch.long)

        with torch.no_grad():
            pred_bps = model(X_tensor, etf_idx=idx_t).item() * 10000

        return {
            'etf':                      etf,
            'predicted_1d_return_bps':  pred_bps,
            'predicted_1d_return':      pred_bps,
            'predictability_index':     global_predictability,
            'koopman_regime':           global_regime,
            'is_predictable':           global_predictability >= config['signals']['predictability_threshold'],
            'data_quality':             1.0,
        }

    # Run predictions — picks universe (no SPY)
    predictions = []
    for etf in etf_universe:
        result = predict_etf(etf, etf_to_idx.get(etf, 0))
        if result:
            predictions.append(result)

    # Benchmark reference (SPY, not ranked)
    spy_ref = predict_etf(benchmark, etf_to_idx.get(benchmark, 0))

    if not predictions:
        raise RuntimeError("No predictions generated — check ETF data availability.")

    threshold = config['signals']['predictability_threshold']
    filtered  = [p for p in predictions if p['predictability_index'] >= threshold]
    filtered.sort(key=lambda x: x['predicted_1d_return_bps'], reverse=True)
    top3 = (filtered or sorted(predictions,
                                key=lambda x: x['predicted_1d_return_bps'],
                                reverse=True))[:3]

    signals = {
        "engine":       "KOOPMAN-SPECTRAL",
        "version":      "1.1.0",
        "timestamp":    datetime.now().isoformat(),
        "signal_date":  datetime.now().strftime("%Y-%m-%d"),
        "target_date":  "Next NYSE Open",
        "objective":    "MAXIMUM PREDICTED RETURN",
        "data_source":  "HF: P2SAMAPA/p2-etf-deepm-data/data/master.parquet",
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
        "benchmark": {
            "etf":                     benchmark,
            "note":                    "Benchmark only — excluded from picks",
            "predicted_1d_return_bps": round(spy_ref['predicted_1d_return_bps'], 1) if spy_ref else None,
        },
        "koopman_modes": {
            "regime":                    global_regime,
            "predictability_index":      round(global_predictability, 3),
            "growth_modes":              global_modes['growth'],
            "oscillatory_modes":         global_modes['oscillatory'],
            "decay_modes":               global_modes['decay'],
            "dominant_frequency_cycles": round(dominant_freq, 4),
        },
        "all_etfs": sorted(predictions,
                           key=lambda x: x['predicted_1d_return_bps'],
                           reverse=True),
        "metadata": {
            "total_etfs_analyzed": len(predictions),
            "predictable_etfs":    len(filtered),
            "lookback_window":     lookback,
            "scaler_applied":      scaler is not None,
            "etf_embedding_used":  model.use_etf_emb,
            "benchmark_excluded":  benchmark,
        },
    }
    return signals


def main():
    print(f"Generating signals at {datetime.now()}")
    config  = load_config()
    signals = generate_signals(config)

    out_dir = Path(config['signals']['output_dir'])
    out_dir.mkdir(exist_ok=True)

    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"koopman_signals_{ts}.json"
    with open(path, 'w') as f:
        json.dump(signals, f, indent=2)
    with open(out_dir / "latest.json", 'w') as f:
        json.dump(signals, f, indent=2)

    print(f"Saved → {path}")
    print(f"\nTop pick: {signals['primary_pick']['etf']} "
          f"{signals['primary_pick']['predicted_1d_return_bps']:+.1f} bps")
    print("\nFull ranking:")
    for p in signals['all_etfs']:
        flag = " ← PICK" if p['etf'] in [top['etf'] for top in [signals['primary_pick']] + signals['runner_up_picks']] else ""
        print(f"  {p['etf']:6s}  {p['predicted_1d_return_bps']:+7.1f} bps{flag}")


if __name__ == "__main__":
    main()
