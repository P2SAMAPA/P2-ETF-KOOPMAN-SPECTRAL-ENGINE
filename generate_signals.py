"""
Daily signal generation for Koopman-Spectral engine.
Outputs: ETF pick + Koopman modes + predictability index
Uses HF dataset: P2SAMAPA/p2-etf-deepm-data/data/master.parquet
"""

import torch
import pandas as pd
import numpy as np
import yaml
import json
from datetime import datetime
from pathlib import Path

from data_loader import load_config, HFDataLoader, NYSECalendar
from koopman_model import KoopmanSpectral


def load_model(config, checkpoint_path='koopman_spectral_best.pt'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = KoopmanSpectral(config).to(device)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {checkpoint_path}")
    except FileNotFoundError:
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Using randomly initialized model (for testing only)")
    
    model.eval()
    return model, device


def generate_daily_signals(date_str, model, device, config):
    """
    Generate signals for all ETFs on given date.
    Returns top 3 by predicted return.
    """
    date = pd.Timestamp(date_str)
    lookback = config['data']['lookback_window']
    etfs = config['data']['etf_universe']
    
    # Load data
    loader = HFDataLoader(use_local=True)
    master = loader.load_master()
    
    if master.empty:
        raise ValueError("Cannot load master.parquet for signal generation")
    
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
    
    returns_col = None
    for candidate in ['log_returns', 'returns', 'ret', 'log_ret']:
        matches = [c for c in available_cols if candidate.lower() in c.lower()]
        if matches:
            returns_col = matches[0]
            break
    
    if etf_col is None or returns_col is None:
        raise ValueError(f"Required columns not found. Have: {available_cols[:10]}...")
    
    signals = []
    
    for etf in etfs:
        etf_data = master[master[etf_col] == etf].copy()
        
        if len(etf_data) < lookback:
            continue
        
        if date_col:
            etf_data = etf_data.sort_values(date_col)
        
        # Get window ending at date
        if date_col:
            idx = etf_data[date_col].searchsorted(date)
            if idx < lookback:
                continue
            window = etf_data.iloc[idx-lookback:idx]
        else:
            window = etf_data.iloc[-lookback:]
        
        if len(window) < lookback:
            continue
        
        # Prepare features
        feature_cols = [returns_col]
        
        # Add volatility if available
        vol_matches = [c for c in available_cols if 'vol' in c.lower()]
        if vol_matches:
            feature_cols.append(vol_matches[0])
        
        # Add macro
        macro_cols = [c for c in available_cols if any(m.lower() in c.lower() 
                      for m in config['data']['macro_features'])]
        feature_cols.extend(macro_cols)
        
        existing_cols = [c for c in feature_cols if c in window.columns]
        
        if len(existing_cols) < 2:
            continue
        
        # Get data
        window_data = window[existing_cols].dropna()
        if len(window_data) < lookback * 0.9:  # 90% coverage required
            continue
        
        # Pad if needed
        if len(window_data) < lookback:
            continue
        
        X = torch.FloatTensor(window_data.values).unsqueeze(0).to(device)
        
        with torch.no_grad():
            returns_pred, modes, Z_future = model(X, return_modes=True)
        
        pred_5day = returns_pred[0].cpu().numpy()
        pred_1day = pred_5day[0]
        
        spectral_gap = modes['spectral_gap']
        predictability = min(1.0, spectral_gap / 0.5)
        
        if modes['growth_count'] > modes['decay_count']:
            regime = "expansion"
        elif modes['oscillatory_count'] > 3:
            regime = "oscillatory"
        else:
            regime = "contraction"
        
        signals.append({
            'date': date_str,
            'etf': etf,
            'predicted_1d_return': float(pred_1day),
            'predicted_5d_return': float(pred_5day.mean()),
            'predictability_index': float(predictability),
            'is_predictable': predictability > config['signals'].get('predictability_threshold', 0.6),
            'koopman_regime': regime,
            'growth_modes': modes['growth_count'],
            'oscillatory_modes': modes['oscillatory_count'],
            'decay_modes': modes['decay_count'],
            'dominant_frequency': float(torch.abs(modes['eigenvalues'].angle()).max()),
            'confidence_score': float(torch.sigmoid(torch.tensor(predictability * 2 - 1)))
        })
    
    if not signals:
        raise ValueError("No signals generated for any ETF")
    
    # SORT BY PREDICTED 1-DAY RETURN (DESCENDING) - MAXIMIZE RETURN
    signals_sorted = sorted(signals, key=lambda x: x['predicted_1d_return'], reverse=True)
    
    # Top 3
    top3 = signals_sorted[:3]
    primary = top3[0]
    runners = top3[1:3] if len(top3) > 1 else []
    
    return {
        'engine': 'KOOPMAN-SPECTRAL',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat(),
        'signal_date': date_str,
        'target_date': 'Next NYSE Open: ' + NYSECalendar.format_trading_date(
            NYSECalendar.get_next_trading_date(datetime.strptime(date_str, '%Y-%m-%d'))),
        'objective': 'MAXIMUM PREDICTED RETURN',
        'data_source': 'HF: P2SAMAPA/p2-etf-deepm-data/data/master.parquet',
        'primary_pick': {
            'etf': primary['etf'],
            'rank': 1,
            'predicted_1d_return_bps': round(primary['predicted_1d_return'] * 10000, 1),
            'predicted_1d_return_pct': round(primary['predicted_1d_return'] * 100, 3),
            'predictability_index': round(primary['predictability_index'], 3),
            'regime': primary['koopman_regime'],
            'conviction_derived': round(primary['confidence_score'] * 100, 1)
        },
        'runner_up_picks': [
            {
                'rank': i + 2,
                'etf': r['etf'],
                'predicted_1d_return_bps': round(r['predicted_1d_return'] * 10000, 1),
                'predicted_1d_return_pct': round(r['predicted_1d_return'] * 100, 3),
                'predictability_index': round(r['predictability_index'], 3),
                'regime': r['koopman_regime']
            }
            for i, r in enumerate(runners)
        ],
        'koopman_modes': {
            'regime': primary['koopman_regime'],
            'predictability_index': round(primary['predictability_index'], 3),
            'dominant_frequency_cycles': round(primary['dominant_frequency'], 3)
        },
        'all_etfs': signals_sorted
    }


def main():
    config = load_config()
    
    # Get next trading date
    next_trading = NYSECalendar.get_next_trading_date()
    today = next_trading.strftime('%Y-%m-%d')
    
    print(f"Generating Koopman-Spectral signals for {today}...")
    print(f"Target: Next NYSE Open ({NYSECalendar.format_trading_date(next_trading)})")
    
    try:
        model, device = load_model(config)
        signals = generate_daily_signals(today, model, device, config)
        
        # Save
        output_dir = Path(config['signals']['output_dir'])
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f"koopman_signals_{today}.json"
        with open(output_file, 'w') as f:
            json.dump(signals, f, indent=2)
        
        print(f"Signals saved to {output_file}")
        print(f"Primary pick: {signals['primary_pick']['etf']} "
              f"({signals['primary_pick']['conviction_derived']}% conviction)")
        print(f"Predicted return: {signals['primary_pick']['predicted_1d_return_bps']:.1f} bps")
        print(f"Koopman regime: {signals['koopman_modes']['regime']}")
        
    except Exception as e:
        print(f"Error generating signals: {e}")
        raise


if __name__ == "__main__":
    main()
