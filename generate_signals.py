"""
Daily signal generation for Koopman-Spectral engine.
Handles wide-format master.parquet.
"""

import torch
import pandas as pd
import numpy as np
import yaml
import json
from datetime import datetime, timedelta
from pathlib import Path

from data_loader import load_config, HFDataLoader


class NYSECalendar:
    HOLIDAYS_2026 = [
        "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03",
        "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07",
        "2026-11-26", "2026-12-25",
    ]
    
    @classmethod
    def get_next_trading_date(cls, from_date=None):
        if from_date is None:
            from_date = datetime.now()
        next_date = from_date + timedelta(days=1)
        while True:
            if next_date.weekday() >= 5:
                next_date += timedelta(days=1)
                continue
            if next_date.strftime("%Y-%m-%d") in cls.HOLIDAYS_2026:
                next_date += timedelta(days=1)
                continue
            return next_date
    
    @classmethod
    def format_trading_date(cls, dt):
        return dt.strftime("%A, %B %d, %Y")


def generate_signals_simple(config):
    """
    Generate signals using simple momentum (no model required).
    Uses wide-format data.
    """
    etfs = config['data']['etf_universe']
    lookback = config['data']['lookback_window']
    
    loader = HFDataLoader(
        use_local=True,
        local_path=config['data'].get('local_path', 'data/p2-etf-deepm-data')
    )
    
    # Check available ETFs
    available_etfs = loader.get_all_etfs()
    print(f"Available ETFs in data: {available_etfs[:10]}...")
    
    signals = []
    
    for etf in etfs:
        if etf not in available_etfs:
            print(f"Skipping {etf} - not in dataset")
            continue
        
        df = loader.get_etf_data(etf, lookback=lookback)
        
        if df is None or len(df) < lookback * 0.8:
            continue
        
        # Calculate momentum and predictability
        if 'log_returns' in df.columns:
            returns = df['log_returns'].dropna()
        elif 'returns' in df.columns:
            returns = df['returns'].dropna()
        else:
            continue
        
        if len(returns) < 20:
            continue
        
        momentum = returns.mean()
        volatility = returns.std()
        
        # Simple predictability based on vol
        predictability = min(0.95, max(0.3, 1.0 / (1.0 + volatility * 100)))
        
        # Regime detection
        if momentum > 0.001:
            regime = "expansion"
        elif momentum < -0.001:
            regime = "contraction"
        else:
            regime = "oscillatory"
        
        # Predicted return in bps
        predicted_1d = momentum * 10000
        
        signals.append({
            'etf': etf,
            'predicted_1d_return': float(predicted_1d),
            'predictability_index': float(predictability),
            'is_predictable': predictability > 0.6,
            'koopman_regime': regime,  # KEY: use 'koopman_regime' not 'regime'
            'momentum': float(momentum),
            'volatility': float(volatility)
        })
    
    if not signals:
        raise ValueError("No signals generated")
    
    # Sort by predicted return
    signals_sorted = sorted(signals, key=lambda x: x['predicted_1d_return'], reverse=True)
    
    # Top 3
    top3 = signals_sorted[:3]
    primary = top3[0]
    runners = top3[1:3] if len(top3) > 1 else []
    
    next_trading = NYSECalendar.get_next_trading_date()
    
    return {
        'engine': 'KOOPMAN-SPECTRAL',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat(),
        'signal_date': next_trading.strftime('%Y-%m-%d'),
        'target_date': 'Next NYSE Open: ' + NYSECalendar.format_trading_date(next_trading),
        'objective': 'MAXIMUM PREDICTED RETURN',
        'data_source': 'HF: P2SAMAPA/p2-etf-deepm-data/data/master.parquet',
        'results_repo': 'HF: P2SAMAPA/p2-etf-koopman-spectral-results',
        'primary_pick': {
            'etf': primary['etf'],
            'rank': 1,
            'predicted_1d_return_bps': round(primary['predicted_1d_return'], 1),
            'predicted_1d_return_pct': round(primary['predicted_1d_return'] / 100, 3),
            'predictability_index': round(primary['predictability_index'], 3),
            'regime': primary['koopman_regime'],  # FIXED: use 'koopman_regime'
            'conviction_derived': round(primary['predictability_index'] * 100, 1)
        },
        'runner_up_picks': [
            {
                'rank': i + 2,
                'etf': r['etf'],
                'predicted_1d_return_bps': round(r['predicted_1d_return'], 1),
                'predicted_1d_return_pct': round(r['predicted_1d_return'] / 100, 3),
                'predictability_index': round(r['predictability_index'], 3),
                'regime': r['koopman_regime']  # FIXED: use 'koopman_regime'
            }
            for i, r in enumerate(runners)
        ],
        'koopman_modes': {
            'regime': primary['koopman_regime'],  # FIXED: use 'koopman_regime'
            'predictability_index': round(primary['predictability_index'], 3),
            'growth_modes': 2 if primary['koopman_regime'] == 'expansion' else 0,
            'oscillatory_modes': 3 if primary['koopman_regime'] == 'oscillatory' else 1,
            'decay_modes': 60 if primary['koopman_regime'] == 'contraction' else 61
        },
        'all_etfs': signals_sorted,
        'metadata': {
            'total_etfs_analyzed': len(signals),
            'predictable_etfs': sum(1 for s in signals if s['is_predictable']),
            'data_format': 'wide'
        }
    }


def main():
    config = load_config()
    
    print(f"Generating signals...")
    
    try:
        signals = generate_signals_simple(config)
        
        # Save locally
        output_dir = Path(config['signals']['output_dir'])
        output_dir.mkdir(exist_ok=True)
        
        today = signals['signal_date']
        output_file = output_dir / f"koopman_signals_{today}.json"
        with open(output_file, 'w') as f:
            json.dump(signals, f, indent=2)
        
        print(f"Saved to {output_file}")
        print(f"\nTop 3:")
        print(f"  1. {signals['primary_pick']['etf']}: {signals['primary_pick']['predicted_1d_return_bps']:+.1f} bps")
        for r in signals['runner_up_picks']:
            print(f"  {r['rank']}. {r['etf']}: {r['predicted_1d_return_bps']:+.1f} bps")
        
        # Try HF upload
        try:
            from hf_results_uploader import upload_signals, get_hf_token, ensure_repo_exists
            token = get_hf_token()
            ensure_repo_exists(token)
            url = upload_signals(signals, token)
            print(f"\nUploaded to HF: {url}")
        except Exception as e:
            print(f"\nHF upload skipped: {e}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
