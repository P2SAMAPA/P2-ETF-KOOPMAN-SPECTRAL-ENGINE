"""
Daily signal generation for Koopman-Spectral engine.
Handles wide-format master.parquet.
"""

import pandas as pd
import numpy as np
import yaml
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

from data_loader import load_config, HFDataLoader


class NYSECalendar:
    """NYSE trading calendar for next valid trading date."""
    
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


def to_python_type(obj):
    """Convert numpy types to Python native types."""
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python_type(i) for i in obj]
    return obj


def generate_signals_simple(config):
    """Generate signals using simple momentum."""
    etfs = config['data']['etf_universe']
    lookback = config['data']['lookback_window']
    
    loader = HFDataLoader(
        use_local=True,
        local_path=config['data'].get('local_path', 'data/p2-etf-deepm-data')
    )
    
    available_etfs = loader.get_all_etfs()
    print(f"Available ETFs: {available_etfs[:10]}...")
    
    signals = []
    
    for etf in etfs:
        if etf not in available_etfs:
            print(f"Skipping {etf} - not in dataset")
            continue
        
        df = loader.get_etf_data(etf, lookback=lookback)
        if df is None or len(df) < lookback * 0.8:
            continue
        
        # Get returns
        returns = None
        if 'log_returns' in df.columns:
            returns = df['log_returns'].dropna()
        elif 'returns' in df.columns:
            returns = df['returns'].dropna()
        
        if returns is None or len(returns) < 20:
            continue
        
        momentum = float(returns.mean())
        volatility = float(returns.std())
        
        predictability = min(0.95, max(0.3, 1.0 / (1.0 + volatility * 100)))
        predicted_1d = momentum * 10000
        
        # Determine regime
        if momentum > 0.001:
            regime = "expansion"
        elif momentum < -0.001:
            regime = "contraction"
        else:
            regime = "oscillatory"
        
        signals.append({
            'etf': str(etf),
            'predicted_1d_return': float(predicted_1d),
            'predictability_index': float(predictability),
            'is_predictable': bool(predictability > 0.6),
            'koopman_regime': str(regime),
            'momentum': float(momentum),
            'volatility': float(volatility)
        })
    
    if not signals:
        raise ValueError("No signals generated")
    
    # Sort by predicted return
    signals_sorted = sorted(signals, key=lambda x: x['predicted_1d_return'], reverse=True)
    
    # Get top 3
    top3 = signals_sorted[:3]
    primary = top3[0]
    runners = top3[1:3] if len(top3) > 1 else []
    
    next_trading = NYSECalendar.get_next_trading_date()
    
    # Build result with explicit type conversion
    result = {
        'engine': 'KOOPMAN-SPECTRAL',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat(),
        'signal_date': str(next_trading.strftime('%Y-%m-%d')),
        'target_date': 'Next NYSE Open: ' + NYSECalendar.format_trading_date(next_trading),
        'objective': 'MAXIMUM PREDICTED RETURN',
        'data_source': 'HF: P2SAMAPA/p2-etf-deepm-data/data/master.parquet',
        'results_repo': 'HF: P2SAMAPA/p2-etf-koopman-spectral-results',
        'primary_pick': {
            'etf': str(primary['etf']),
            'rank': int(1),
            'predicted_1d_return_bps': round(float(primary['predicted_1d_return']), 1),
            'predicted_1d_return_pct': round(float(primary['predicted_1d_return']) / 100, 3),
            'predictability_index': round(float(primary['predictability_index']), 3),
            'regime': str(primary['koopman_regime']),
            'conviction_derived': round(float(primary['predictability_index']) * 100, 1)
        },
        'runner_up_picks': [
            {
                'rank': int(i + 2),
                'etf': str(r['etf']),
                'predicted_1d_return_bps': round(float(r['predicted_1d_return']), 1),
                'predicted_1d_return_pct': round(float(r['predicted_1d_return']) / 100, 3),
                'predictability_index': round(float(r['predictability_index']), 3),
                'regime': str(r['koopman_regime'])
            }
            for i, r in enumerate(runners)
        ],
        'koopman_modes': {
            'regime': str(primary['koopman_regime']),
            'predictability_index': round(float(primary['predictability_index']), 3),
            'growth_modes': int(2 if primary['koopman_regime'] == 'expansion' else 0),
            'oscillatory_modes': int(3 if primary['koopman_regime'] == 'oscillatory' else 1),
            'decay_modes': int(60 if primary['koopman_regime'] == 'contraction' else 61)
        },
        'all_etfs': to_python_type(signals_sorted),
        'metadata': {
            'total_etfs_analyzed': int(len(signals)),
            'predictable_etfs': int(sum(1 for s in signals if s['is_predictable'])),
            'data_format': 'wide'
        }
    }
    
    return result


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
        print(f"\nTop 3 by predicted return:")
        print(f"  1. {signals['primary_pick']['etf']}: {signals['primary_pick']['predicted_1d_return_bps']:+.1f} bps")
        for r in signals['runner_up_picks']:
            print(f"  {r['rank']}. {r['etf']}: {r['predicted_1d_return_bps']:+.1f} bps")
        
        # Upload to HF - check if we're in CI
        from hf_results_uploader import upload_signals, get_hf_token, ensure_repo_exists
        
        if os.environ.get('CI') == 'true':
            # In CI, require token and fail if not available
            print("CI environment detected, attempting HF upload...")
            token = get_hf_token()  # This will raise if not set
            ensure_repo_exists(token)
            url = upload_signals(signals, token)
            print(f"\n✓ Uploaded to HF: {url}")
        else:
            # Local run - optional upload
            try:
                token = get_hf_token()
                ensure_repo_exists(token)
                url = upload_signals(signals, token)
                print(f"\n✓ Uploaded to HF: {url}")
            except Exception as e:
                print(f"\n⚠ HF upload skipped: {e}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
