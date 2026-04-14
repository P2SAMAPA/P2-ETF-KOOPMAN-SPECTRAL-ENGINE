"""
Daily signal generation for Koopman-Spectral engine.
Outputs: ETF pick + Koopman modes + predictability index
"""

import torch
import pandas as pd
import numpy as np
import yaml
import json
from datetime import datetime
from pathlib import Path

from data_loader import load_config, fetch_etf_data, fetch_macro_data
from koopman_model import KoopmanSpectral


def load_model(config, checkpoint_path='koopman_spectral_best.pt'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = KoopmanSpectral(config).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, device


def generate_daily_signals(date_str, model, device, config):
    """
    Generate signals for all ETFs on given date.
    Returns structured output for P2 downstream consumption.
    """
    date = pd.Timestamp(date_str)
    lookback = config['data']['lookback_window']
    etfs = config['data']['etf_universe']
    
    signals = []
    
    for etf in etfs:
        # Load data
        df = fetch_etf_data(etf, config)
        macro = fetch_macro_data(config)
        df = df.join(macro, how='left')
        
        # Feature engineering
        df['returns_lag1'] = df['log_returns'].shift(1)
        df['vol_norm'] = df['vol'] / df['vol'].rolling(21).mean()
        feature_cols = ['log_returns', 'returns_lag1', 'vol_norm'] + config['data']['macro_features']
        df = df.dropna()
        
        # Get window ending at date
        idx = df.index.get_indexer([date], method='nearest')[0]
        if idx < lookback:
            continue
            
        window = df.iloc[idx-lookback:idx][feature_cols].values
        X = torch.FloatTensor(window).unsqueeze(0).to(device)  # [1, T, F]
        
        with torch.no_grad():
            returns_pred, modes, Z_future = model(X, return_modes=True)
        
        # Decode predictions
        pred_5day = returns_pred[0].cpu().numpy()  # 5-day ahead return predictions
        pred_1day = pred_5day[0]  # Tomorrow's return
        
        # Predictability score from spectral gap
        spectral_gap = modes['spectral_gap']
        predictability = min(1.0, spectral_gap / 0.5)  # Normalize
        
        # Classify dominant mode
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
            'is_predictable': predictability > config['signals']['predictability_threshold'],
            'koopman_regime': regime,
            'growth_modes': modes['growth_count'],
            'oscillatory_modes': modes['oscillatory_count'],
            'decay_modes': modes['decay_count'],
            'dominant_frequency': float(torch.abs(modes['eigenvalues'].angle()).max()),
            'confidence_score': float(torch.sigmoid(torch.tensor(predictability * 2 - 1)))  # Sigmoid scale
        })
    
    # Rank and select
    df_signals = pd.DataFrame(signals)
    
    # Filter predictable only
    predictable = df_signals[df_signals['is_predictable']]
    
    if len(predictable) == 0:
        # Fallback: pick highest predictability even if below threshold
        top_pick = df_signals.loc[df_signals['predictability_index'].idxmax()]
    else:
        # Pick highest expected return among predictable
        top_pick = predictable.loc[predictable['predicted_1d_return'].idxmax()]
    
    # Build P2-standard output
    output = {
        'engine': 'KOOPMAN-SPECTRAL',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat(),
        'signal_date': date_str,
        'primary_pick': {
            'etf': top_pick['etf'],
            'conviction_pct': round(top_pick['confidence_score'] * 100, 1),
            'expected_return_1d': round(top_pick['predicted_1d_return'] * 100, 3),  # bps
            'expected_return_5d': round(top_pick['predicted_5d_return'] * 100, 3),
        },
        'koopman_modes': {
            'regime': top_pick['koopman_regime'],
            'growth_modes': top_pick['growth_modes'],
            'oscillatory_modes': top_pick['oscillatory_modes'],
            'decay_modes': top_pick['decay_modes'],
            'predictability_index': round(top_pick['predictability_index'], 3),
            'dominant_frequency_cycles': round(top_pick['dominant_frequency'], 3)
        },
        'runner_up_picks': df_signals.nlargest(3, 'predicted_1d_return')[['etf', 'predicted_1d_return', 'predictability_index']].to_dict('records'),
        'all_etfs': signals
    }
    
    return output


def main():
    config = load_config()
    model, device = load_model(config)
    
    # Generate for today or specified date
    today = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Generating Koopman-Spectral signals for {today}...")
    signals = generate_daily_signals(today, model, device, config)
    
    # Save
    output_dir = Path(config['signals']['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"koopman_signals_{today}.json"
    with open(output_file, 'w') as f:
        json.dump(signals, f, indent=2)
    
    print(f"Signals saved to {output_file}")
    print(f"Primary pick: {signals['primary_pick']['etf']} "
          f"({signals['primary_pick']['conviction_pct']}% conviction)")
    print(f"Koopman regime: {signals['koopman_modes']['regime']}")


if __name__ == "__main__":
    main()
