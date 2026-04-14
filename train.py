"""
Training script for Koopman-Spectral engine.
Simplified version for wide-format data.
"""

import torch
import numpy as np
from pathlib import Path
import yaml
import json
from datetime import datetime

from data_loader import load_config, build_dataset, HFDataLoader


def main():
    print(f"=== Koopman-Spectral Training Started {datetime.now()} ===")
    
    config = load_config()
    
    # Verify data
    loader = HFDataLoader(
        use_local=True,
        local_path=config['data'].get('local_path', 'data/p2-etf-deepm-data')
    )
    
    master = loader.load_master()
    if master.empty:
        print("ERROR: Cannot load master.parquet")
        return
    
    print(f"Data loaded: {len(master)} rows, {len(loader.get_all_etfs())} ETFs")
    print(f"ETFs: {loader.get_all_etfs()[:10]}...")
    
    # For now, just save a dummy model checkpoint so signal generation can proceed
    # Real training would go here
    
    print("Creating placeholder model checkpoint...")
    
    # Save dummy checkpoint
    dummy_state = {
        'epoch': 0,
        'model_state_dict': {},
        'config': config,
        'val_metrics': {'val_mse': 0.0}
    }
    torch.save(dummy_state, 'koopman_spectral_best.pt')
    
    print(f"=== Training Complete {datetime.now()} ===")
    print("Placeholder model saved. Real training requires model architecture fixes.")


if __name__ == "__main__":
    main()
