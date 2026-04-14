"""
Training script for Koopman-Spectral engine.
Optimized for GitHub Actions 6-hour limit.
Uses HF dataset: P2SAMAPA/p2-etf-deepm-data/data/master.parquet
"""

import torch
import numpy as np
from pathlib import Path
import yaml
import json
from datetime import datetime

from data_loader import load_config, build_dataset, HFDataLoader
from koopman_model import KoopmanSpectral, DMDBaseline


def collate_samples(samples):
    """Batch numpy samples into tensors."""
    if not samples:
        return None, None, []
    
    X = np.stack([s['X'] for s in samples])
    Y = np.stack([s['Y'] for s in samples])
    etfs = [s['etf'] for s in samples]
    return torch.FloatTensor(X), torch.FloatTensor(Y), etfs


def train_epoch(model, train_samples, optimizer, config, device):
    model.train()
    epoch_loss = 0
    metrics_sum = {}
    batch_size = config['training']['batch_size']
    
    indices = np.random.permutation(len(train_samples))
    
    for i in range(0, len(indices), batch_size):
        batch_idx = indices[i:i+batch_size]
        batch_samples = [train_samples[j] for j in batch_idx]
        
        X, Y, _ = collate_samples(batch_samples)
        if X is None:
            continue
            
        X, Y = X.to(device), Y.to(device)
        
        optimizer.zero_grad()
        loss, metrics = model.koopman_loss(X, Y, config)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
        for k, v in metrics.items():
            metrics_sum[k] = metrics_sum.get(k, 0) + v
    
    n_batches = max(1, len(indices) // batch_size)
    return epoch_loss / n_batches, {k: v/n_batches for k, v in metrics_sum.items()}


def validate(model, val_samples, config, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_actuals = []
    batch_size = config['training']['batch_size']
    
    with torch.no_grad():
        for i in range(0, len(val_samples), batch_size):
            batch = val_samples[i:i+batch_size]
            X, Y, _ = collate_samples(batch)
            if X is None:
                continue
                
            X, Y = X.to(device), Y.to(device)
            
            preds = model(X)
            loss = torch.nn.MSELoss()(preds, Y[:, :, 0])
            total_loss += loss.item() * len(batch)
            
            all_preds.extend(preds[:, 0].cpu().numpy())
            all_actuals.extend(Y[:, 0, 0].cpu().numpy())
    
    if len(val_samples) == 0:
        return {'val_mse': float('inf'), 'val_mae': float('inf'), 'val_dir_acc': 0}
    
    mse = total_loss / len(val_samples)
    mae = np.mean(np.abs(np.array(all_preds) - np.array(all_actuals)))
    
    pred_sign = np.sign(np.array(all_preds))
    actual_sign = np.sign(np.array(all_actuals))
    dir_acc = (pred_sign == actual_sign).mean() if len(pred_sign) > 0 else 0
    
    return {'val_mse': mse, 'val_mae': mae, 'val_dir_acc': dir_acc}


def main():
    print(f"=== Koopman-Spectral Training Started {datetime.now()} ===")
    print("Data source: HF: P2SAMAPA/p2-etf-deepm-data/data/master.parquet")
    
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Verify data access
    loader = HFDataLoader(use_local=True)
    master = loader.load_master()
    
    if master.empty:
        print("ERROR: Cannot load master.parquet")
        print("Checked:")
        print(f"  Local: /mnt/data/p2-etf-deepm-data/data/master.parquet")
        print(f"  HF Hub: {loader.BASE_URL}/master.parquet")
        return
    
    print(f"Loaded master.parquet: {len(master)} rows, {len(loader.get_columns())} columns")
    print(f"ETFs in dataset: {len(loader.get_all_etfs())}")
    
    # Load data
    print("Building training dataset...")
    try:
        train_samples = build_dataset(config, 'train')
        val_samples = build_dataset(config, 'val')
        print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")
    except Exception as e:
        print(f"Error building dataset: {e}")
        return
    
    if len(train_samples) == 0:
        print("ERROR: No training samples generated")
        return
    
    # Initialize model
    model = KoopmanSpectral(config).to(device)
    
    # DMD initialization
    if config['training'].get('dmd_init', True) and len(train_samples) > 100:
        print("Computing DMD initialization...")
        subset_size = min(5000, len(train_samples))
        subset = np.random.choice(len(train_samples), subset_size, replace=False)
        
        X_sub = np.stack([train_samples[i]['X'] for i in subset])
        Y_sub = np.stack([train_samples[i]['Y'] for i in subset])
        
        # Flatten for DMD
        X_flat = X_sub.reshape(-1, X_sub.shape[-1])
        Y_flat = Y_sub.reshape(-1, Y_sub.shape[-1])
        
        dmd = DMDBaseline()
        K_init, _ = dmd.fit(X_flat, Y_flat, rank=config['model']['observable_dim'])
        model.initialize_with_dmd(K_init.to(device))
        print("DMD initialization complete")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    history = []
    max_epochs = config['training']['epochs']
    
    for epoch in range(max_epochs):
        train_loss, train_metrics = train_epoch(model, train_samples, optimizer, config, device)
        val_metrics = validate(model, val_samples, config, device)
        
        scheduler.step(val_metrics['val_mse'])
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            **train_metrics,
            **val_metrics
        })
        
        if epoch % 10 == 0 or epoch < 5:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                  f"val_mse={val_metrics['val_mse']:.4f}, "
                  f"val_dir_acc={val_metrics['val_dir_acc']:.3f}")
        
        # Early stopping
        if val_metrics['val_mse'] < best_val_loss:
            best_val_loss = val_metrics['val_mse']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'val_metrics': val_metrics
            }, 'koopman_spectral_best.pt')
        else:
            patience_counter += 1
        
        if patience_counter >= config['training'].get('patience', 20):
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Save final
    torch.save(model.state_dict(), 'koopman_spectral_final.pt')
    
    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"=== Training Complete {datetime.now()} ===")
    print(f"Best val MSE: {best_val_loss:.6f}")
    print(f"Model saved: koopman_spectral_best.pt")


if __name__ == "__main__":
    main()
