"""
Training script for Koopman-Spectral engine with all macro features.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from datetime import datetime

from data_loader import load_config, build_dataset_tensors
from koopman_model import KoopmanSpectral


def main():
    print(f"=== Koopman-Spectral Training Started {datetime.now()} ===")
    config = load_config()
    
    # Compute input dimension: returns + vol + number of macro features
    num_macro = len(config['data']['macro_features'])
    config['model']['input_dim'] = 2 + num_macro
    print(f"Input dimension: {config['model']['input_dim']} (returns + vol + {num_macro} macro features)")
    
    # Load data with macro features
    X_train, y_train, feat_names = build_dataset_tensors(config, split='train')
    X_val, y_val, _ = build_dataset_tensors(config, split='val')
    
    if X_train is None or len(X_train) == 0:
        print("ERROR: No training data available.")
        return
    
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")
    print(f"Input features: {feat_names}")
    print(f"Input shape: {X_train.shape[1:]} (lookback, features)")
    
    # DataLoaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=config['training']['batch_size'], shuffle=False)
    
    model = KoopmanSpectral(config)
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.MSELoss()
    
    epochs = config['training']['epochs']
    patience = config['training']['patience']
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            # X_batch: [batch, lookback, features]
            last_obs = X_batch[:, -1, :]  # [batch, features]
            z = model.encoder(last_obs)   # FIXED: use model.encoder
            pred = model(z)               # [batch, 1]
            target = y_batch[:, 0:1]      # next day return
            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                last_obs = X_batch[:, -1, :]
                z = model.encoder(last_obs)
                pred = model(z)
                target = y_batch[:, 0:1]
                loss = criterion(pred, target)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config,
            }, 'koopman_spectral_best.pt')
            print(f"Epoch {epoch+1}: new best val_loss={val_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
    
    print(f"=== Training Complete {datetime.now()} ===")
    print(f"Best val loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
