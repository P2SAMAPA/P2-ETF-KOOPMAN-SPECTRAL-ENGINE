"""
Training script for Koopman-Spectral engine.
Trains MLP encoder + linear Koopman operator + readout.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import yaml
import json
from datetime import datetime
import sys

from data_loader import load_config, build_dataset_tensors, HFDataLoader
from koopman_model import KoopmanSpectral, DMDBaseline


def main():
    print(f"=== Koopman-Spectral Training Started {datetime.now()} ===")
    
    # Load config
    config = load_config()
    
    # Load data as PyTorch tensors
    print("Loading training data...")
    X_train, y_train, feature_names = build_dataset_tensors(config, split='train')
    X_val, y_val, _ = build_dataset_tensors(config, split='val')
    
    if X_train is None or len(X_train) == 0:
        print("ERROR: No training data available.")
        return
    
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")
    print(f"Input shape: {X_train[0].shape}, Target shape: {y_train[0].shape}")
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Instantiate model
    model = KoopmanSpectral(config)
    
    # DMD initialization (optional)
    if config['training'].get('dmd_init', True):
        print("Computing DMD initialization...")
        # Collect all observables from training data
        all_obs = []
        all_obs_next = []
        with torch.no_grad():
            for X_batch, _ in train_loader:
                # X_batch: [batch, lookback, features]
                # Encode each timestep
                batch_size_curr = X_batch.shape[0]
                T = X_batch.shape[1]
                # Reshape to [batch*T, features]
                flat = X_batch.view(-1, X_batch.shape[-1])
                z = model.encoder(flat).detach().numpy()  # [batch*T, obs_dim]
                # Reshape back to [batch, T, obs_dim]
                z = z.reshape(batch_size_curr, T, -1)
                # z[t] -> z[t+1] pairs
                obs = z[:, :-1, :].reshape(-1, model.obs_dim)
                obs_next = z[:, 1:, :].reshape(-1, model.obs_dim)
                all_obs.append(obs)
                all_obs_next.append(obs_next)
        obs_all = np.concatenate(all_obs, axis=0)
        obs_next_all = np.concatenate(all_obs_next, axis=0)
        # Compute DMD
        dmd_K = DMDBaseline.fit(obs_all, obs_next_all, rank=model.obs_dim)
        model.initialize_with_dmd(dmd_K)
        print("DMD initialization complete.")
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.MSELoss()
    
    # Training loop
    epochs = config['training']['epochs']
    patience = config['training'].get('patience', 20)
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            # X_batch: [batch, lookback, features]
            # y_batch: [batch, target_horizon] (returns)
            optimizer.zero_grad()
            
            # Forward: predict returns from the last observable
            # Encode last timestep
            last_obs = X_batch[:, -1, :]  # [batch, features]
            z_last = model.encoder(last_obs)  # [batch, obs_dim]
            # Predict next returns
            pred = model(z_last)  # [batch, 1] (1-day return)
            # Target: first element of y_batch (next day return)
            target = y_batch[:, 0:1]  # [batch, 1]
            
            loss = criterion(pred, target)
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
                z_last = model.encoder(last_obs)
                pred = model(z_last)
                target = y_batch[:, 0:1]
                loss = criterion(pred, target)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config,
            }, 'koopman_spectral_best.pt')
            print(f"  -> New best model saved (val_loss={val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
    
    print(f"=== Training Complete {datetime.now()} ===")
    print(f"Best val loss: {best_val_loss:.6f} at epoch {best_epoch+1}")
    print("Model saved as koopman_spectral_best.pt")


if __name__ == "__main__":
    main()
