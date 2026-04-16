"""
Training script for Koopman-Spectral engine with LSTM sequence encoder.

Fixes applied:
  1. Passes etf_idx to model so ETF embedding is trained.
  2. Fits StandardScaler on training set and saves it with the checkpoint.
  3. Calls dmd_warm_start() when config dmd_init=true.
  4. Saves num_etfs and etf_to_idx in the checkpoint so generate_signals.py
     can rebuild the model correctly.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from datetime import datetime

from data_loader import load_config, build_dataset_tensors, save_scaler
from koopman_model import KoopmanSpectral


def main():
    print(f"=== Koopman-Spectral Training Started {datetime.now()} ===")

    config = load_config()

    num_macro = len(config['data']['macro_features'])
    config['model']['input_dim'] = 2 + num_macro
    print(f"Input dimension: {config['model']['input_dim']} "
          f"(returns + vol + {num_macro} macro features)")

    # ── Load data ──────────────────────────────────────────────────────────
    # FIX: fit_scaler=True on training set; scaler returned and saved.
    X_train, y_train, idx_train, feat_names, etf_to_idx, scaler = \
        build_dataset_tensors(config, split='train', fit_scaler=True)

    # Validation uses the same scaler (no re-fitting).
    X_val, y_val, idx_val, _, _, _ = \
        build_dataset_tensors(config, split='val', scaler=scaler, fit_scaler=False)

    if X_train is None or len(X_train) == 0:
        print("ERROR: No training data available.")
        return

    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")
    print(f"Input features: {feat_names}")
    print(f"Input shape: {X_train.shape[1:]} (lookback, features)")
    print(f"ETF universe size: {len(etf_to_idx)}")

    # ── Configure model with ETF embedding ────────────────────────────────
    config['model']['num_etfs']    = len(etf_to_idx)
    config['model']['etf_emb_dim'] = config['model'].get('etf_emb_dim', 16)

    # ── DataLoaders ────────────────────────────────────────────────────────
    bs = config['training']['batch_size']
    train_loader = DataLoader(
        TensorDataset(X_train, y_train, idx_train), batch_size=bs, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val, idx_val), batch_size=bs, shuffle=False
    )

    model     = KoopmanSpectral(config)
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.MSELoss()

    # ── Optional DMD warm-start ────────────────────────────────────────────
    if config['training'].get('dmd_init', False):
        print("Running DMD warm-start on training latents …")
        model.eval()
        latents = []
        with torch.no_grad():
            for X_batch, _, idx_batch in train_loader:
                z = model.get_latent(X_batch, etf_idx=idx_batch)
                latents.append(z)
        Z = torch.cat(latents, dim=0)
        model.dmd_warm_start(Z)

    # ── Training loop ──────────────────────────────────────────────────────
    epochs          = config['training']['epochs']
    patience        = config['training']['patience']
    best_val_loss   = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch, idx_batch in train_loader:
            pred   = model(X_batch, etf_idx=idx_batch)   # [batch, 1]
            target = y_batch[:, 0:1]                      # next-day return
            loss   = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch, idx_batch in val_loader:
                pred   = model(X_batch, etf_idx=idx_batch)
                target = y_batch[:, 0:1]
                val_loss += criterion(pred, target).item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(
                {
                    'epoch':             epoch,
                    'model_state_dict':  model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss':          val_loss,
                    'config':            config,
                    'etf_to_idx':        etf_to_idx,   # needed at inference
                    'num_etfs':          len(etf_to_idx),
                },
                'koopman_spectral_best.pt',
            )
            print(f"Epoch {epoch+1}: new best val_loss={val_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    # Save scaler separately so generate_signals.py can load it
    save_scaler(scaler, "koopman_scaler.pkl")

    print(f"=== Training Complete {datetime.now()} ===")
    print(f"Best val loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
