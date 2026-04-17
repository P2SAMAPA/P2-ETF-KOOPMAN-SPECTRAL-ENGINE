"""
Training script for Koopman-Spectral engine.

Fixes vs original:
  1. Passes etf_idx to model so ETF embedding is trained.
  2. Fits and saves StandardScaler with the checkpoint.
  3. Saves etf_to_idx in checkpoint for use at inference.
  4. Optional DMD warm-start.
  5. Gradient clipping for LSTM stability.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime

from data_loader import load_config, build_dataset_tensors, save_scaler
from koopman_model import KoopmanSpectral


def main():
    print(f"=== Koopman-Spectral Training Started {datetime.now()} ===")
    config = load_config()

    num_macro = len(config['data']['macro_features'])
    config['model']['input_dim'] = 2 + num_macro
    print(f"Input dim: {config['model']['input_dim']} "
          f"(returns + vol + {num_macro} macro)")

    # Load & normalise — fit scaler on train only
    X_tr, y_tr, idx_tr, feat_names, etf_to_idx, scaler = \
        build_dataset_tensors(config, split='train', fit_scaler=True)

    X_vl, y_vl, idx_vl, _, _, _ = \
        build_dataset_tensors(config, split='val', scaler=scaler)

    if X_tr is None or len(X_tr) == 0:
        print("ERROR: No training data.")
        return

    print(f"Train: {len(X_tr)}  Val: {len(X_vl)}")
    print(f"Features: {feat_names}")
    print(f"ETF universe ({len(etf_to_idx)}): {list(etf_to_idx.keys())}")

    # Wire ETF count into model config
    config['model']['num_etfs']    = len(etf_to_idx)
    config['model']['etf_emb_dim'] = config['model'].get('etf_emb_dim', 16)

    bs = config['training']['batch_size']
    train_loader = DataLoader(TensorDataset(X_tr, y_tr, idx_tr),
                              batch_size=bs, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_vl, y_vl, idx_vl),
                              batch_size=bs, shuffle=False)

    model     = KoopmanSpectral(config)
    optimizer = optim.Adam(model.parameters(),
                           lr=config['training']['learning_rate'])
    criterion = nn.MSELoss()

    # Optional DMD warm-start
    if config['training'].get('dmd_init', False):
        print("Running DMD warm-start …")
        model.eval()
        latents = []
        with torch.no_grad():
            for Xb, _, ib in train_loader:
                latents.append(model.get_latent(Xb, etf_idx=ib))
        model.dmd_warm_start(torch.cat(latents, dim=0))

    epochs, patience = config['training']['epochs'], config['training']['patience']
    best_val, patience_ctr = float('inf'), 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for Xb, yb, ib in train_loader:
            pred = model(Xb, etf_idx=ib)
            loss = criterion(pred, yb[:, 0:1])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * Xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb, ib in val_loader:
                val_loss += criterion(model(Xb, etf_idx=ib),
                                      yb[:, 0:1]).item() * Xb.size(0)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val:
            best_val, patience_ctr = val_loss, 0
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss':             val_loss,
                'config':               config,
                'etf_to_idx':           etf_to_idx,
                'num_etfs':             len(etf_to_idx),
            }, 'koopman_spectral_best.pt')
            print(f"Epoch {epoch+1}: new best val_loss={val_loss:.6f}")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"train={train_loss:.6f} | val={val_loss:.6f}")

    save_scaler(scaler, "koopman_scaler.pkl")
    print(f"=== Training Complete {datetime.now()} | best val={best_val:.6f} ===")


if __name__ == "__main__":
    main()
