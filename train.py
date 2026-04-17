"""
Training script for Koopman-Spectral engine.
Designed to run safely within GitHub Actions (7 GB RAM, CPU only).

Key CI safeguards:
  - max_samples cap so the dataset fits in memory
  - Streams dataset in chunks rather than building one giant array
  - Explicit memory logging so you can see where RAM is going
  - timeout-aware: saves best checkpoint even if job is cancelled mid-run
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
import signal
import sys
import os

from data_loader import load_config, build_dataset_tensors, save_scaler
from koopman_model import KoopmanSpectral


# ── Graceful shutdown on SIGTERM (Actions cancellation) ───────────────────────
_best_model_state = None
_best_val         = float('inf')
_model_ref        = None

def _save_on_signal(sig, frame):
    print(f"\nSIGTERM received — saving best checkpoint before exit …")
    if _best_model_state is not None:
        torch.save(_best_model_state, 'koopman_spectral_best.pt')
        print("Checkpoint saved.")
    sys.exit(0)

signal.signal(signal.SIGTERM, _save_on_signal)


def _ram_mb() -> str:
    try:
        import psutil
        return f"{psutil.Process().memory_info().rss / 1e6:.0f} MB"
    except ImportError:
        return "unknown"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global _best_model_state, _best_val

    print(f"=== Koopman-Spectral Training Started {datetime.now()} ===")
    print(f"RAM at start: {_ram_mb()}")

    config = load_config()

    # ── CI memory guard ───────────────────────────────────────────────────
    # GitHub free runners: 7 GB RAM, 2 vCPUs.
    # Cap samples so tensor build stays well under 4 GB.
    ci_mode     = os.environ.get("CI", "false").lower() == "true"
    max_samples = int(os.environ.get("MAX_SAMPLES", 50_000 if ci_mode else 500_000))
    print(f"CI mode: {ci_mode} | max_samples: {max_samples:,}")

    num_macro = len(config['data']['macro_features'])
    config['model']['input_dim'] = 2 + num_macro
    print(f"Input dim: {config['model']['input_dim']}")

    # ── Build datasets ────────────────────────────────────────────────────
    print(f"Building training tensors … RAM: {_ram_mb()}")
    result = build_dataset_tensors(
        config, split='train',
        fit_scaler=True,
        max_samples=max_samples,
    )
    X_tr, y_tr, idx_tr, feat_names, etf_to_idx, scaler = result

    if X_tr is None or len(X_tr) == 0:
        print("ERROR: No training data — check that master.parquet is accessible.")
        sys.exit(1)

    print(f"Train samples: {len(X_tr):,}  |  RAM: {_ram_mb()}")

    result_val = build_dataset_tensors(
        config, split='val',
        scaler=scaler,
        max_samples=max_samples // 4,
    )
    X_vl, y_vl, idx_vl, _, _, _ = result_val
    print(f"Val samples:   {len(X_vl):,}  |  RAM: {_ram_mb()}")

    # ── Model ─────────────────────────────────────────────────────────────
    config['model']['num_etfs']    = len(etf_to_idx)
    config['model']['etf_emb_dim'] = config['model'].get('etf_emb_dim', 16)

    # Reduce model size in CI to stay within RAM
    if ci_mode:
        config['model']['observable_dim'] = min(config['model'].get('observable_dim', 128), 64)
        print(f"CI: observable_dim capped at {config['model']['observable_dim']}")

    bs = config['training']['batch_size']
    train_loader = DataLoader(TensorDataset(X_tr, y_tr, idx_tr),
                              batch_size=bs, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(TensorDataset(X_vl, y_vl, idx_vl),
                              batch_size=bs, shuffle=False,
                              num_workers=0, pin_memory=False)

    model     = KoopmanSpectral(config)
    optimizer = optim.Adam(model.parameters(),
                           lr=config['training']['learning_rate'])
    criterion = nn.MSELoss()

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}  |  RAM: {_ram_mb()}")

    # ── Optional DMD warm-start ───────────────────────────────────────────
    if config['training'].get('dmd_init', False):
        print("Running DMD warm-start …")
        model.eval()
        latents = []
        with torch.no_grad():
            for Xb, _, ib in train_loader:
                latents.append(model.get_latent(Xb, etf_idx=ib))
                if len(latents) * bs > 10_000:   # cap DMD data
                    break
        model.dmd_warm_start(torch.cat(latents, dim=0))

    # ── Training loop ─────────────────────────────────────────────────────
    epochs       = config['training']['epochs']
    patience     = config['training']['patience']
    patience_ctr = 0

    # In CI, cap epochs so the job completes within ~4 hours
    if ci_mode:
        epochs  = min(epochs,  200)
        patience = min(patience, 30)
        print(f"CI: epochs capped at {epochs}, patience at {patience}")

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

        if val_loss < _best_val:
            _best_val    = val_loss
            patience_ctr = 0
            _best_model_state = {
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss':             val_loss,
                'config':               config,
                'etf_to_idx':           etf_to_idx,
                'num_etfs':             len(etf_to_idx),
            }
            torch.save(_best_model_state, 'koopman_spectral_best.pt')
            print(f"Epoch {epoch+1:4d}: new best val={val_loss:.6f}  RAM={_ram_mb()}")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:4d}/{epochs} | "
                  f"train={train_loss:.6f} | val={val_loss:.6f} | RAM={_ram_mb()}")

    save_scaler(scaler, "koopman_scaler.pkl")
    print(f"=== Training Complete {datetime.now()} | best val={_best_val:.6f} ===")


if __name__ == "__main__":
    main()
