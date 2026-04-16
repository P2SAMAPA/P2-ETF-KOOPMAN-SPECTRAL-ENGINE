"""
Koopman-Spectral model core with LSTM sequence encoder and ETF embedding.

Fixes applied:
  1. K initialised as identity (was 0.01 * I — collapsed z_next to near-zero).
  2. Added learnable ETF embedding so per-ticker identity is preserved
     through the shared model.
  3. Added dmd_warm_start() utility so train.py can call it before gradient
     descent (mirrors the README claim).
  4. get_latent() now accepts optional etf_idx for consistency.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class SequenceEncoder(nn.Module):
    """
    LSTM encoder that processes the entire lookback window.
    Outputs final hidden state as the Koopman observable.
    """

    def __init__(self, input_dim: int, hidden_dim: int,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, lookback, features]
        Returns: [batch, hidden_dim]  (last-layer final hidden state)
        """
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]   # [batch, hidden_dim]


class KoopmanSpectral(nn.Module):
    """
    Koopman model with sequence encoder, learnable linear dynamics,
    and optional per-ETF embedding for cross-sectional differentiation.

    Key changes vs original:
      - K is initialised as identity (not 0.01 * I).
      - An ETF embedding table is added; if num_etfs > 0 the embedding
        is concatenated to the LSTM hidden state before the Koopman step.
      - The readout operates on the post-K latent, which now carries
        meaningful magnitude.
    """

    def __init__(self, config: dict):
        super().__init__()

        input_dim   = config['model']['input_dim']
        self.obs_dim = config['model']['observable_dim']
        self.lookback = config['data']['lookback_window']

        num_etfs   = config['model'].get('num_etfs', 0)
        etf_emb_dim = config['model'].get('etf_emb_dim', 16)
        dropout    = config['model'].get('dropout', 0.1)

        # ── Sequence encoder ──────────────────────────────────────────────
        self.encoder = SequenceEncoder(
            input_dim=input_dim,
            hidden_dim=self.obs_dim,
            num_layers=config['model'].get('lstm_layers', 2),
            dropout=dropout,
        )

        # ── Optional per-ETF embedding ────────────────────────────────────
        self.use_etf_emb = num_etfs > 0
        if self.use_etf_emb:
            self.etf_embedding = nn.Embedding(num_etfs, etf_emb_dim)
            koopman_in = self.obs_dim + etf_emb_dim
        else:
            koopman_in = self.obs_dim

        # ── Learnable Koopman operator ────────────────────────────────────
        # FIX: initialised as identity, not 0.01 * I.
        # A near-zero K makes z_next ≈ 0 for all inputs → identical readout.
        self.K = nn.Parameter(torch.eye(koopman_in))
        self.koopman_in = koopman_in

        # ── Readout: latent → 1-day return prediction ─────────────────────
        self.readout = nn.Sequential(
            nn.Linear(koopman_in, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor,
                etf_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x:       [batch, lookback, features]
        etf_idx: [batch] int64 ETF indices (required if use_etf_emb=True)
        Returns: [batch, 1] predicted next-day return
        """
        z = self.encoder(x)                          # [batch, obs_dim]

        if self.use_etf_emb:
            if etf_idx is None:
                raise ValueError("etf_idx required when use_etf_emb=True")
            emb = self.etf_embedding(etf_idx)        # [batch, etf_emb_dim]
            z = torch.cat([z, emb], dim=-1)          # [batch, obs_dim + etf_emb_dim]

        z_next = z @ self.K.T                        # [batch, koopman_in]
        return self.readout(z_next)                  # [batch, 1]

    # ── Utilities ─────────────────────────────────────────────────────────

    def get_latent(self, x: torch.Tensor,
                   etf_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return pre-K latent representation (for analysis / DMD init)."""
        z = self.encoder(x)
        if self.use_etf_emb and etf_idx is not None:
            emb = self.etf_embedding(etf_idx)
            z = torch.cat([z, emb], dim=-1)
        return z

    def get_eigenvalues(self) -> torch.Tensor:
        """Return eigenvalues of the current Koopman operator."""
        with torch.no_grad():
            return torch.linalg.eigvals(self.K)

    @torch.no_grad()
    def dmd_warm_start(self, Z: torch.Tensor):
        """
        Initialise K from data using Dynamic Mode Decomposition.

        Z: [T, koopman_in]  — sequence of latent states from the encoder
                              collected on the training set.

        Solves  K ≈ Z[1:].T @ pinv(Z[:-1].T)  (standard DMD).
        Falls back to identity if the solve is numerically ill-conditioned.
        """
        try:
            Z0 = Z[:-1]   # [T-1, d]
            Z1 = Z[1:]    # [T-1, d]
            # K* = Z1^T @ pinv(Z0^T)  =  (Z0 \ Z1)^T  via least-squares
            K_init, _, _, _ = torch.linalg.lstsq(Z0, Z1)
            if torch.isfinite(K_init).all():
                self.K.copy_(K_init.T)
                print("DMD warm-start applied successfully.")
            else:
                print("DMD warm-start produced non-finite K — keeping identity.")
        except Exception as e:
            print(f"DMD warm-start failed ({e}) — keeping identity.")


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = {
        'model': {
            'input_dim': 15,
            'observable_dim': 64,
            'lstm_layers': 2,
            'dropout': 0.1,
            'num_etfs': 22,
            'etf_emb_dim': 16,
        },
        'data': {'lookback_window': 63},
    }
    model = KoopmanSpectral(cfg)
    x = torch.randn(4, 63, 15)
    idx = torch.tensor([0, 3, 7, 21])
    y = model(x, etf_idx=idx)
    print("Output shape:", y.shape)          # [4, 1]
    eigs = model.get_eigenvalues()
    print("K eigenvalue magnitudes (first 5):", torch.abs(eigs[:5]))
