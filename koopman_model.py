"""
Koopman-Spectral model core with LSTM sequence encoder and ETF embedding.

Fixes vs original:
  1. K initialised as identity (was 0.01 * I — collapsed z_next to near-zero).
  2. Learnable ETF embedding for per-ticker differentiation.
  3. dmd_warm_start() actually implemented via least-squares DMD.
"""

import torch
import torch.nn as nn
from typing import Optional


class SequenceEncoder(nn.Module):
    """LSTM encoder — processes full lookback window, returns final hidden state."""

    def __init__(self, input_dim: int, hidden_dim: int,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, lookback, features] -> [batch, hidden_dim]"""
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]


class KoopmanSpectral(nn.Module):
    """
    Koopman model with LSTM encoder, learnable linear Koopman operator,
    and optional per-ETF embedding.
    """

    def __init__(self, config: dict):
        super().__init__()

        input_dim    = config['model']['input_dim']
        self.obs_dim = config['model']['observable_dim']
        dropout      = config['model'].get('dropout', 0.1)
        num_etfs     = config['model'].get('num_etfs', 0)
        etf_emb_dim  = config['model'].get('etf_emb_dim', 16)

        self.encoder = SequenceEncoder(
            input_dim=input_dim,
            hidden_dim=self.obs_dim,
            num_layers=config['model'].get('lstm_layers', 2),
            dropout=dropout,
        )

        # Per-ETF embedding so the shared model learns ticker-specific priors
        self.use_etf_emb = num_etfs > 0
        if self.use_etf_emb:
            self.etf_embedding = nn.Embedding(num_etfs, etf_emb_dim)
            koopman_in = self.obs_dim + etf_emb_dim
        else:
            koopman_in = self.obs_dim
        self.koopman_in = koopman_in

        # FIX: identity init — was 0.01 * I which made z_next ~= 0 for all inputs
        self.K = nn.Parameter(torch.eye(koopman_in))

        self.readout = nn.Sequential(
            nn.Linear(koopman_in, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor,
                etf_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x:       [batch, lookback, features]
        etf_idx: [batch] int64  (required when use_etf_emb=True)
        Returns: [batch, 1] predicted next-day return
        """
        z = self.encoder(x)
        if self.use_etf_emb:
            if etf_idx is None:
                raise ValueError("etf_idx required when use_etf_emb=True")
            z = torch.cat([z, self.etf_embedding(etf_idx)], dim=-1)
        z_next = z @ self.K.T
        return self.readout(z_next)

    def get_latent(self, x: torch.Tensor,
                   etf_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        z = self.encoder(x)
        if self.use_etf_emb and etf_idx is not None:
            z = torch.cat([z, self.etf_embedding(etf_idx)], dim=-1)
        return z

    def get_eigenvalues(self) -> torch.Tensor:
        with torch.no_grad():
            return torch.linalg.eigvals(self.K)

    @torch.no_grad()
    def dmd_warm_start(self, Z: torch.Tensor):
        """
        Initialise K from latent data via least-squares DMD.
        Z: [T, koopman_in] — sequence of encoder outputs on the training set.
        """
        try:
            Z0, Z1 = Z[:-1], Z[1:]
            K_init, _, _, _ = torch.linalg.lstsq(Z0, Z1)
            if torch.isfinite(K_init).all():
                self.K.copy_(K_init.T)
                print("DMD warm-start applied.")
            else:
                print("DMD warm-start produced non-finite K — keeping identity.")
        except Exception as e:
            print(f"DMD warm-start failed ({e}) — keeping identity.")


if __name__ == "__main__":
    cfg = {
        'model': {'input_dim': 15, 'observable_dim': 64,
                  'lstm_layers': 2, 'dropout': 0.1,
                  'num_etfs': 21, 'etf_emb_dim': 16},
        'data': {'lookback_window': 63},
    }
    model = KoopmanSpectral(cfg)
    x   = torch.randn(4, 63, 15)
    idx = torch.tensor([0, 3, 7, 20])
    print("Output shape:", model(x, etf_idx=idx).shape)
    print("K eig mags (first 5):", torch.abs(model.get_eigenvalues()[:5]))
