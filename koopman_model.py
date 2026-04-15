"""
Koopman-Spectral model core.
DMD initialization + MLP encoder + learnable linear Koopman operator.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import lstsq
from typing import Tuple, Dict, Optional


class DMDBaseline:
    """Dynamic Mode Decomposition for warm initialization of Koopman operator."""
    
    @staticmethod
    def fit(observables: np.ndarray, observables_next: np.ndarray, rank: int) -> torch.Tensor:
        """
        Compute DMD matrix K such that observables_next ≈ observables @ K.T
        
        Args:
            observables: [N, K]  (N samples, K observables)
            observables_next: [N, K]
            rank: truncation rank (must be <= K)
        
        Returns:
            K: [rank, rank] torch float tensor
        """
        # Ensure rank is not larger than observable dimension
        rank = min(rank, observables.shape[1])
        
        # Compute SVD of observables
        U, s, Vh = np.linalg.svd(observables, full_matrices=False)
        # U: [N, N], Vh: [K, K]
        
        # Truncate to rank
        U_r = U[:, :rank]        # [N, rank]
        Vh_r = Vh[:rank, :]      # [rank, K]
        s_r = s[:rank]
        
        # Project data onto rank-r subspace
        # Instead of using U_r, we use Vh_r for projection: observables_proj = observables @ Vh_r.T
        observables_proj = observables @ Vh_r.T   # [N, rank]
        observables_next_proj = observables_next @ Vh_r.T
        
        # Compute reduced Koopman matrix K_r = observables_proj \ observables_next_proj
        K_r, _, _, _ = lstsq(observables_proj, observables_next_proj)
        K_r = K_r.T  # [rank, rank]
        
        return torch.tensor(K_r, dtype=torch.float32)


class MLPEncoder(nn.Module):
    """MLP that maps raw features to Koopman observables."""
    
    def __init__(self, input_dim: int, observable_dim: int, hidden_dims: list, dropout: float = 0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, observable_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, features] -> [batch, observable_dim]"""
        return self.net(x)


class KoopmanSpectral(nn.Module):
    """
    Koopman autoencoder with learnable linear dynamics and spectral analysis.
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        self.obs_dim = config['model']['observable_dim']
        self.input_dim = config['model'].get('input_dim', 2)
        hidden_dims = config['model']['encoder_hidden']
        dropout = config['model'].get('dropout', 0.1)
        
        self.encoder = MLPEncoder(self.input_dim, self.obs_dim, hidden_dims, dropout)
        
        # Learnable Koopman operator (will be initialized via DMD if enabled)
        self.K = nn.Parameter(torch.eye(self.obs_dim) * 0.01)
        
        # Readout: observables -> 1-day return prediction
        self.readout = nn.Sequential(
            nn.Linear(self.obs_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    
    def initialize_with_dmd(self, dmd_K: torch.Tensor):
        """Warm start the Koopman matrix from DMD solution."""
        with torch.no_grad():
            if dmd_K.shape != self.K.shape:
                min_rows = min(dmd_K.shape[0], self.K.shape[0])
                min_cols = min(dmd_K.shape[1], self.K.shape[1])
                self.K[:min_rows, :min_cols] = dmd_K[:min_rows, :min_cols]
            else:
                self.K.copy_(dmd_K)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Predict next return from current observable.
        z: [batch, obs_dim]
        Returns: next_return [batch, 1]
        """
        z_next = z @ self.K.T
        r_next = self.readout(z_next)
        return r_next
    
    def compute_koopman_loss(self, X: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss: prediction MSE + linearity constraint + spectral penalty.
        X: [batch, lookback, features]
        y: [batch, target_horizon] (returns)
        """
        batch_size, T, D = X.shape
        flat = X.view(-1, D)
        z_all = self.encoder(flat).view(batch_size, T, -1)
        
        # Linearity loss
        z_in = z_all[:, :-1, :].reshape(-1, self.obs_dim)
        z_out = z_all[:, 1:, :].reshape(-1, self.obs_dim)
        z_pred = z_in @ self.K.T
        linearity_loss = nn.MSELoss()(z_pred, z_out)
        
        # Prediction loss
        z_last = z_all[:, -1, :]
        pred_returns = self.forward(z_last)
        target = y[:, 0:1]
        pred_loss = nn.MSELoss()(pred_returns, target)
        
        # Spectral penalty
        with torch.no_grad():
            eigs = torch.linalg.eigvals(self.K)
            max_mag = torch.abs(eigs).max()
        spectral_penalty = torch.relu(max_mag - 1.2) ** 2
        
        total_loss = (config['model']['prediction_weight'] * pred_loss +
                      config['model']['linearity_weight'] * linearity_loss +
                      config['model']['spectral_weight'] * spectral_penalty)
        
        return {
            'loss': total_loss,
            'pred_loss': pred_loss,
            'linearity_loss': linearity_loss,
            'spectral_penalty': spectral_penalty,
            'max_eig_magnitude': max_mag
        }
