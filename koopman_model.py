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
            rank: truncation rank
        
        Returns:
            K: [rank, rank] torch float tensor
        """
        # SVD of observables
        U, s, Vh = np.linalg.svd(observables, full_matrices=False)
        U_r = U[:, :rank]
        s_r = s[:rank]
        
        # Project to rank-r space
        observables_proj = observables @ U_r  # [N, rank]
        observables_next_proj = observables_next @ U_r
        
        # Least squares for K_proj: observables_next_proj ≈ observables_proj @ K_proj.T
        K_proj, _, _, _ = lstsq(observables_proj, observables_next_proj)
        K_proj = K_proj.T  # [rank, rank]
        
        # Transform back to original basis if needed? Not necessary; we'll use K_proj directly.
        return torch.tensor(K_proj, dtype=torch.float32)


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
        # Input dimension: number of features per timestep (e.g., returns + volatility)
        # This should be set in config or inferred from data. We'll read from config.
        self.input_dim = config['model'].get('input_dim', 2)  # default: returns and vol
        hidden_dims = config['model']['encoder_hidden']
        dropout = config['model'].get('dropout', 0.1)
        
        self.encoder = MLPEncoder(self.input_dim, self.obs_dim, hidden_dims, dropout)
        
        # Learnable Koopman operator (will be initialized via DMD)
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
            # Ensure dimensions match
            if dmd_K.shape != self.K.shape:
                # Pad or truncate
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
        # Linear dynamics: z_next = z @ K.T
        z_next = z @ self.K.T
        r_next = self.readout(z_next)
        return r_next
    
    def compute_koopman_loss(self, X: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss: prediction MSE + linearity constraint + spectral penalty.
        X: [batch, lookback, features]
        y: [batch, target_horizon] (returns)
        """
        # Encode all timesteps
        batch_size, T, D = X.shape
        flat = X.view(-1, D)
        z_all = self.encoder(flat).view(batch_size, T, -1)  # [batch, T, obs_dim]
        
        # Linearity loss: z[t+1] ≈ z[t] @ K.T
        z_in = z_all[:, :-1, :].reshape(-1, self.obs_dim)
        z_out = z_all[:, 1:, :].reshape(-1, self.obs_dim)
        z_pred = z_in @ self.K.T
        linearity_loss = nn.MSELoss()(z_pred, z_out)
        
        # Prediction loss: from last observable to next return
        z_last = z_all[:, -1, :]
        pred_returns = self.forward(z_last)  # [batch, 1]
        target = y[:, 0:1]  # next day return
        pred_loss = nn.MSELoss()(pred_returns, target)
        
        # Spectral penalty: encourage eigenvalues within unit circle
        with torch.no_grad():
            eigs = torch.linalg.eigvals(self.K)
            max_mag = torch.abs(eigs).max()
        spectral_penalty = torch.relu(max_mag - 1.2) ** 2
        
        # Weighted sum
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
