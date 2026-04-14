"""
Koopman-Spectral model core.
DMD initialization + MLP encoder + learnable linear Koopman operator.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import eig, lstsq


class DMDBaseline:
    """Dynamic Mode Decomposition for initialization."""
    
    @staticmethod
    def fit(X, Y, rank=64):
        """
        X: [N, T, D] — trajectories
        Y: [N, T, D] — advanced trajectories (X shifted)
        Returns: Koopman matrix K [rank, rank], encoder/decoder approximations
        """
        # Flatten time: treat each timestep as sample
        X_flat = X.reshape(-1, X.shape[-1])  # [N*T, D]
        Y_flat = Y.reshape(-1, Y.shape[-1])
        
        # SVD for rank reduction
        U, s, Vh = np.linalg.svd(X_flat.T, full_matrices=False)
        U_r = U[:, :rank]
        
        # Project to rank-dim space
        X_proj = X_flat @ U_r  # [N*T, rank]
        Y_proj = Y_flat @ U_r
        
        # Least squares for K: Y_proj ≈ X_proj @ K.T
        K, _, _, _ = lstsq(X_proj, Y_proj)
        K = K.T  # [rank, rank]
        
        return torch.FloatTensor(K), torch.FloatTensor(U_r)


class MLPEncoder(nn.Module):
    """MLP to Koopman observables."""
    
    def __init__(self, input_dim, observable_dim, hidden_dims, activation='relu'):
        super().__init__()
        self.input_dim = input_dim
        
        layers = []
        prev_dim = input_dim
        act = nn.ReLU() if activation == 'relu' else nn.Tanh()
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                act,
                nn.Dropout(0.1)
            ])
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, observable_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        x: [batch, time, features] or [batch, features]
        Returns: [batch, time, observable_dim] or [batch, observable_dim]
        """
        if x.ndim == 3:
            B, T, D = x.shape
            x = x.reshape(B * T, D)
            z = self.net(x)
            return z.reshape(B, T, -1)
        else:
            return self.net(x)


class KoopmanSpectral(nn.Module):
    """
    Full Koopman model with learnable linear dynamics.
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.observable_dim = config['model']['observable_dim']
        input_features = len(config['data']['macro_features']) + 3  # returns, lag, vol
        
        self.encoder = MLPEncoder(
            input_dim=input_features,
            observable_dim=self.observable_dim,
            hidden_dims=config['model']['encoder_hidden'],
            activation=config['model']['encoder_activation']
        )
        
        # Learnable Koopman operator (initialized via DMD)
        self.K = nn.Parameter(torch.randn(self.observable_dim, self.observable_dim) * 0.01)
        
        # Readout head: observables → return prediction
        self.readout = nn.Sequential(
            nn.Linear(self.observable_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Predict next-day return
        )
        
        # Eigenvalue buffer (computed after forward)
        self.register_buffer('eigenvalues', torch.zeros(self.observable_dim, dtype=torch.complex64))
    
    def initialize_with_dmd(self, dmd_K):
        """Warm start from DMD solution."""
        with torch.no_grad():
            self.K.copy_(dmd_K)
    
    def forward(self, X, return_modes=False):
        """
        X: [batch, time, features] — input trajectory
        Returns: predictions + optional Koopman modes
        """
        batch_size, T, _ = X.shape
        
        # Encode to observable space
        Z = self.encoder(X)  # [batch, time, K]
        
        # Apply Koopman operator: linear dynamics in latent space
        # Z_next = Z @ K.T
        Z_last = Z[:, -1, :]  # [batch, K] — final observable state
        
        # Multi-step prediction via Koopman operator power
        Z_future = []
        Z_current = Z_last
        for _ in range(5):  # Predict 5 days ahead
            Z_current = Z_current @ self.K.T  # Linear evolution
            Z_future.append(Z_current)
        
        Z_future = torch.stack(Z_future, dim=1)  # [batch, 5, K]
        
        # Decode to returns
        returns_pred = self.readout(Z_future.reshape(-1, self.observable_dim))
        returns_pred = returns_pred.reshape(batch_size, 5)
        
        if return_modes:
            # Compute eigen-decomposition for interpretability
            with torch.no_grad():
                eigs, _ = torch.linalg.eig(self.K)
                self.eigenvalues = eigs
            
            # Classify modes
            modes = self._classify_modes(eigs)
            return returns_pred, modes, Z_future
        
        return returns_pred
    
    def _classify_modes(self, eigenvalues):
        """
        Classify Koopman eigenvalues:
        - Growth: |λ| > 1 (unstable/expanding)
        - Oscillatory: Im(λ) ≠ 0 (rotating)
        - Decay: |λ| < 1 (stable/contracting)
        """
        magnitudes = torch.abs(eigenvalues)
        angles = torch.angle(eigenvalues)
        
        growth_mask = magnitudes > 1.05
        decay_mask = magnitudes < 0.95
        osc_mask = torch.abs(angles) > 0.1
        
        return {
            'eigenvalues': eigenvalues,
            'magnitudes': magnitudes,
            'frequencies': angles / (2 * np.pi),  # Cycles per step
            'growth_count': growth_mask.sum().item(),
            'oscillatory_count': (osc_mask & ~growth_mask & ~decay_mask).sum().item(),
            'decay_count': decay_mask.sum().item(),
            'spectral_gap': (magnitudes.max() - magnitudes[magnitudes < 1].max()).item() if (magnitudes < 1).any() else 0.0
        }
    
    def koopman_loss(self, X, Y_true, config):
        """
        Combined loss: prediction + linearity + spectral penalty
        """
        # Encode full trajectory
        Z = self.encoder(X)  # [batch, time, K]
        
        # Linearity: Z[t+1] ≈ Z[t] @ K.T
        Z_in = Z[:, :-1, :].reshape(-1, self.observable_dim)  # All but last
        Z_out = Z[:, 1:, :].reshape(-1, self.observable_dim)   # All but first
        
        Z_pred = Z_in @ self.K.T
        linearity_loss = nn.MSELoss()(Z_pred, Z_out)
        
        # Prediction loss on returns
        returns_pred = self.forward(X)
        Y_returns = Y_true[:, :, 0]  # First feature is returns
        pred_loss = nn.MSELoss()(returns_pred, Y_returns)
        
        # Spectral penalty: encourage stable eigenvalues (|λ| < 1.2)
        with torch.no_grad():
            eigs = torch.linalg.eigvals(self.K)
            max_mag = torch.abs(eigs).max()
        
        spectral_penalty = torch.relu(max_mag - 1.2) ** 2
        
        total_loss = (config['model']['prediction_weight'] * pred_loss + 
                    config['model']['linearity_weight'] * linearity_loss +
                    config['model']['spectral_weight'] * spectral_penalty)
        
        return total_loss, {
            'pred_loss': pred_loss.item(),
            'linearity_loss': linearity_loss.item(),
            'spectral_penalty': spectral_penalty.item(),
            'max_eig_magnitude': max_mag.item()
        }
