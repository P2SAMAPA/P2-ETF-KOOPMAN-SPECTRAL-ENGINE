"""
Koopman-Spectral model core with GRU sequence encoder.
Uses full lookback window to generate latent state.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple


class KoopmanSpectral(nn.Module):
    """
    Koopman model with sequence encoder (GRU) and linear Koopman operator.
    """
    
    def __init__(self, config: dict):
        super().__init__()
        self.obs_dim = config['model']['observable_dim']
        self.input_dim = config['model']['input_dim']
        hidden_dims = config['model']['encoder_hidden']
        dropout = config['model'].get('dropout', 0.05)
        self.lookback = config['data']['lookback_window']
        
        # MLP to map each timestep's features to a higher dimension
        self.input_mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # GRU to aggregate sequence
        self.gru = nn.GRU(
            input_size=hidden_dims[0],
            hidden_size=self.obs_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        # Output projection from GRU hidden state to return prediction
        self.readout = nn.Sequential(
            nn.Linear(self.obs_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        # Learnable Koopman operator (not used in this variant, but kept for compatibility)
        self.K = nn.Parameter(torch.eye(self.obs_dim) * 0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, lookback, features]
        Returns: predicted next return [batch, 1]
        """
        batch_size, T, _ = x.shape
        
        # Encode each timestep
        x_enc = self.input_mlp(x)  # [batch, T, hidden_dim0]
        
        # GRU forward
        _, h_n = self.gru(x_enc)   # h_n: [num_layers*2, batch, obs_dim]
        # Take last layer's hidden state
        latent = h_n[-1]            # [batch, obs_dim]
        
        # Predict return
        pred = self.readout(latent)
        return pred
    
    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Return latent state for a sequence (for analysis)."""
        batch_size, T, _ = x.shape
        x_enc = self.input_mlp(x)
        _, h_n = self.gru(x_enc)
        return h_n[-1]


if __name__ == "__main__":
    # test
    cfg = {
        'model': {'observable_dim': 64, 'input_dim': 15, 'encoder_hidden': [128], 'dropout': 0.05},
        'data': {'lookback_window': 63}
    }
    model = KoopmanSpectral(cfg)
    x = torch.randn(32, 63, 15)
    out = model(x)
    print(out.shape)
