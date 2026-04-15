"""
Koopman-Spectral model core with LSTM sequence encoder.
Processes full lookback window to capture temporal dynamics.
"""

import torch
import torch.nn as nn
import numpy as np


class SequenceEncoder(nn.Module):
    """
    LSTM encoder that processes the entire lookback window.
    Outputs final hidden state as the Koopman observable.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout, bidirectional=False
        )
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, lookback, features]
        Returns: [batch, hidden_dim] (last hidden state)
        """
        _, (h_n, _) = self.lstm(x)
        # h_n shape: [num_layers, batch, hidden_dim]
        return h_n[-1]  # last layer


class KoopmanSpectral(nn.Module):
    """
    Koopman model with sequence encoder and learnable linear dynamics.
    """
    def __init__(self, config: dict):
        super().__init__()
        input_dim = config['model']['input_dim']
        lookback = config['data']['lookback_window']
        self.obs_dim = config['model']['observable_dim']
        self.lookback = lookback
        
        # Sequence encoder: processes full window
        self.encoder = SequenceEncoder(
            input_dim=input_dim,
            hidden_dim=self.obs_dim,
            num_layers=config['model'].get('lstm_layers', 2),
            dropout=config['model'].get('dropout', 0.1)
        )
        
        # Learnable Koopman operator (linear in latent space)
        self.K = nn.Parameter(torch.eye(self.obs_dim) * 0.01)
        
        # Readout: latent -> 1-day return prediction
        self.readout = nn.Sequential(
            nn.Linear(self.obs_dim, 64),
            nn.ReLU(),
            nn.Dropout(config['model'].get('dropout', 0.1)),
            nn.Linear(64, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, lookback, features]
        Returns: [batch, 1] predicted next-day return
        """
        z = self.encoder(x)            # [batch, obs_dim]
        # Apply Koopman operator (optional: could be used for multi-step)
        # For 1-step prediction, we can directly use z
        # But we also could evolve z: z_next = z @ K.T
        # For simplicity, we use the current latent to predict return.
        # To incorporate dynamics, we can do:
        z_next = z @ self.K.T
        return self.readout(z_next)
    
    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Return latent representation for analysis."""
        return self.encoder(x)


if __name__ == "__main__":
    # Test
    cfg = {
        'model': {'input_dim': 15, 'observable_dim': 64, 'lstm_layers': 2, 'dropout': 0.1},
        'data': {'lookback_window': 63}
    }
    model = KoopmanSpectral(cfg)
    x = torch.randn(4, 63, 15)
    y = model(x)
    print(y.shape)
