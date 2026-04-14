"""
Utilities for Koopman-Spectral engine.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


def compute_spectral_analysis(model, save_path='spectral_plot.png'):
    """Visualize Koopman eigenvalue spectrum."""
    with torch.no_grad():
        eigs = torch.linalg.eigvals(model.K).cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, label='Unit circle')
    
    # Eigenvalues
    ax.scatter(eigs.real, eigs.imag, c='blue', alpha=0.6, s=50)
    
    # Color by magnitude
    magnitudes = np.abs(eigs)
    for i, (re, im, mag) in enumerate(zip(eigs.real, eigs.imag, magnitudes)):
        color = 'red' if mag > 1 else 'green'
        ax.scatter(re, im, c=color, s=50, alpha=0.6)
    
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.set_title('Koopman Operator Eigenvalue Spectrum')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    return eigs


def mode_interpretation(modes_dict):
    """Human-readable interpretation of Koopman modes."""
    interpretations = []
    
    if modes_dict['growth_count'] > 0:
        interpretations.append(f"Growth modes ({modes_dict['growth_count']}): Unstable/expanding dynamics detected")
    
    if modes_dict['oscillatory_count'] > 0:
        interpretations.append(f"Oscillatory modes ({modes_dict['oscillatory_count']}): Cyclical patterns present")
    
    if modes_dict['decay_count'] > 0:
        interpretations.append(f"Decay modes ({modes_dict['decay_count']}): Stable/contracting dynamics")
    
    if modes_dict['spectral_gap'] > 0.5:
        interpretations.append("High spectral gap: System is predictable (slow/fast dynamics well-separated)")
    else:
        interpretations.append("Low spectral gap: Mixed timescales, prediction uncertainty elevated")
    
    return " | ".join(interpretations)
