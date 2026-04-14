# P2 ETF Koopman-Spectral Engine

**Research only · Not financial advice**

Linearized dynamics engine using Koopman operator theory for ETF return prediction.

## Overview

| Attribute | Value |
|:---|:---|
| **Core Mechanism** | DMD initialization + MLP encoder + learnable linear Koopman operator |
| **Training Time** | 30–90 minutes (GitHub Actions compatible) |
| **Output Signal** | ETF pick + Koopman modes (growth/oscillatory/decay) + predictability index |
| **Key Distinction** | Physics-inspired decomposition; eigenvalue interpretability |
| **Use Case** | Regime identification via spectral analysis of market dynamics |

## Architecture
OHLCV + Macro Features
↓
[MLP Encoder] → Koopman Observables z(t) ∈ R^K
↓
[Linear Operator K] → z(t+1) = K · z(t)  (learnable)
↓
[Readout] → Return Predictions
↓
[Eigendecomposition] → Modes + Predictability Index
plain
Copy

## Signal Output Format

```json
{
  "engine": "KOOPMAN-SPECTRAL",
  "primary_pick": {
    "etf": "XLK",
    "conviction_pct": 78.5,
    "expected_return_1d": 12.3,
    "expected_return_5d": 45.2
  },
  "koopman_modes": {
    "regime": "expansion",
    "growth_modes": 2,
    "oscillatory_modes": 1,
    "decay_modes": 61,
    "predictability_index": 0.82,
    "dominant_frequency_cycles": 0.15
  }
}
Training
bash
Copy
python train.py
DMD initialization provides warm start (converges in ~50-100 epochs vs 500+ cold).
Signal Generation
bash
Copy
python generate_signals.py
GitHub Actions
Automatic training and signal generation scheduled 2 AM UTC weekdays.
Dependencies
See requirements.txt. CPU-only compatible.
References
Koopman operator theory: Koopman (1931), Mezic (2005)
DMD: Schmid (2010), Tu et al. (2014)
Deep Koopman: Lusch et al. (2018), Morton et al. (2019)
plain
Copy

---

### `.gitignore`
Data (large, cached externally)
*.parquet
*.csv
/mnt/data/
data/
Model outputs (large binaries)
*.pt
*.pth
*.ckpt
Generated signals (committed optionally)
signals/*.json
Python
pycache/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.env
Jupyter
.ipynb_checkpoints
IDE
.vscode/
.idea/
*.swp
*.swo
OS
.DS_Store
Thumbs.db
Logs
*.log
training_history.json
plain
Copy

---

## Quick Start Commands

```bash
# Local setup
git clone https://github.com/P2SAMAPA/P2-ETF-KOOPMAN-SPECTRAL.git
cd P2-ETF-KOOPMAN-SPECTRAL
pip install -r requirements.txt

# Train
python train.py

# Generate signals
python generate_signals.py
