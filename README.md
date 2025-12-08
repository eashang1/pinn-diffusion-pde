# PINN Transfer Learning: Black-Scholes → Heat Equation

Transfer learning across PDEs. Shows networks learn universal math structure, not just patterns.

**Result**: 8x accuracy improvement vs training from scratch.

## Quick Start

```bash
conda env create -f environment.yml
conda activate pinn-pde
python run_transfer_learning.py
```

## Structure

```
├── run_transfer_learning.py    # Main experiment
├── run_fd_verification.py      # FD validation
├── config/experiment_config.yaml
└── src/
    ├── pinn/                    # Networks
    ├── pde/                     # BS & Heat
    ├── training/                # Loops
    ├── finite_difference/       # FD solver
    └── data/                    # Synthetic data
```

## Scripts

### `run_transfer_learning.py`
Trains on Black-Scholes (8k epochs), transfers to Heat (3k epochs fine-tune), compares vs fresh training (5k epochs). ~20 min on M4 Mac.

### `run_fd_verification.py`
Generates AAPL option data, prices with FD and PINN, compares results. ~8 min on M4 Mac.

## Math

**Black-Scholes**: `∂V/∂t + 0.5*σ²*S²*∂²V/∂S² + (r-q)*S*∂V/∂S - r*V = 0`

**Heat**: `∂u/∂t = α*∂²u/∂x²` with analytical solution `u(x,t) = exp(-α*π²*t)*sin(πx)`

Both are parabolic diffusion PDEs.

## Implementation

**Normalization**: All domains → [0,1]³ to enable transfer learning

**Loss**: `total = pde_residual + 10*boundary + 10*initial`

**Derivatives**: Computed via PyTorch autograd

## Usage

**Config**: Edit `config/experiment_config.yaml` for epochs, learning rate, etc.

**Data**:
```python
from src.data.options_data_loader import OptionsDataLoader
loader = OptionsDataLoader()
data = loader.generate_realistic_data('AAPL', strike=200, days_to_expiry=7)
```

**Custom PINN**:
```python
from src.pde.black_scholes import create_black_scholes_pinn
from src.training.trainer import train_pinn

pinn = create_black_scholes_pinn(strike=200, volatility=0.25)
train_pinn(pinn, epochs=10000)
```

## Setup

```bash
conda env create -f environment.yml && conda activate pinn-pde
```

## Data Generator

Includes realistic synthetic data for AAPL, SPY, TSLA with appropriate:
- Volatilities (AAPL: 25%, SPY: 18%, TSLA: 55%)
- Interest rates (~4.5% based on 2024)
- Dividend yields (AAPL: 0.5%, SPY: 1.5%, TSLA: 0%)

## Results

**Hypothesis**: PINNs trained on mathematically similar PDEs learn transferable physics

**Evidence**:
1. Transfer outperforms fresh training
2. Both PDEs are parabolic diffusion processes
3. Normalization enables weight transfer across domains