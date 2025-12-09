# PINN Transfer Learning: Black-Scholes → Heat Equation

Demonstrates cross-domain transfer learning in Physics-Informed Neural Networks by training on option pricing (Black-Scholes PDE) and transferring learned representations to solve the heat equation. The core hypothesis: PINNs trained on mathematically similar PDEs learn universal differential operator structure rather than problem-specific patterns.

**Result**: Transfer learning achieves 8x lower L2 error vs. training from scratch, with faster convergence.

## Motivation

Traditional neural networks struggle with transfer learning across different physical domains. This project shows that proper normalization enables PINNs to transfer learned physics between the Black-Scholes equation (finance) and heat equation (thermodynamics), both parabolic diffusion PDEs with similar mathematical structure.

## Quick Start

```bash
conda env create -f environment.yml
conda activate pinn-pde
python run_transfer_learning.py      # Main transfer learning experiment
python run_fd_verification.py        # Finite difference validation
pytest tests/                        # Run test suite
```

## Architecture

**Network**: 3-layer feedforward (64 hidden units per layer, tanh activation, Xavier initialization)

**Input**: (x, t) spatial-temporal coordinates normalized to [0,1]²

**Output**: Solution value u(x,t) denormalized to physical units

**Loss function**: L = L_PDE + 10·L_boundary + 10·L_initial
- L_PDE: Mean squared PDE residual at collocation points
- L_boundary: Boundary condition enforcement
- L_initial: Initial condition enforcement (heat equation only)

**Optimizer**: Adam with ReduceLROnPlateau scheduler (patience=1000, factor=0.8)

## Method

### 1. Coordinate Normalization
All problems are normalized to the unit hypercube [0,1]² for spatial-temporal inputs and [0,1] for outputs. This crucial step enables weight transfer across different physical domains:

```
S_norm = (S - S_min) / (S_max - S_min)
t_norm = (t - t_min) / (t_max - t_min)
V_norm = (V - V_min) / (V_max - V_min)
```

### 2. Automatic Differentiation
Derivatives are computed via autograd in normalized space, then transformed to physical space using the chain rule:

```
∂V/∂t_phys = (∂V/∂t_norm) · (V_scale / t_scale)
∂²V/∂S²_phys = (∂²V/∂S²_norm) · (V_scale / S_scale²)
```

### 3. PDE Enforcement

**Black-Scholes PDE** (Option Pricing):
```
∂V/∂t + 0.5σ²S²∂²V/∂S² + (r-q)S∂V/∂S - rV = 0
```
- Terminal condition: V(S,T) = max(S-K, 0) for calls
- Boundary: V(0,t) = 0, V(S_max,t) = S_max - Ke^(-r(T-t))

**Heat Equation** (Diffusion):
```
∂u/∂t = α∂²u/∂x²
```
- Initial condition: u(x,0) = sin(πx)
- Boundary: u(0,t) = u(1,t) = 0 (Dirichlet)
- Analytical solution: u(x,t) = e^(-απ²t)sin(πx)

### 4. Transfer Learning Protocol

1. Train source PINN on Black-Scholes (8k epochs)
2. Transfer weights: `target_net.load_state_dict(source_net.state_dict())`
3. Fine-tune on heat equation (3k epochs)
4. Compare with baseline trained from scratch (5k epochs)

## Results

The transferred model achieves superior accuracy with less training:

| Model | Epochs | Avg L2 Error | Relative Performance |
|-------|--------|--------------|---------------------|
| Fresh training | 5000 | ~0.008 | Baseline |
| Transfer learning | 8000 + 3000 | ~0.001 | **8x better** |

This demonstrates that the network learns generalizable representations of diffusion operators that transfer across domains.

## Validation

The finite difference verification script (`run_fd_verification.py`) validates the Black-Scholes PINN against a Crank-Nicolson finite difference solver using realistic market parameters for AAPL options. Both methods agree within 5% relative error.

## Project Structure

```
├── src/
│   ├── pinn/
│   │   ├── network.py           # Feedforward architecture
│   │   └── normalized_pinn.py   # Normalized PINN with PDE losses
│   ├── pde/
│   │   ├── black_scholes.py     # BS PDE configuration
│   │   └── heat_equation.py     # Heat PDE configuration
│   ├── training/
│   │   └── trainer.py           # Training loop & evaluation
│   ├── finite_difference/
│   │   └── solver.py            # Crank-Nicolson FD solver
│   └── data/
│       └── options_data_loader.py  # Synthetic market data
├── tests/                       # Pytest test suite
├── config/
│   └── experiment_config.yaml   # Hyperparameters
├── run_transfer_learning.py     # Main experiment
└── run_fd_verification.py       # FD validation
```
