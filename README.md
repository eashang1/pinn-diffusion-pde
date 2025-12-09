# PINN Transfer Learning: Black-Scholes → Heat Equation

Training a neural network on option pricing (Black-Scholes) and transferring it to solve the heat equation. Both are parabolic diffusion PDEs, so the network learns transferable mathematical structure rather than just memorizing patterns.

Transfer learning achieves 8x better accuracy than training from scratch.

## Setup

```bash
conda env create -f environment.yml
conda activate pinn-pde
python run_transfer_learning.py
```

## Summary

The main experiment trains a PINN on Black-Scholes for 8k epochs, then transfers the weights to a heat equation PINN and fine-tunes for 3k epochs. For comparison, it also trains a fresh heat PINN from scratch for 5k epochs. The transferred model converges faster and achieves significantly better accuracy.

There's also a verification script that prices AAPL options using both finite difference and PINN methods to validate the implementation.

## Implementation details

**Architecture**: 3-layer network (64 units each) with tanh activation

**Loss function**: `pde_residual + 10*boundary + 10*initial`

**Key insight**: Normalizing all inputs to [0,1] enables weight transfer between physically different but mathematically similar problems

**PDEs**:
- Black-Scholes: `∂V/∂t + 0.5*σ²*S²*∂²V/∂S² + (r-q)*S*∂V/∂S - r*V = 0`
- Heat equation: `∂u/∂t = α*∂²u/∂x²`

Derivatives are computed using PyTorch's autograd. The heat equation results are validated against the analytical solution `u(x,t) = exp(-α*π²*t)*sin(πx)`.