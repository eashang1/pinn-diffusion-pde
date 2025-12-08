# Heat equation: ∂u/∂t = α*∂²u/∂x². Analytical: u(x,t) = exp(-α*π²*t)*sin(πx)

import numpy as np
import torch
from typing import Dict
from ..pinn.normalized_pinn import NormalizedPINN


def create_heat_equation_pinn(
    spatial_range: tuple = (0.0, 1.0),
    time_range: tuple = (0.0, 0.1),
    value_range: tuple = (-1.0, 1.0),
    diffusivity: float = 0.01
) -> NormalizedPINN:
    config = {
        'S_range': spatial_range,  # Reusing S for spatial coordinate
        't_range': time_range,
        'V_range': value_range,
        'pde_type': 'heat',
        'parameters': {
            'diffusivity': diffusivity
        }
    }

    return NormalizedPINN(config)


def heat_analytical_solution(x: np.ndarray, t: float, diffusivity: float = 0.01) -> np.ndarray:
    # Analytical solution: u(x,t) = exp(-α*π²*t) * sin(πx)
    if isinstance(x, torch.Tensor):
        return torch.exp(-diffusivity * np.pi**2 * t) * torch.sin(np.pi * x)
    else:
        return np.exp(-diffusivity * np.pi**2 * t) * np.sin(np.pi * x)


def compute_heat_error(
    pinn: NormalizedPINN, x_test: torch.Tensor, t_test: float, diffusivity: float = 0.01
) -> Dict[str, float]:
    pinn.net.eval()

    with torch.no_grad():
        t_expanded = torch.full_like(x_test, t_test)
        u_pred = pinn.forward(x_test, t_expanded)

    u_analytical = heat_analytical_solution(x_test, t_test, diffusivity)

    l2_error = torch.mean((u_pred - u_analytical)**2).item()
    max_error = torch.max(torch.abs(u_pred - u_analytical)).item()

    return {
        'l2_error': l2_error,
        'max_error': max_error
    }
