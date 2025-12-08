import torch
import numpy as np
from typing import List, Dict
from ..pinn.normalized_pinn import NormalizedPINN
from ..pinn.network import get_device


def train_pinn(
    pinn: NormalizedPINN,
    epochs: int = 10000,
    learning_rate: float = 1e-3,
    n_collocation: int = 1000,
    pde_weight: float = 1.0,
    boundary_weight: float = 10.0,
    initial_weight: float = 10.0,
    print_freq: int = 1000,
    scheduler_patience: int = 1000,
    scheduler_factor: float = 0.8
) -> Dict[str, List[float]]:
    device = get_device()

    optimizer = torch.optim.Adam(pinn.net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=scheduler_patience,
        factor=scheduler_factor,
        verbose=False
    )

    pinn.net.train()

    loss_history = {
        'total': [],
        'pde': [],
        'boundary': [],
        'initial': []
    }

    for epoch in range(epochs):
        optimizer.zero_grad()

        S_norm = torch.rand(n_collocation, device=device, requires_grad=True)
        t_norm = torch.rand(n_collocation, device=device, requires_grad=True)

        S_phys = S_norm * (pinn.S_max - pinn.S_min) + pinn.S_min
        t_phys = t_norm * (pinn.t_max - pinn.t_min) + pinn.t_min

        pde_loss = pinn.compute_pde_loss(S_phys, t_phys)
        boundary_loss = pinn.compute_boundary_loss()
        initial_loss = pinn.compute_initial_condition_loss()

        total_loss = pde_weight * pde_loss + boundary_weight * boundary_loss + initial_weight * initial_loss

        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss)

        loss_history['total'].append(total_loss.item())
        loss_history['pde'].append(pde_loss.item())
        loss_history['boundary'].append(boundary_loss.item())
        loss_history['initial'].append(initial_loss.item())

        if epoch % print_freq == 0:
            print(
                f'Epoch {epoch:5d}: '
                f'Total={total_loss.item():.6f}, '
                f'PDE={pde_loss.item():.6f}, '
                f'Boundary={boundary_loss.item():.6f}, '
                f'Initial={initial_loss.item():.6f}'
            )

    return loss_history


def evaluate_heat_pinn(
    pinn: NormalizedPINN,
    test_times: List[float] = [0.02, 0.05, 0.08],
    n_spatial_points: int = 50,
    diffusivity: float = 0.01
) -> Dict[str, any]:
    device = get_device()

    x_test = torch.linspace(0, 1, n_spatial_points, device=device)
    t_test = torch.tensor(test_times, device=device)

    errors = {
        'times': [],
        'l2_errors': [],
        'max_errors': []
    }

    pinn.net.eval()

    with torch.no_grad():
        for t_val in t_test:
            t_expanded = torch.full_like(x_test, t_val.item())
            u_pred = pinn.forward(x_test, t_expanded)
            u_analytical = torch.exp(-diffusivity * np.pi**2 * t_val) * torch.sin(np.pi * x_test)

            l2_error = torch.mean((u_pred - u_analytical)**2).item()
            max_error = torch.max(torch.abs(u_pred - u_analytical)).item()

            errors['times'].append(t_val.item())
            errors['l2_errors'].append(l2_error)
            errors['max_errors'].append(max_error)

    errors['avg_l2'] = np.mean(errors['l2_errors'])

    return errors


def print_evaluation_results(errors: Dict[str, any], name: str = "PINN"):
    print(f"\n{name}:")
    print("Time    | Max Error | Mean L2 Error")
    print("-" * 40)

    for t, l2, max_err in zip(errors['times'], errors['l2_errors'], errors['max_errors']):
        print(f"{t:.3f}   | {max_err:.6f} | {l2:.6f}")

    print(f"Average |           | {errors['avg_l2']:.6f}")


def transfer_weights(source_pinn: NormalizedPINN, target_pinn: NormalizedPINN):
    target_pinn.net.load_state_dict(source_pinn.net.state_dict())
    print(f"Transferred weights from {source_pinn.pde_type} to {target_pinn.pde_type} PINN")
