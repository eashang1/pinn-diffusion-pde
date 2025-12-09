import torch
import numpy as np
from typing import Dict, Tuple, Optional
from .network import create_pinn_network, get_device


class NormalizedPINN:
    def __init__(self, problem_config: Dict):
        self.S_min, self.S_max = problem_config['S_range']
        self.t_min, self.t_max = problem_config['t_range']
        self.V_min, self.V_max = problem_config['V_range']
        self.pde_type = problem_config['pde_type']
        self.params = problem_config.get('parameters', {})
        self.device = get_device()

        if self.pde_type == 'heat' and self.V_min < 0:
            self.heat_offset = abs(self.V_min)
            self.heat_scale = self.V_max - self.V_min
            self.V_min, self.V_max = 0.0, 1.0
        else:
            self.heat_offset = 0
            self.heat_scale = 1.0

        self.net = create_pinn_network(input_dim=2, output_dim=1, hidden_layers=[64, 64, 64], device=self.device)

    def normalize_inputs(self, S: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        S_norm = (S - self.S_min) / (self.S_max - self.S_min)
        t_norm = (t - self.t_min) / (self.t_max - self.t_min)
        return S_norm, t_norm

    def denormalize_output(self, V_norm: torch.Tensor) -> torch.Tensor:
        V_phys = V_norm * (self.V_max - self.V_min) + self.V_min
        if self.pde_type == 'heat' and hasattr(self, 'heat_offset'):
            V_phys = V_phys * self.heat_scale - self.heat_offset
        return V_phys

    def forward(self, S: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        S_norm, t_norm = self.normalize_inputs(S, t)
        inputs = torch.cat([S_norm.unsqueeze(-1), t_norm.unsqueeze(-1)], dim=-1)
        V_norm = self.net(inputs).squeeze()
        return self.denormalize_output(V_norm)

    def forward_normalized(
        self,
        S_norm: torch.Tensor,
        t_norm: torch.Tensor
    ) -> torch.Tensor:
        inputs = torch.cat([S_norm.unsqueeze(-1), t_norm.unsqueeze(-1)], dim=-1)
        return self.net(inputs).squeeze()

    def compute_pde_loss(self, S: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Compute derivatives in normalized space, then transform to physical via chain rule
        S_norm, t_norm = self.normalize_inputs(S, t)
        S_norm.requires_grad_(True)
        t_norm.requires_grad_(True)

        V_norm = self.forward_normalized(S_norm, t_norm)

        # Compute derivatives in normalized space using autograd
        V_t_norm = torch.autograd.grad(
            V_norm.sum(), t_norm, create_graph=True
        )[0]
        V_S_norm = torch.autograd.grad(
            V_norm.sum(), S_norm, create_graph=True
        )[0]
        V_SS_norm = torch.autograd.grad(
            V_S_norm.sum(), S_norm, create_graph=True
        )[0]

        # Transform derivatives to physical space via chain rule
        S_scale = self.S_max - self.S_min
        t_scale = self.t_max - self.t_min
        V_scale = self.V_max - self.V_min

        V_t = V_t_norm * V_scale / t_scale
        V_S = V_S_norm * V_scale / S_scale
        V_SS = V_SS_norm * V_scale / (S_scale**2)

        # Convert to physical coordinates for PDE evaluation
        S_phys = S_norm * S_scale + self.S_min
        V_phys = self.denormalize_output(V_norm)

        # Compute PDE residual based on problem type
        if self.pde_type == 'black_scholes':
            pde_residual = self._compute_bs_residual(
                V_t, V_S, V_SS, S_phys, V_phys
            )
        elif self.pde_type == 'heat':
            pde_residual = self._compute_heat_residual(V_t, V_SS)
        else:
            raise ValueError(f"Unknown PDE type: {self.pde_type}")

        return torch.mean(pde_residual**2)

    def _compute_bs_residual(
        self,
        V_t: torch.Tensor,
        V_S: torch.Tensor,
        V_SS: torch.Tensor,
        S: torch.Tensor,
        V: torch.Tensor
    ) -> torch.Tensor:
        # Black-Scholes: ∂V/∂t + 0.5*σ²*S²*∂²V/∂S² + (r-q)*S*∂V/∂S - r*V = 0
        sigma = torch.full_like(S, self.params.get('volatility', 0.2))
        r = torch.full_like(S, self.params.get('risk_free_rate', 0.05))
        q = torch.full_like(S, self.params.get('dividend_rate', 0.03))

        return V_t + 0.5 * sigma**2 * S**2 * V_SS + (r - q) * S * V_S - r * V

    def _compute_heat_residual(
        self,
        V_t: torch.Tensor,
        V_SS: torch.Tensor
    ) -> torch.Tensor:
        # Heat equation: ∂u/∂t - α*∂²u/∂x² = 0
        alpha = self.params.get('diffusivity', 0.01)
        return V_t - alpha * V_SS

    def compute_boundary_loss(self) -> torch.Tensor:
        if self.pde_type == 'black_scholes':
            return self._compute_bs_boundary_loss()
        elif self.pde_type == 'heat':
            return self._compute_heat_boundary_loss()
        else:
            raise ValueError(f"Unknown PDE type: {self.pde_type}")

    def _compute_bs_boundary_loss(self) -> torch.Tensor:
        losses = []

        # Terminal condition: V(S, T) = payoff(S)
        t_terminal_norm = torch.ones(50, device=self.device, requires_grad=True)
        S_terminal_norm = torch.linspace(0, 1, 50, device=self.device, requires_grad=True)

        V_terminal_norm = self.forward_normalized(S_terminal_norm, t_terminal_norm)

        # Compute payoff in physical space
        S_terminal_phys = S_terminal_norm * (self.S_max - self.S_min) + self.S_min
        strike = self.params['strike']

        if self.params.get('is_call', True):
            payoff_phys = torch.maximum(
                S_terminal_phys - strike,
                torch.tensor(0.0, device=self.device)
            )
        else:
            payoff_phys = torch.maximum(
                strike - S_terminal_phys,
                torch.tensor(0.0, device=self.device)
            )

        # Convert payoff to normalized space
        payoff_norm = (payoff_phys - self.V_min) / (self.V_max - self.V_min)
        payoff_norm = torch.clamp(payoff_norm, 0, 1)

        terminal_loss = torch.mean((V_terminal_norm - payoff_norm)**2)
        losses.append(terminal_loss)

        # Spatial boundary conditions
        t_boundary_norm = torch.linspace(0, 1, 50, device=self.device, requires_grad=True)

        # Lower boundary (S = 0)
        S_low_norm = torch.zeros_like(t_boundary_norm, requires_grad=True)
        V_low_norm = self.forward_normalized(S_low_norm, t_boundary_norm)

        if self.params.get('is_call', True):
            boundary_low_norm = torch.zeros_like(V_low_norm)
        else:
            # Put option: V(0,t) = K*exp(-r*(T-t))
            t_phys = t_boundary_norm * (self.t_max - self.t_min) + self.t_min
            r = self.params.get('risk_free_rate', 0.05)
            boundary_low_phys = strike * torch.exp(-r * (self.t_max - t_phys))
            boundary_low_norm = (boundary_low_phys - self.V_min) / (self.V_max - self.V_min)
            boundary_low_norm = torch.clamp(boundary_low_norm, 0, 1)

        low_loss = torch.mean((V_low_norm - boundary_low_norm)**2)
        losses.append(low_loss)

        # Upper boundary (S = S_max)
        S_high_norm = torch.ones_like(t_boundary_norm, requires_grad=True)
        V_high_norm = self.forward_normalized(S_high_norm, t_boundary_norm)

        if self.params.get('is_call', True):
            # Call option: V(S_max,t) ≈ S_max - K*exp(-r*(T-t))
            t_phys = t_boundary_norm * (self.t_max - self.t_min) + self.t_min
            r = self.params.get('risk_free_rate', 0.05)
            boundary_high_phys = self.S_max - strike * torch.exp(-r * (self.t_max - t_phys))
            boundary_high_norm = (boundary_high_phys - self.V_min) / (self.V_max - self.V_min)
            boundary_high_norm = torch.clamp(boundary_high_norm, 0, 1)
        else:
            boundary_high_norm = torch.zeros_like(V_high_norm)

        high_loss = torch.mean((V_high_norm - boundary_high_norm)**2)
        losses.append(high_loss)

        return sum(losses)

    def _compute_heat_boundary_loss(self) -> torch.Tensor:
        t_boundary_norm = torch.linspace(0, 1, 50, device=self.device, requires_grad=True)

        # Dirichlet boundaries: u(0,t) = u(1,t) = 0
        x_left_norm = torch.zeros_like(t_boundary_norm, requires_grad=True)
        x_right_norm = torch.ones_like(t_boundary_norm, requires_grad=True)

        u_left_norm = self.forward_normalized(x_left_norm, t_boundary_norm)
        u_right_norm = self.forward_normalized(x_right_norm, t_boundary_norm)

        # u=0 maps to normalized value of 0.5 if original range was [-1,1]
        if hasattr(self, 'heat_offset') and self.heat_offset > 0:
            target_norm = 0.5
        else:
            target_norm = 0.0

        target_tensor = torch.full_like(u_left_norm, target_norm)

        return (
            torch.mean((u_left_norm - target_tensor)**2) +
            torch.mean((u_right_norm - target_tensor)**2)
        )

    def compute_initial_condition_loss(self) -> torch.Tensor:
        if self.pde_type != 'heat':
            return torch.tensor(0.0, device=self.device)

        x_norm = torch.linspace(0, 1, 100, device=self.device, requires_grad=True)
        t_init_norm = torch.zeros_like(x_norm, requires_grad=True)

        u_pred_norm = self.forward_normalized(x_norm, t_init_norm)

        # Initial condition: u(x,0) = sin(πx)
        u_initial_phys = torch.sin(np.pi * x_norm)

        # Convert to normalized space
        if hasattr(self, 'heat_offset') and self.heat_offset > 0:
            u_initial_norm = (u_initial_phys + self.heat_offset) / self.heat_scale
        else:
            u_initial_norm = (u_initial_phys - self.V_min) / (self.V_max - self.V_min)

        return torch.mean((u_pred_norm - u_initial_norm)**2)
