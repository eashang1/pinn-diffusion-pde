import torch
import numpy as np
from src.pinn.normalized_pinn import NormalizedPINN


class TestNormalization:
    def test_input_normalization(self):
        config = {
            'S_range': (0.0, 400.0),
            't_range': (0.0, 0.1),
            'V_range': (0.0, 50.0),
            'pde_type': 'black_scholes',
            'parameters': {'strike': 200.0}
        }
        pinn = NormalizedPINN(config)

        S = torch.tensor([0.0, 200.0, 400.0])
        t = torch.tensor([0.0, 0.05, 0.1])

        S_norm, t_norm = pinn.normalize_inputs(S, t)

        assert torch.allclose(S_norm, torch.tensor([0.0, 0.5, 1.0]))
        assert torch.allclose(t_norm, torch.tensor([0.0, 0.5, 1.0]))

    def test_output_denormalization(self):
        config = {
            'S_range': (0.0, 400.0),
            't_range': (0.0, 0.1),
            'V_range': (0.0, 50.0),
            'pde_type': 'black_scholes',
            'parameters': {'strike': 200.0}
        }
        pinn = NormalizedPINN(config)

        V_norm = torch.tensor([0.0, 0.5, 1.0])
        V_phys = pinn.denormalize_output(V_norm)

        assert torch.allclose(V_phys, torch.tensor([0.0, 25.0, 50.0]))

    def test_heat_equation_offset(self):
        config = {
            'S_range': (0.0, 1.0),
            't_range': (0.0, 0.1),
            'V_range': (-1.0, 1.0),
            'pde_type': 'heat',
            'parameters': {'diffusivity': 0.01}
        }
        pinn = NormalizedPINN(config)

        # Check that offset and scale are set correctly
        assert pinn.heat_offset == 1.0
        assert pinn.heat_scale == 2.0


class TestPDEResiduals:
    def test_black_scholes_residual_shape(self):
        config = {
            'S_range': (0.0, 400.0),
            't_range': (0.0, 0.02),
            'V_range': (0.0, 50.0),
            'pde_type': 'black_scholes',
            'parameters': {
                'strike': 200.0,
                'volatility': 0.2,
                'risk_free_rate': 0.05,
                'dividend_rate': 0.03
            }
        }
        pinn = NormalizedPINN(config)

        S = torch.linspace(50, 350, 100)
        t = torch.linspace(0.001, 0.019, 100)

        loss = pinn.compute_pde_loss(S, t)

        assert loss.shape == torch.Size([])
        assert loss.item() >= 0

    def test_heat_equation_residual_shape(self):
        config = {
            'S_range': (0.0, 1.0),
            't_range': (0.0, 0.1),
            'V_range': (-1.0, 1.0),
            'pde_type': 'heat',
            'parameters': {'diffusivity': 0.01}
        }
        pinn = NormalizedPINN(config)

        x = torch.linspace(0.01, 0.99, 100)
        t = torch.linspace(0.01, 0.09, 100)

        loss = pinn.compute_pde_loss(x, t)

        assert loss.shape == torch.Size([])
        assert loss.item() >= 0

    def test_gradients_computed(self):
        config = {
            'S_range': (0.0, 1.0),
            't_range': (0.0, 0.1),
            'V_range': (-1.0, 1.0),
            'pde_type': 'heat',
            'parameters': {'diffusivity': 0.01}
        }
        pinn = NormalizedPINN(config)

        x = torch.tensor([0.5], requires_grad=True)
        t = torch.tensor([0.05], requires_grad=True)

        # This should compute derivatives internally
        loss = pinn.compute_pde_loss(x, t)

        # Loss should be computable
        assert not torch.isnan(loss)


class TestBoundaryConditions:
    def test_heat_dirichlet_boundaries(self):
        config = {
            'S_range': (0.0, 1.0),
            't_range': (0.0, 0.1),
            'V_range': (-1.0, 1.0),
            'pde_type': 'heat',
            'parameters': {'diffusivity': 0.01}
        }
        pinn = NormalizedPINN(config)

        boundary_loss = pinn.compute_boundary_loss()

        assert boundary_loss.shape == torch.Size([])
        assert boundary_loss.item() >= 0

    def test_bs_terminal_condition(self):
        config = {
            'S_range': (0.0, 400.0),
            't_range': (0.0, 0.02),
            'V_range': (0.0, 200.0),
            'pde_type': 'black_scholes',
            'parameters': {
                'strike': 200.0,
                'is_call': True
            }
        }
        pinn = NormalizedPINN(config)

        boundary_loss = pinn.compute_boundary_loss()

        assert boundary_loss.shape == torch.Size([])
        assert boundary_loss.item() >= 0


class TestInitialConditions:
    def test_heat_initial_condition(self):
        config = {
            'S_range': (0.0, 1.0),
            't_range': (0.0, 0.1),
            'V_range': (-1.0, 1.0),
            'pde_type': 'heat',
            'parameters': {'diffusivity': 0.01}
        }
        pinn = NormalizedPINN(config)

        initial_loss = pinn.compute_initial_condition_loss()

        assert initial_loss.shape == torch.Size([])
        assert initial_loss.item() >= 0

    def test_bs_no_initial_condition(self):
        config = {
            'S_range': (0.0, 400.0),
            't_range': (0.0, 0.02),
            'V_range': (0.0, 50.0),
            'pde_type': 'black_scholes',
            'parameters': {'strike': 200.0}
        }
        pinn = NormalizedPINN(config)

        initial_loss = pinn.compute_initial_condition_loss()

        assert initial_loss.item() == 0.0
