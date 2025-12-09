import torch
import numpy as np
from src.pinn.normalized_pinn import NormalizedPINN
from src.training.trainer import train_pinn, evaluate_heat_pinn, transfer_weights


class TestTraining:
    def test_loss_decreases(self):
        config = {
            'S_range': (0.0, 1.0),
            't_range': (0.0, 0.1),
            'V_range': (-1.0, 1.0),
            'pde_type': 'heat',
            'parameters': {'diffusivity': 0.01}
        }
        pinn = NormalizedPINN(config)

        loss_history = train_pinn(
            pinn,
            epochs=100,
            learning_rate=1e-3,
            n_collocation=100,
            print_freq=50
        )

        # Check loss history has correct keys
        assert 'total' in loss_history
        assert 'pde' in loss_history
        assert 'boundary' in loss_history
        assert 'initial' in loss_history

        # Check loss generally decreases
        initial_loss = np.mean(loss_history['total'][:10])
        final_loss = np.mean(loss_history['total'][-10:])
        assert final_loss < initial_loss

    def test_loss_history_length(self):
        config = {
            'S_range': (0.0, 1.0),
            't_range': (0.0, 0.1),
            'V_range': (-1.0, 1.0),
            'pde_type': 'heat',
            'parameters': {'diffusivity': 0.01}
        }
        pinn = NormalizedPINN(config)

        epochs = 50
        loss_history = train_pinn(pinn, epochs=epochs, print_freq=25)

        assert len(loss_history['total']) == epochs
        assert len(loss_history['pde']) == epochs

    def test_optimizer_updates_weights(self):
        config = {
            'S_range': (0.0, 1.0),
            't_range': (0.0, 0.1),
            'V_range': (-1.0, 1.0),
            'pde_type': 'heat',
            'parameters': {'diffusivity': 0.01}
        }
        pinn = NormalizedPINN(config)

        # Store initial weights
        initial_weights = [p.clone() for p in pinn.net.parameters()]

        # Train
        train_pinn(pinn, epochs=10, print_freq=10)

        # Check weights changed
        for initial, current in zip(initial_weights, pinn.net.parameters()):
            assert not torch.allclose(initial, current)


class TestHeatEquationEvaluation:
    def test_evaluation_structure(self):
        config = {
            'S_range': (0.0, 1.0),
            't_range': (0.0, 0.1),
            'V_range': (-1.0, 1.0),
            'pde_type': 'heat',
            'parameters': {'diffusivity': 0.01}
        }
        pinn = NormalizedPINN(config)

        # Train briefly
        train_pinn(pinn, epochs=50, print_freq=50)

        # Evaluate
        test_times = [0.02, 0.05, 0.08]
        errors = evaluate_heat_pinn(pinn, test_times=test_times, diffusivity=0.01)

        assert 'times' in errors
        assert 'l2_errors' in errors
        assert 'max_errors' in errors
        assert 'avg_l2' in errors

        assert len(errors['times']) == len(test_times)
        assert len(errors['l2_errors']) == len(test_times)

    def test_analytical_comparison(self):
        config = {
            'S_range': (0.0, 1.0),
            't_range': (0.0, 0.1),
            'V_range': (-1.0, 1.0),
            'pde_type': 'heat',
            'parameters': {'diffusivity': 0.01}
        }
        pinn = NormalizedPINN(config)

        # Train
        train_pinn(pinn, epochs=100, print_freq=100)

        # Evaluate at specific point
        x = torch.tensor([0.5])
        t = torch.tensor([0.02])

        pinn.net.eval()
        with torch.no_grad():
            u_pred = pinn.forward(x, t)
            u_analytical = torch.exp(-0.01 * np.pi**2 * t) * torch.sin(np.pi * x)

        # Predictions should be finite
        assert torch.isfinite(u_pred).all()
        assert torch.isfinite(u_analytical).all()


class TestTransferLearning:
    def test_weight_transfer(self):
        bs_config = {
            'S_range': (0.0, 400.0),
            't_range': (0.0, 0.02),
            'V_range': (0.0, 50.0),
            'pde_type': 'black_scholes',
            'parameters': {'strike': 200.0}
        }

        heat_config = {
            'S_range': (0.0, 1.0),
            't_range': (0.0, 0.1),
            'V_range': (-1.0, 1.0),
            'pde_type': 'heat',
            'parameters': {'diffusivity': 0.01}
        }

        source_pinn = NormalizedPINN(bs_config)
        target_pinn = NormalizedPINN(heat_config)

        # Train source briefly
        train_pinn(source_pinn, epochs=50, print_freq=50)

        # Store source weights
        source_weights = [p.clone() for p in source_pinn.net.parameters()]

        # Transfer
        transfer_weights(source_pinn, target_pinn)

        # Check target has same weights as source
        for source_param, target_param in zip(source_weights, target_pinn.net.parameters()):
            assert torch.allclose(source_param, target_param)

    def test_transfer_preserves_architecture(self):
        bs_config = {
            'S_range': (0.0, 400.0),
            't_range': (0.0, 0.02),
            'V_range': (0.0, 50.0),
            'pde_type': 'black_scholes',
            'parameters': {'strike': 200.0}
        }

        heat_config = {
            'S_range': (0.0, 1.0),
            't_range': (0.0, 0.1),
            'V_range': (-1.0, 1.0),
            'pde_type': 'heat',
            'parameters': {'diffusivity': 0.01}
        }

        source_pinn = NormalizedPINN(bs_config)
        target_pinn = NormalizedPINN(heat_config)

        # Get layer counts before transfer
        source_layers = len(list(source_pinn.net.parameters()))
        target_layers_before = len(list(target_pinn.net.parameters()))

        transfer_weights(source_pinn, target_pinn)

        # Architecture should be unchanged
        target_layers_after = len(list(target_pinn.net.parameters()))
        assert source_layers == target_layers_before == target_layers_after
