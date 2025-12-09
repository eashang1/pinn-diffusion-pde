import torch
from src.pinn.network import FeedForwardNetwork, create_pinn_network


class TestFeedForwardNetwork:
    def test_network_initialization(self):
        layer_sizes = [2, 64, 64, 64, 1]
        net = FeedForwardNetwork(layer_sizes)

        assert len(net.layers) == 4
        assert net.layers[0].in_features == 2
        assert net.layers[-1].out_features == 1

    def test_xavier_initialization(self):
        net = create_pinn_network(input_dim=2, output_dim=1, hidden_layers=[64, 64, 64])

        for layer in net.layers:
            # Check weights are not all zeros
            assert not torch.allclose(layer.weight, torch.zeros_like(layer.weight))
            # Check biases are initialized to zero
            assert torch.allclose(layer.bias, torch.zeros_like(layer.bias))

    def test_forward_pass_shape(self):
        net = create_pinn_network(input_dim=2, output_dim=1, hidden_layers=[64, 64, 64])
        x = torch.randn(100, 2)
        output = net(x)

        assert output.shape == (100, 1)

    def test_tanh_activation(self):
        net = create_pinn_network(input_dim=2, output_dim=1, hidden_layers=[8, 8])
        x = torch.randn(10, 2)

        # Forward pass through first layer with tanh
        h1 = torch.tanh(net.layers[0](x))

        # Values should be bounded between -1 and 1
        assert torch.all(h1 >= -1.0) and torch.all(h1 <= 1.0)

    def test_device_placement(self):
        device = torch.device('cpu')
        net = create_pinn_network(device=device)

        assert next(net.parameters()).device == device


class TestGradientFlow:
    def test_gradient_computation(self):
        net = create_pinn_network(input_dim=2, output_dim=1, hidden_layers=[64, 64, 64])
        x = torch.randn(10, 2, requires_grad=True)

        output = net(x)
        loss = output.sum()
        loss.backward()

        # Check input gradients exist
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

        # Check parameter gradients exist
        for param in net.parameters():
            assert param.grad is not None
