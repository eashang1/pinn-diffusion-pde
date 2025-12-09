import pytest
import torch


@pytest.fixture(autouse=True)
def set_random_seed():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


@pytest.fixture
def device():
    return torch.device('cpu')
