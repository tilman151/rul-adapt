import pytest
import torch


@pytest.fixture
def inputs():
    torch.manual_seed(42)
    inputs = torch.randn(8, 14, 30)

    return inputs
