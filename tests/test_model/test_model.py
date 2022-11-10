import pytest
import torch
import numpy.testing as npt

from rul_adapt import model


def lstm_extractor():
    return model.LstmExtractor(14, [16, 16], 8)


def lstm_extractor_differing_units():
    return model.LstmExtractor(14, [16, 8], 8)


pytestmark = pytest.mark.parametrize(
    ["net_func", "inputs"],
    [
        (lstm_extractor, torch.randn(8, 14, 30)),
        (lstm_extractor_differing_units, torch.randn(8, 14, 30)),
    ],
)


@torch.no_grad()
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU was detected")
def test_device_moving(net_func, inputs):
    net = net_func()
    net.eval()

    torch.manual_seed(42)
    outputs_cpu = net(inputs)

    net_on_gpu = net.to("cuda:0")
    torch.manual_seed(42)
    outputs_gpu = net_on_gpu(inputs.to("cuda:0"))

    torch.manual_seed(42)
    net_back_on_cpu = net_on_gpu.cpu()
    outputs_back_on_cpu = net_back_on_cpu(inputs)

    npt.assert_almost_equal(0.0, torch.sum(outputs_cpu - outputs_gpu.cpu()).item())
    npt.assert_almost_equal(0.0, torch.sum(outputs_cpu - outputs_back_on_cpu).item())


def test_batch_independence(net_func, inputs):
    net = net_func()
    inputs = inputs.clone()
    inputs.requires_grad = True

    # Compute forward pass in eval mode to deactivate batch norm
    net.eval()
    outputs = net(inputs)
    net.train()

    # Mask loss for certain samples in batch
    batch_size = inputs.shape[0]
    mask_idx = torch.randint(0, batch_size, ())
    mask = torch.ones_like(outputs)
    mask[mask_idx] = 0
    outputs = outputs * mask

    # Compute backward pass
    loss = outputs.mean()
    loss.backward()

    # Check if gradient exists and is zero for masked samples
    for i, grad in enumerate(inputs.grad):
        if i == mask_idx:
            assert torch.all(grad == 0).item()
        else:
            assert not torch.all(grad == 0)


def test_all_parameters_updated(net_func, inputs):
    net = net_func()
    optim = torch.optim.SGD(net.parameters(), lr=0.1)

    outputs = net(inputs)
    loss = outputs.mean()
    loss.backward()
    optim.step()

    for param_name, param in net.named_parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert not 0.0 == torch.sum(param.grad**2)
