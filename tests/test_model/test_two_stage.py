import pytest
import torch

from rul_adapt.model import TwoStageExtractor


@pytest.fixture()
def extractor():
    lower_stage = torch.nn.Sequential(
        torch.nn.Conv1d(3, 8, 3),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(8 * 62, 8),
    )
    upper_stage = torch.nn.Sequential(
        torch.nn.Conv1d(8, 8, 2),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(8 * 3, 8),
    )
    extractor = TwoStageExtractor(lower_stage, upper_stage)

    return extractor


@pytest.fixture()
def inputs():
    return torch.rand(16, 4, 3, 64)


def test_forward_shape(inputs, extractor):
    outputs = extractor(inputs)

    assert outputs.shape == (16, 8)


def test_forward_upper_lower_interaction(inputs, extractor):
    one_sample = inputs[3]

    lower_outputs = extractor.lower_stage(one_sample)
    upper_outputs = extractor.upper_stage(
        torch.transpose(lower_outputs.unsqueeze(0), 1, 2)
    )
    outputs = extractor(inputs)

    assert torch.allclose(upper_outputs, outputs[3])
