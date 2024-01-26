import torch
from torch import nn


class TwoStageExtractor(nn.Module):
    """This module combines two feature extractors into a single network.

    The input data is expected to be of shape `[batch_size, upper_seq_len,
    input_channels, lower_seq_len]`. An example would be vibration data recorded in
    spaced intervals, where lower_seq_len is the length of an interval and
    upper_seq_len is the window size of a sliding window over the intervals.

    The lower_stage is applied to each interval individually to extract features.
    The upper_stage is then applied to the extracted features of the window.
    The resulting feature vector should represent the window without the need to
    manually extract features from the raw data of the intervals.
    """

    def __init__(
        self,
        lower_stage: nn.Module,
        upper_stage: nn.Module,
    ):
        """
        Create a new two-stage extractor.

        The lower stage needs to take a tensor of shape `[batch_size, input_channels,
        seq_len]` and return a tensor of shape `[batch_size, lower_output_units]`. The
        upper stage needs to take a tensor of shape `[batch_size, upper_seq_len,
        lower_output_units]` and return a tensor of shape `[batch_size,
        upper_output_units]`. Args: lower_stage: upper_stage:
        """
        super().__init__()

        self.lower_stage = lower_stage
        self.upper_stage = upper_stage

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Apply the two-stage extractor to the input tensor.

        The input tensor is expected to be of shape `[batch_size, upper_seq_len,
        input_channels, lower_seq_len]`. The output tensor will be of shape
        `[batch_size, upper_output_units]`.

        Args:
            inputs: the input tensor

        Returns:
            an output tensor of shape `[batch_size, upper_output_units]`
        """
        batch_size, upper_seq_len, input_channels, lower_seq_len = inputs.shape
        inputs = inputs.reshape(-1, input_channels, lower_seq_len)
        inputs = self.lower_stage(inputs)
        _, lower_output_units = inputs.shape
        inputs = inputs.reshape(batch_size, upper_seq_len, lower_output_units)
        inputs = torch.transpose(inputs, 1, 2)
        inputs = self.upper_stage(inputs)

        return inputs
