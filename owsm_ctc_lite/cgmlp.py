import torch.nn as nn


class ConvolutionalGatingMLP(nn.Module):
    def __init__(
        self,
        size,
        linear_units,
        kernel_size,
        dropout_rate,
        use_linear_after_conv,
        gate_activation,
    ):
        super().__init__()
