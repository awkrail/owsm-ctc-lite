import torch.nn as nn

from owsm_ctc_lite.positional_encoding import PositionalEncoding


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_size,
        output_size = 256,
        attention_heads = 4,
        linear_units = 2048,
        num_blocks = 6,
        dropout_rate = 0.1,
        positional_dropout_rate = 0.1,
        attention_dropout_rate = 0.0,
        input_layer = "conv2d",
        pos_enc_class = PositionalEncoding,
        pos_enc_layer_type = "abs_cos",
        normalize_before = True,
        concat_after = False,
        positionwise_layer_type = "linear",
        positionwise_conv_kernel_size = 1,
        padding_idx = -1,
        interctc_layer_idx = [],
        interctc_use_conditioning = False,
        layer_drop_rate = 0.0,
        qk_norm = False,
        use_flash_attn = True,
    ):
        super().__init__()
        self._output_size = output_size


    def output_size(self):
        return self._output_size
