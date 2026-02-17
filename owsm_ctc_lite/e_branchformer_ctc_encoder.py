import torch.nn as nn

from owsm_ctc_lite.subsampling import Conv2dSubsampling8
from owsm_ctc_lite.positional_encoding import PositionalEncoding
from owsm_ctc_lite.positionwise_feed_forward import PositionwiseFeedForward
from owsm_ctc_lite.attention import MultiheadAttention
from owsm_ctc_lite.cgmlp import ConvolutionalGatingMLP
from owsm_ctc_lite.layer_norm import LayerNorm
from owsm_ctc_lite.repeat import repeat

from owsm_ctc_lite.swish import Swish


class EBranchformerEncoderLayer(nn.Module):
    def __init__(
        self,
        size,
        attn,
        cgmlp,
        feed_forward,
        feed_forward_macaron,
        cross_attn,
        dropout_rate,
        merge_conv_kernel = 3,
    ):
        super().__init__()
        self.size = size
        self.attn = attn
        self.cgmlp = cgmlp
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        
        self.ff_scale = 1.0
        if self.feed_forward is not None:
            self.norm_ff = LayerNorm(size)

        if self.feed_forward_macaron is not None:
            self.ff_scale = 0.5
            self.norm_ff_macaron = LayerNorm(size)
        
        self.norm_mha = LayerNorm(size)
        self.norm_mlp = LayerNorm(size)
        self.norm_final = LayerNorm(size)

        self.cross_attn = cross_attn
        if self.cross_attn is not None:
            self.norm_cross_attn = LayerNorm(size)

        self.dropout = nn.Dropout(dropout_rate)
        self.depthwise_conv_fusion = nn.Conv1d(
            size + size,
            size + size,
            kernel_size = merge_conv_kernel,
            stride = 1,
            padding = (merge_conv_kernel - 1) // 2,
            groups = size + size,
            bias = True,
        )
        self.merge_proj = nn.Linear(size + size, size)


class EBranchformerCTCEncoder(nn.Module):
    def __init__(
        self,
        input_size,
        output_size = 256,
        attention_heads = 4,
        attention_layer_type = "rel_selfattn",
        pos_enc_layer_type = "rel_pos",
        rel_pos_type = "latest",
        cgmlp_linear_units = 2048,
        cgmlp_conv_kernel = 31,
        use_linear_after_conv = False,
        gate_activation = "identify",
        num_blocks = 12,
        dropout_rate = 0.1,
        positional_dropout_rate = 0.1,
        attention_dropout_rate = 0.0,
        input_layer = "conv2d8",
        zero_triu = False,
        padding_idx = -1,
        layer_drop_rate = 0.0,
        max_pos_emb_len = 5000,
        use_ffn = False,
        macaron_ffn = False,
        ffn_activation_type = "swish",
        linear_units = 2048,
        positionwise_layer_type = "linear",
        merge_conv_kernel = 3,
        interctc_layer_idx = None,
        interctc_use_conditioning = False,
        use_cross_attention = True,
        use_flash_attn = False,
    ):
        super().__init__()
        self._output_size = output_size
        
        self.embed = Conv2dSubsampling8(
            input_size,
            output_size,
            dropout_rate,
            PositionalEncoding(output_size, positional_dropout_rate, max_pos_emb_len),
        )

        activation = Swish()
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
            activation,
        )

        encoder_selfattn_layer = MultiheadAttention
        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            attention_dropout_rate,
            False,
            use_flash_attn,
        )

        cgmlp_layer = ConvolutionalGatingMLP
        cgmlp_layer_args = (
            output_size,
            cgmlp_linear_units,
            cgmlp_conv_kernel,
            dropout_rate,
            use_linear_after_conv,
            gate_activation,
        )
        
        # todo: implement repeat
        self.encoders = repeat(
            num_blocks,
            lambda lnum: EBranchformerEncoderLayer(
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                cgmlp_layer(*cgmlp_layer_args),
                positionwise_layer(*positionwise_layer_args) if use_ffn else None,
                positionwise_layer(*positionwise_layer_args) if use_ffn and macaron_ffn else None,
                MultiheadAttention(
                    attention_heads,
                    output_size,
                    attention_dropout_rate,
                    False,
                    use_flash_attn,
                    cross_attn = True,
                ) if use_cross_attention[lnum] else None,
                dropout_rate,
                merge_conv_kernel,
            ),
            layer_drop_rate,
        )
        self.after_norm = LayerNorm(output_size)
        self.interctc_layer_idx = interctc_layer_idx
        self.interctc_use_conditioning = interctc_use_conditioning
        self.conditioning_layer = None


    def output_size(self):
        return self._output_size


    def forward(self, xs_pad, ilens):
        import ipdb; ipdb.set_trace()
