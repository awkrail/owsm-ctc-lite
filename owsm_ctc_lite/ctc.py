import torch.nn as nn


class CTC(nn.Module):
    def __init__(
        self,
        odim,
        encoder_output_size,
        dropout_rate = 0.0,
        ctc_type = "builtin",
        reduce = True,
        ignore_nan_grad = None,
        zero_infinity = True,
        brctc_risk_strategy = "exp",
        brctc_group_strategy = "end",
        brctc_risk_factor = 0.0,
    ):
        super().__init__()
