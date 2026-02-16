import torch.nn as nn


class Conv2dSubsampling8(nn.Module):
    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        super(Conv2dSubsampling8, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, odim, 3, 2),
            nn.ReLU(),
            nn.Conv2d(odim, odim, 3, 2),
            nn.ReLU(),
            nn.Conv2d(odim, odim, 3, 2),
            nn.ReLU(),
        )
        self.out = nn.Linear(odim * ((((idim - 1) // 2 - 1) // 2 - 1) // 2), odim)
        self.pos_enc = (
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate)
        )
