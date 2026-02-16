import math
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout_rate, max_len = 5000, reverse = False):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.reverse = reverse
        self.xscale = math.sqrt(self.d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.pe = None
