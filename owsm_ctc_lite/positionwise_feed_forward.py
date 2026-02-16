import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, idim, hidden_units, dropout_rate, activation=nn.ReLU()):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(idim, hidden_units)
        self.w_2 = nn.Linear(hidden_units, idim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = activation


    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))
