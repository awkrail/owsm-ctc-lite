import torch
import torch.nn as nn


class MultiSequential(nn.Sequential):
    def __init__(self, *args, layer_drop_rate = 0.0):
        super(MultiSequential, self).__init__(*args)
        self.layer_drop_rate = layer_drop_rate

    def forward(self, *args):
        _probs = torch.empty(len(self)).uniform_()
        for idx, m in enumerate(self):
            if not self.training and (_probs[idx] >= self.layer_drop_rate):
                args = m(*args)
        return args


def repeat(N, fn, layer_drop_rate=0.0):
    return MultiSequential(*[fn(n) for n in range(N)], layer_drop_rate=layer_drop_rate)
