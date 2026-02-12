import torch
import numpy as np
import torch.nn as nn


class GlobalMVN(nn.Module):
    def __init__(
        self,
        stats_file,
        norm_means = True,
        norm_vars = True,
        eps = 1.0e-20
    ):
        super().__init__()
        self.norm_means = norm_means
        self.norm_vars = norm_vars
        self.stats = stats_file
        stats = np.load(stats_file)

        count = stats["count"]
        sum_v = stats["sum"]
        sum_square_v = stats["sum_square"]
        mean = sum_v / count
        var = sum_square_v / count - mean * mean
        std = np.sqrt(np.maximum(var, eps))

        mean = torch.from_numpy(mean)
        std = torch.from_numpy(std)

        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    
    def forward(self, x):
        norm_means = self.norm_means
        norm_vars = self.norm_vars
        self.mean = self.mean.to(x.device, x.dtype)
        self.std = self.std.to(x.device, x.dtype)

        if norm_means:
            x -= self.mean

        if norm_vars:
            x /= self.std

        return x
