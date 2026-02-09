import torch.nn as nn


class Stft(nn.Module):
    def __init__(
        self,
        n_fft = 512,
        win_length = None,
        hop_length = 128,
        window = "hann",
        center = True,
        normalized = False,
        onesided = True,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.normalized = normalized
        self.onesided = onesided
        self.window = window
