import torch.nn as nn


class LogMel(nn.Module):
    def __init__(
        self,
        fs = 16000,
        n_fft = 512,
        n_mels = 80,
        fmin = None,
        fmax = None,
        htk = None,
        log_base = None,
    ):
        super().__init__()

        fmin = 0 if fmin is None else fmin
        fmax = fs / 2 if fmax is None else fmax
        _mel_options = dict(
            sr = fs,
            n_fft = n_fft,
            n_mels = n_mels,
            fmin = fmin,
            fmax = fmax,
            htk = htk,
        )
        self.mel_options = _mel_options
        self.log_base = log_base
