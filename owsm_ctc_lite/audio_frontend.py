import torch.nn as nn
from torch_complex import ComplexTensor

from owsm_ctc_lite.log_mel import LogMel
from owsm_ctc_lite.stft import Stft


class AudioFrontEnd(nn.Module):
    def __init__(
        self, 
        fs = 16000,
        n_fft = 512,
        win_length = None,
        hop_length = 128,
        window = "hann",
        center = True,
        normalized = False,
        onesided = True,
        n_mels = 80,
        fmin = None,
        fmax = None,
        htk = False,
        apply_stft = True,
    ):
        super().__init__()
        self.sample_rate = fs
        self.hop_length = hop_length

        if apply_stft:
            self.stft = Stft(
                n_fft = n_fft,
                win_length = win_length,
                hop_length = hop_length,
                center = center,
                window = window,
                normalized = normalized,
                onesided = onesided,
            )
        else:
            self.stft = None

        self.apply_stft = apply_stft
        self.logmel = LogMel(
            fs = fs,
            n_fft = n_fft,
            n_mels = n_mels,
            fmin = fmin,
            fmax = fmax,
            htk = htk,
        )
        self.n_mels = n_mels
        self.frontend_type = "default"


    def output_size(self):
        return self.n_mels 


    def forward(self, input, input_lengths):
        # 1. STFT
        input_stft, feats_lens = self._compute_stft(input, input_lengths)
        # 2. STFT -> Power spectrum
        input_power = input_stft.real**2 + input_stft.imag**2
        # 3. LogMel
        input_feats, _ = self.logmel(input_power, feats_lens)
        return input_feats, feats_lens


    def _compute_stft(self, input, input_lengths):
        input_stft, feats_lens = self.stft(input, input_lengths)
        input_stft = ComplexTensor(input_stft[..., 0], input_stft[..., 1])
        return input_stft, feats_lens
