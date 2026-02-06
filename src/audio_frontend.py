import torch.nn as nn


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
        self.sample_rate = sample_rate
        self.hop_length = hop_length

        # This frontend_conf is copied from the original ESPNnet2
        # May not be necessary?
        frontend_conf = {
            'use_wpe': False,
            'wtype': 'blstmp',
            'wlayers': 3,
            'wunits': 300,
            'wprojs': 320,
            'wdropout_rate': 0.0,
            'taps': 5,
            'delay': 3,
            'use_dnn_mask_for_wpe': True,
            'use_beamformer': False,
            'btype': 'blstmp',
            'blayers': 3,
            'bunits': 300,
            'bprojs': 320,
            'bnmask': 2,
            'badim': 320,
            'ref_channel': -1,
            'bdropout_rate': 0.0
        }

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


    def forward(self, input, input_lengths):
        pass
