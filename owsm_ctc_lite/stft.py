import torch
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


    def forward(self, input, ilens):
        bs = input.size(0)
        if input.dim() == 3:
            multi_channel = True
            input = input.transpose(1, 2).reshape(-1, input.size(1))
        else:
            multi_channel = False
        
        if self.window is not None:
            # hanning window
            window_func = getattr(torch, f"{self.window}_window")
            window = window_func(self.win_length, dtype=input.dtype, device=input.device)
        else:
            window = None

        stft_kwargs = dict(
            n_fft = self.n_fft,
            win_length = self.win_length,
            hop_length = self.hop_length,
            center = self.center,
            window = window,
            normalized = self.normalized,
            onesided = self.onesided,
        )
        stft_kwargs["return_complex"] = True
        output = torch.stft(input.float(), **stft_kwargs)
        output = torch.view_as_real(output).type(input.dtype)

        output = output.transpose(1, 2)

        if multi_channel:
            output = output.view(bs, -1, output.size(1), output.size(2), 2).transpose(1, 2)

        # now expected a single file input - so no need for batchfied inputs
        if ilens is not None:
            if self.center:
                pad = self.n_fft // 2
                ilens = ilens + 2 * pad
            olens = (
                torch.div(ilens - self.n_fft, self.hop_length, rounding_mode = "trunc")
                + 1
            )
        else:
            olens = None

        return output, olens
