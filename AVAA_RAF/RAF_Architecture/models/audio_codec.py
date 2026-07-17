"""
models/audio_codec.py
Conv-TasNet-style learnable encoder/decoder. Decoder now reconstructs
N_SOURCES waveforms (Refactor.md Section 5, Phase A/C).
"""

import torch
import torch.nn as nn
from config import cfg


class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = nn.Conv1d(
            1, cfg.ENC_CHANNELS, kernel_size=cfg.ENC_KERNEL, stride=cfg.ENC_STRIDE, bias=False
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, wav):
        # wav: (B, T) -> (B, 1, T) -> (B, N, L)
        x = wav.unsqueeze(1)
        x = self.conv1d(x)
        return self.act(x)


class AudioDecoder(nn.Module):
    """Reconstructs N_SOURCES waveforms from N_SOURCES masked latent
    representations, per Refactor.md Phase C: fold (B, S, N, L) ->
    (B*S, N, L), run the transposed conv once, unfold back to (B, S, T)."""

    def __init__(self):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(
            cfg.ENC_CHANNELS, 1, kernel_size=cfg.ENC_KERNEL, stride=cfg.ENC_STRIDE, bias=False
        )

    def forward(self, x, out_len):
        # x: (B, S, N, L)
        B, S, N, L = x.shape
        x = x.reshape(B * S, N, L)
        wav = self.deconv(x).squeeze(1)              # (B*S, T')
        if wav.shape[-1] < out_len:
            wav = torch.nn.functional.pad(wav, (0, out_len - wav.shape[-1]))
        else:
            wav = wav[..., :out_len]
        return wav.view(B, S, out_len)
