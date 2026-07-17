"""
models/audio_codec.py
Learnable audio encoder/decoder (Conv-TasNet style). The encoder turns the
raw waveform into a sequence of latent frames; the decoder is its transpose
and turns separated latent frames back into a waveform.
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
        # wav: (B, T) -> (B, 1, T)
        x = wav.unsqueeze(1)
        x = self.conv1d(x)
        return self.act(x)  # (B, N, L)


class AudioDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(
            cfg.ENC_CHANNELS, 1, kernel_size=cfg.ENC_KERNEL, stride=cfg.ENC_STRIDE, bias=False
        )

    def forward(self, x, out_len):
        # x: (B, N, L) -> (B, T)
        wav = self.deconv(x).squeeze(1)
        if wav.shape[-1] < out_len:
            wav = torch.nn.functional.pad(wav, (0, out_len - wav.shape[-1]))
        else:
            wav = wav[..., :out_len]
        return wav
