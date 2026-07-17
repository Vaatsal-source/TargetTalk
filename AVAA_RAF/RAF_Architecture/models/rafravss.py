"""
models/rafravss.py
Wires together: AudioEncoder -> DualPathSeparator (dual mask, no visual
conditioning) -> mask applied to encoded audio, replicated per source ->
AudioDecoder -> (B, N_SOURCES, T) waveforms.
"""

import torch
import torch.nn as nn
from config import cfg
from models.audio_codec import AudioEncoder, AudioDecoder
from models.separator import DualPathSeparator


class TwoSpeakerSeparator(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_encoder = AudioEncoder()
        self.audio_decoder = AudioDecoder()
        self.separator = DualPathSeparator()

    def forward(self, mix_wav):
        """
        mix_wav : (B, T)
        returns : est_wav (B, N_SOURCES, T)
        """
        out_len = mix_wav.shape[-1]
        enc = self.audio_encoder(mix_wav)                 # (B, N, L)
        masks = self.separator(enc)                        # (B, S, N, L)
        enc_rep = enc.unsqueeze(1).expand(-1, cfg.N_SOURCES, -1, -1)  # (B, S, N, L)
        masked = enc_rep * masks
        est_wav = self.audio_decoder(masked, out_len)         # (B, S, T)
        return est_wav

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Backwards-compat alias: earlier revision of this file exposed the class as
# RAFRAVSS. Kept so any existing import in old notebook cells doesn't break.
RAFRAVSS = TwoSpeakerSeparator
