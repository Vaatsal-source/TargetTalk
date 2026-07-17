"""
models/rafravss.py

Wires together the audio encoder -> separator -> audio decoder pipeline.
The visual frontend is retained only for compatibility with the older
audio-visual setup; blind multi-speaker training disables it by default.
"""

import torch
import torch.nn as nn

from config import cfg
from models.audio_codec import AudioEncoder, AudioDecoder
from models.visual_encoder import VisualFrontend
from models.separator import DualPathSeparator


class RAFRAVSS(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_encoder = AudioEncoder()
        self.audio_decoder = AudioDecoder()
        self.use_visual = cfg.USE_VISUAL
        self.visual_frontend = VisualFrontend() if self.use_visual else None
        self.separator = DualPathSeparator()

    def forward(self, mix_wav, lip_frames=None):
        """
        mix_wav    : (B, T)
        lip_frames : optional, kept for backward compatibility
        returns    : est_wav (B, S, T)
        """
        out_len = mix_wav.shape[-1]
        enc = self.audio_encoder(mix_wav)  # (B, N, L)
        vis_embed = self.visual_frontend(lip_frames) if (self.use_visual and lip_frames is not None) else None
        masks = self.separator(enc, vis_embed)  # (B, S, N, L)
        masked = masks * enc.unsqueeze(1)
        B, S, N, L = masked.shape
        est = self.audio_decoder(masked.reshape(B * S, N, L), out_len)
        return est.reshape(B, S, out_len)

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
