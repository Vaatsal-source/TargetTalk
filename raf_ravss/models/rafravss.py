"""
models/rafravss.py
Wires together: AudioEncoder -> DualPathSeparator (FiLM-conditioned on
VisualFrontend output) -> mask applied to encoded audio -> AudioDecoder.
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
        self.visual_frontend = VisualFrontend()
        self.separator = DualPathSeparator()

    def forward(self, mix_wav, lip_frames):
        """
        mix_wav    : (B, T)
        lip_frames : (B, F, 1, H, W)
        returns    : est_wav (B, T)
        """
        out_len = mix_wav.shape[-1]
        enc = self.audio_encoder(mix_wav)              # (B, N, L)
        vis_embed = self.visual_frontend(lip_frames)     # (B, F, VIS_EMBED_DIM)
        mask = self.separator(enc, vis_embed)             # (B, N, L)
        masked = enc * mask
        est_wav = self.audio_decoder(masked, out_len)
        return est_wav

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
