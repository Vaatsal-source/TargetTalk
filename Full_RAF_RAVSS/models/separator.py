"""
models/separator.py

Compact SepFormer-style dual-path separator for blind multi-speaker
separation. The block structure stays small enough for Kaggle, while the
output head predicts one mask per source and uses a softmax across sources.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import cfg


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class DualPathBlock(nn.Module):
    def __init__(self):
        super().__init__()
        d = cfg.SEP_D_MODEL
        self.intra = TransformerEncoderBlock(d, cfg.SEP_N_HEADS, cfg.SEP_FF_DIM, cfg.SEP_DROPOUT)
        self.inter = TransformerEncoderBlock(d, cfg.SEP_N_HEADS, cfg.SEP_FF_DIM, cfg.SEP_DROPOUT)

    def forward(self, x):
        # x: (B, D, K, S)
        B, D, K, S = x.shape
        intra_in = x.permute(0, 3, 2, 1).reshape(B * S, K, D)
        intra_out = self.intra(intra_in)
        x = intra_out.reshape(B, S, K, D).permute(0, 3, 2, 1)

        inter_in = x.permute(0, 2, 3, 1).reshape(B * K, S, D)
        inter_out = self.inter(inter_in)
        x = inter_out.reshape(B, K, S, D).permute(0, 3, 1, 2)
        return x


class DualPathSeparator(nn.Module):
    def __init__(self):
        super().__init__()
        self.K = cfg.SEP_CHUNK_SIZE
        self.hop = cfg.SEP_HOP_SIZE
        self.in_norm = nn.GroupNorm(1, cfg.ENC_CHANNELS)
        self.in_proj = nn.Conv1d(cfg.ENC_CHANNELS, cfg.SEP_D_MODEL, 1)
        self.blocks = nn.ModuleList([DualPathBlock() for _ in range(cfg.SEP_N_LAYERS)])
        self.out_proj = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(cfg.SEP_D_MODEL, cfg.ENC_CHANNELS * cfg.MAX_SOURCES, 1),
        )
        self.use_ckpt = cfg.USE_GRAD_CKPT

    def _chunk(self, x):
        B, D, L = x.shape
        K, hop = self.K, self.hop
        pad = (hop - (L - K) % hop) % hop if L > K else (K - L)
        x = F.pad(x, (hop, hop + pad))
        L_p = x.shape[-1]
        n_chunks = (L_p - K) // hop + 1
        chunks = torch.stack([x[..., i * hop:i * hop + K] for i in range(n_chunks)], dim=-1)
        return chunks, L_p

    def _overlap_add(self, chunks, L_p, orig_L):
        B, D, K, S = chunks.shape
        hop = self.hop
        out = torch.zeros(B, D, L_p, device=chunks.device, dtype=chunks.dtype)
        norm = torch.zeros(1, 1, L_p, device=chunks.device, dtype=chunks.dtype)
        for i in range(S):
            out[..., i * hop:i * hop + K] += chunks[..., i]
            norm[..., i * hop:i * hop + K] += 1.0
        out = out / norm.clamp_min(1e-8)
        return out[..., hop:hop + orig_L]

    def forward(self, encoded_audio, vis_embed_seq=None):
        """
        encoded_audio: (B, N, L)
        vis_embed_seq: unused in blind separation mode; kept for API
        returns: (B, S, N, L)
        """
        orig_L = encoded_audio.shape[-1]
        x = self.in_norm(encoded_audio)
        x = self.in_proj(x)
        chunks, L_p = self._chunk(x)

        for block in self.blocks:
            if self.use_ckpt and self.training:
                chunks = torch.utils.checkpoint.checkpoint(block, chunks, use_reentrant=False)
            else:
                chunks = block(chunks)

        x = self._overlap_add(chunks, L_p, orig_L)
        mask_logits = self.out_proj(x)
        B, _, L = mask_logits.shape
        mask_logits = mask_logits.view(B, cfg.MAX_SOURCES, cfg.ENC_CHANNELS, L)
        masks = torch.softmax(mask_logits, dim=1)
        return masks
