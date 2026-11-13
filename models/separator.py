"""
models/separator.py

Dual-path transformer separator for BLIND two-speaker separation
(Refactor.md Section 1 & 5). No FiLM, no visual conditioning -- the network
must learn to split the two speakers using only the acoustic mixture, and
PIT-SI-SDR loss (utils/metrics.py) resolves the arbitrary output ordering.

Dual-path structure per block:
  1. Chunk (B, D, L) into overlapping segments -> (B, D, K, S).
  2. Intra-chunk transformer: self-attention within each of the S chunks
     (over K positions) -- short-range dependencies, O(K^2) cost.
  3. Inter-chunk transformer: self-attention across the S chunks (over each
     of the K positions) -- long-range dependencies.
  4. Residual connection around each of intra/inter (Refactor.md Section 5).
  Repeat for SEP_N_LAYERS blocks, gradient-checkpointed during training.

Output head: PReLU -> Conv1d(D -> N_SOURCES * ENC_CHANNELS) -> reshape to
(B, N_SOURCES, ENC_CHANNELS, L) -> ReLU -> two non-negative masks.
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
            nn.Linear(d_model, ff_dim), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(ff_dim, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B*, T, D)
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
        residual = x

        # --- intra-chunk: attend over K within each of S chunks ---
        intra_in = x.permute(0, 3, 2, 1).reshape(B * S, K, D)
        intra_out = self.intra(intra_in)
        x = intra_out.reshape(B, S, K, D).permute(0, 3, 2, 1)
        x = x + residual   # residual skip, Refactor.md Section 5

        residual = x
        # --- inter-chunk: attend over S across each of K positions ---
        inter_in = x.permute(0, 2, 3, 1).reshape(B * K, S, D)
        inter_out = self.inter(inter_in)
        x = inter_out.reshape(B, K, S, D).permute(0, 3, 1, 2)
        x = x + residual   # residual skip

        return x


class DualPathSeparator(nn.Module):
    def __init__(self):
        super().__init__()
        self.K = cfg.SEP_CHUNK_SIZE
        self.hop = cfg.SEP_HOP_SIZE
        self.n_sources = cfg.N_SOURCES

        self.in_norm = nn.GroupNorm(1, cfg.ENC_CHANNELS)
        self.in_proj = nn.Conv1d(cfg.ENC_CHANNELS, cfg.SEP_D_MODEL, 1)
        self.blocks = nn.ModuleList([DualPathBlock() for _ in range(cfg.SEP_N_LAYERS)])
        self.mask_head = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(cfg.SEP_D_MODEL, cfg.N_SOURCES * cfg.ENC_CHANNELS, 1),
        )
        self.use_ckpt = cfg.USE_GRAD_CKPT

    def _chunk(self, x):
        # x: (B, D, L) -> (B, D, K, S), zero-padded both sides for clean overlap-add
        B, D, L = x.shape
        K, hop = self.K, self.hop
        pad = (hop - (L - K) % hop) % hop if L > K else (K - L)
        x = F.pad(x, (hop, hop + pad))
        L_p = x.shape[-1]
        n_chunks = (L_p - K) // hop + 1
        chunks = torch.stack([x[..., i * hop: i * hop + K] for i in range(n_chunks)], dim=-1)
        return chunks, L_p, pad

    def _overlap_add(self, chunks, L_p, pad, orig_L):
        B, D, K, S = chunks.shape
        hop = self.hop
        out = torch.zeros(B, D, L_p, device=chunks.device, dtype=chunks.dtype)
        norm = torch.zeros(1, 1, L_p, device=chunks.device, dtype=chunks.dtype)
        for i in range(S):
            out[..., i * hop: i * hop + K] += chunks[..., i]
            norm[..., i * hop: i * hop + K] += 1.0
        out = out / norm.clamp_min(1e-8)
        out = out[..., hop: hop + orig_L]
        return out

    def forward(self, encoded_audio):
        """
        encoded_audio: (B, N, L) from AudioEncoder
        returns: masks (B, N_SOURCES, N, L), each in [0, inf)
        """
        B, N, orig_L = encoded_audio.shape
        x = self.in_norm(encoded_audio)
        x = self.in_proj(x)                        # (B, D, L)
        chunks, L_p, pad = self._chunk(x)           # (B, D, K, S)

        for block in self.blocks:
            if self.use_ckpt and self.training:
                chunks = torch.utils.checkpoint.checkpoint(block, chunks, use_reentrant=False)
            else:
                chunks = block(chunks)

        x = self._overlap_add(chunks, L_p, pad, orig_L)   # (B, D, L)
        out = self.mask_head(x)                             # (B, N_SOURCES*N, L)
        out = out.view(B, self.n_sources, N, orig_L)
        masks = F.relu(out)
        return masks
