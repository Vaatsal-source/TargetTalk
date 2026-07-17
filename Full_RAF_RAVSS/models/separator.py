"""
models/separator.py

SepFormer-style dual-path transformer separator, conditioned on the visual
embedding via FiLM (feature-wise linear modulation) at every dual-path block.
This is what makes it "audio-visual": instead of predicting a fixed number
of speaker masks (as in vanilla SepFormer), we predict ONE mask -- for
whichever speaker the video belongs to -- so the visual embedding must
actively steer the separation at each layer.

Dual-path idea (Luo et al., DPRNN / SepFormer):
  1. Chunk the (N, L) encoder output into overlapping segments of length K.
  2. "Intra-chunk" transformer: self-attention within each chunk (models
     short-range dependencies cheaply, O(K^2) instead of O(L^2)).
  3. "Inter-chunk" transformer: self-attention across chunks at the same
     position (models long-range dependencies).
  4. Overlap-add back to (N, L), repeat for SEP_N_LAYERS blocks, then a
     final linear + ReLU produces a mask in [0, inf) multiplied into the
     encoder output before decoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg


class FiLM(nn.Module):
    """Predicts per-channel (gamma, beta) from a pooled visual embedding and
    applies x' = gamma * x + beta, channel-wise."""

    def __init__(self, vis_dim, feat_dim):
        super().__init__()
        self.to_gamma_beta = nn.Linear(vis_dim, feat_dim * 2)

    def forward(self, x, vis_embed):
        # x: (B, D, L) ; vis_embed: (B, vis_dim) pooled over time
        gb = self.to_gamma_beta(vis_embed)          # (B, 2D)
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1)                  # (B, D, 1)
        beta = beta.unsqueeze(-1)
        return x * (1 + gamma) + beta


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
        self.film = FiLM(cfg.VIS_EMBED_DIM, d)

    def forward(self, x, vis_embed_pooled):
        # x: (B, D, K, S)  K=chunk size, S=num chunks
        B, D, K, S = x.shape

        # --- intra-chunk attention (within each of S chunks, over K) ---
        intra_in = x.permute(0, 3, 2, 1).reshape(B * S, K, D)
        intra_out = self.intra(intra_in)
        x = intra_out.reshape(B, S, K, D).permute(0, 3, 2, 1)  # back to (B,D,K,S)

        # --- inter-chunk attention (across S chunks, for each of K positions) ---
        inter_in = x.permute(0, 2, 3, 1).reshape(B * K, S, D)
        inter_out = self.inter(inter_in)
        x = inter_out.reshape(B, K, S, D).permute(0, 3, 1, 2)  # back to (B,D,K,S)

        # --- FiLM conditioning on the visual embedding, applied per-frame ---
        x_flat = x.reshape(B, D, K * S)
        x_flat = self.film(x_flat, vis_embed_pooled)
        x = x_flat.reshape(B, D, K, S)
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
            nn.Conv1d(cfg.SEP_D_MODEL, cfg.ENC_CHANNELS, 1),
        )
        self.use_ckpt = cfg.USE_GRAD_CKPT

    def _chunk(self, x):
        # x: (B, D, L) -> (B, D, K, S) with 50% overlap, zero-padded as needed
        B, D, L = x.shape
        K, hop = self.K, self.hop
        pad = (hop - (L - K) % hop) % hop if L > K else (K - L)
        x = F.pad(x, (hop, hop + pad))  # pad both sides for clean overlap-add later
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

    def forward(self, encoded_audio, vis_embed_seq):
        """
        encoded_audio: (B, N, L) from AudioEncoder
        vis_embed_seq: (B, F, VIS_EMBED_DIM) per-frame visual embedding
        returns: mask (B, N, L) in [0, inf)
        """
        orig_L = encoded_audio.shape[-1]
        x = self.in_norm(encoded_audio)
        x = self.in_proj(x)                          # (B, D, L)
        chunks, L_p, pad = self._chunk(x)             # (B, D, K, S)

        vis_pooled = vis_embed_seq.mean(dim=1)        # (B, VIS_EMBED_DIM) - simple temporal pooling

        for block in self.blocks:
            if self.use_ckpt and self.training:
                chunks = torch.utils.checkpoint.checkpoint(block, chunks, vis_pooled, use_reentrant=False)
            else:
                chunks = block(chunks, vis_pooled)

        x = self._overlap_add(chunks, L_p, pad, orig_L)  # (B, D, L)
        mask = self.out_proj(x)
        mask = F.relu(mask)
        return mask
