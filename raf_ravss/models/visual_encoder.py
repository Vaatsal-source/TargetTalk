"""
models/visual_encoder.py

Compact visual frontend that mirrors AV-HuBERT's I/O contract:
  input : (B, F, 1, H, W) grayscale lip-ROI frames at 25 fps
  output: (B, F, VIS_EMBED_DIM) one embedding per video frame

Architecture: 3D-conv stem (space+time) -> 2D ResNet block stack per-frame
-> temporal Conv1D stack to mix across frames -> linear projection.
This is trained from scratch and is deliberately small (~3-4M params) so it
fits alongside the separator in a single Colab GPU.

To use a REAL pretrained AV-HuBERT visual encoder instead:
  1. Download the AV-HuBERT LRS3 checkpoint (fairseq).
  2. Wrap it so its forward(video) -> (B, F, D) matches this class's output.
  3. Freeze it (or LoRA-adapt it) and project D -> VIS_EMBED_DIM.
  4. Swap `VisualFrontend` for your wrapper in models/rafravss.py; nothing
     else in the pipeline needs to change.
"""

import torch
import torch.nn as nn
from config import cfg


class ResBlock2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + identity)


class VisualFrontend(nn.Module):
    def __init__(self, embed_dim=None):
        super().__init__()
        embed_dim = embed_dim or cfg.VIS_EMBED_DIM

        # 3D conv stem: light spatio-temporal downsampling (kept shallow for memory).
        self.stem3d = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        )  # (B,32,F,H/4,W/4)

        self.res_stack = nn.Sequential(
            ResBlock2D(32),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            ResBlock2D(64),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, embed_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: (B, F, 1, H, W) -> 3D conv wants (B, C, F, H, W)
        B, F, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)                    # (B,1,F,H,W)
        x = self.stem3d(x)                              # (B,32,F,H',W')
        _, ch, Ft, Hp, Wp = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * Ft, ch, Hp, Wp)  # per-frame 2D conv
        x = self.res_stack(x)                            # (B*Ft, 64, 1, 1)
        x = x.view(B, Ft, -1).permute(0, 2, 1)            # (B, 64, Ft)
        x = self.temporal_conv(x)                         # (B, embed_dim, Ft)
        return x.permute(0, 2, 1)                          # (B, Ft, embed_dim)
