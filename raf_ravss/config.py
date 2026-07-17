"""
config.py
Central configuration for RAF-RAVSS (audio-visual target-speaker extraction
on LRS2). Every number here was picked with a 16 GB Colab/Kaggle T4/P100 GPU
and a ~9 hour session limit in mind. Shrink BATCH_SIZE / CHUNK_SECONDS first
if you hit OOM; everything else scales off those two.
"""

import os
import torch

class Config:
    # ---------------- paths ----------------
    DATA_ROOT   = "/kaggle/input/datasets/emnabouhajeb/lrs2-v1/mvlrs_v1"   # contains main/ and pretrain/
    TRAIN_SPLIT = "main"          # folder to use for train/val (pretrain is much bigger, optional)
    CKPT_DIR    = "/kaggle/working/checkpoints"
    LOG_DIR     = "/kaggle/working/logs"

    # ---------------- audio ----------------
    SAMPLE_RATE      = 16000
    CHUNK_SECONDS    = 3.0                       # fixed-length audio chunk used per sample
    CHUNK_SAMPLES    = int(SAMPLE_RATE * CHUNK_SECONDS)
    N_FFT            = 512

    # ---------------- video ----------------
    VIDEO_FPS        = 25
    FRAMES_PER_CHUNK = int(VIDEO_FPS * CHUNK_SECONDS)   # 75 frames @ 3s
    LIP_CROP_SIZE    = 96                         # HxW of mouth ROI fed to visual encoder

    # ---------------- audio encoder/decoder (Conv-TasNet style) ----------------
    ENC_KERNEL   = 16
    ENC_STRIDE   = 8
    ENC_CHANNELS = 256                            # N in Conv-TasNet notation

    # ---------------- separator (SepFormer dual-path transformer) ----------------
    SEP_CHUNK_SIZE   = 100      # dual-path local chunk length (in encoder frames)
    SEP_HOP_SIZE     = 50       # 50% overlap
    SEP_D_MODEL      = 256      # kept == ENC_CHANNELS so no extra projection needed
    SEP_N_HEADS      = 4
    SEP_FF_DIM       = 512
    SEP_N_LAYERS     = 4        # 4 dual-path blocks (intra+inter each) -> good accuracy/memory tradeoff
    SEP_DROPOUT      = 0.1

    # ---------------- visual encoder (lightweight AV-HuBERT-style frontend) ----------------
    # NOTE: a full pretrained AV-HuBERT is ~300M+ params and will not comfortably
    # fit + train end-to-end on a single Colab GPU alongside the separator.
    # This is a compact 3D-CNN + ResNet2D + temporal-conv frontend trained from
    # scratch, matching AV-HuBERT's *interface* (per-frame embedding sequence at
    # 25 fps). Swap `VisualFrontend` for a frozen pretrained AV-HuBERT encoder
    # (see models/visual_encoder.py docstring) once you have disk space / time
    # to download the checkpoint (~2.5GB) and it will plug in unchanged.
    VIS_EMBED_DIM = 256          # projected to match SEP_D_MODEL for FiLM conditioning

    # ---------------- training ----------------
    BATCH_SIZE        = 4                 # audio+video pairs per step; raise if memory allows
    GRAD_ACCUM_STEPS  = 4                 # effective batch = 16
    NUM_WORKERS       = 2
    EPOCHS            = 40
    LR                = 1.5e-4
    WEIGHT_DECAY      = 1e-5
    WARMUP_STEPS      = 1000
    GRAD_CLIP         = 5.0
    USE_AMP           = True              # mixed precision -> ~2x memory savings
    USE_GRAD_CKPT     = True              # trade compute for memory in separator blocks

    # ---------------- colab/kaggle session budget ----------------
    MAX_TRAIN_SECONDS   = 8.5 * 3600      # stop + checkpoint before the session gets killed
    CKPT_EVERY_STEPS    = 500             # rolling checkpoint (also acts as resume point)
    VAL_EVERY_STEPS     = 1000

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED   = 42

    @staticmethod
    def make_dirs():
        os.makedirs(Config.CKPT_DIR, exist_ok=True)
        os.makedirs(Config.LOG_DIR, exist_ok=True)


cfg = Config
