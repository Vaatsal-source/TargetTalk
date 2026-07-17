"""
config.py
Central configuration for RAF-RAVSS.

This version is tuned for Kaggle-class training on 3- and 4-speaker blind
separation with PIT loss. The defaults intentionally keep memory and runtime
under control before increasing model capacity.
"""

import os
import torch


class Config:
    # ---------------- paths ----------------
    DATA_ROOT   = "/kaggle/input/datasets/emnabouhajeb/lrs2-v1/mvlrs_v1"   # contains main/ and pretrain/
    TRAIN_SPLIT = "main"
    CKPT_DIR    = "/kaggle/working/checkpoints"
    LOG_DIR     = "/kaggle/working/logs"
    EVAL_DIR    = "/kaggle/working/eval_results"

    # ---------------- audio ----------------
    SAMPLE_RATE   = 16000
    CHUNK_SECONDS = 2.5
    CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SECONDS)
    N_FFT         = 512

    # ---------------- video ----------------
    VIDEO_FPS        = 25
    FRAMES_PER_CHUNK = int(VIDEO_FPS * CHUNK_SECONDS)
    LIP_CROP_SIZE    = 96
    USE_VISUAL       = False

    # ---------------- audio encoder/decoder (Conv-TasNet style) ----------------
    ENC_KERNEL   = 16
    ENC_STRIDE   = 8
    ENC_CHANNELS = 192

    # ---------------- separator (SepFormer dual-path transformer) ----------------
    SEP_CHUNK_SIZE = 80
    SEP_HOP_SIZE   = 40
    SEP_D_MODEL    = 192
    SEP_N_HEADS    = 4
    SEP_FF_DIM     = 384
    SEP_N_LAYERS   = 3
    SEP_DROPOUT    = 0.1
    MAX_SOURCES    = 4
    MIN_SOURCES    = 3

    # ---------------- visual encoder ----------------
    # Kept for compatibility with the original AV-RAVSS architecture, but
    # disabled by default for blind multi-speaker separation.
    VIS_EMBED_DIM = 192

    # ---------------- training ----------------
    BATCH_SIZE       = 2
    GRAD_ACCUM_STEPS = 8
    NUM_WORKERS      = 2
    EPOCHS           = 35
    LR               = 1.2e-4
    WEIGHT_DECAY     = 1e-5
    WARMUP_STEPS     = 1000
    GRAD_CLIP        = 5.0
    USE_AMP          = True
    USE_GRAD_CKPT    = True

    # ---------------- colab/kaggle session budget ----------------
    MAX_TRAIN_SECONDS      = 8.5 * 3600
    CKPT_EVERY_STEPS       = 500
    VAL_EVERY_STEPS        = 1000
    FINAL_EVAL_MAX_SAMPLES = 200
    FINAL_EVAL_SAVE_AUDIO  = 6
    FINAL_EVAL_BATCH_SIZE  = 2

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED   = 42

    @staticmethod
    def make_dirs():
        os.makedirs(Config.CKPT_DIR, exist_ok=True)
        os.makedirs(Config.LOG_DIR, exist_ok=True)
        os.makedirs(Config.EVAL_DIR, exist_ok=True)


cfg = Config
