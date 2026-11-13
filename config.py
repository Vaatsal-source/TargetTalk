"""
config.py
Blind two-speaker audio-only separation (PIT-SI-SDR). No visual dependency.

Param-budget note (Refactor.md Section 4 vs Section 5):
Section 5 specifies ENC_CHANNELS=256 / SEP_D_MODEL=256 / SEP_FF_DIM=512 /
SEP_N_LAYERS=4, which is the reference dual-path SepFormer-style config.
Section 4 separately mandates a 2.0-3.5M total-parameter budget for fast
convergence inside the Kaggle time limit. Those two specs are in tension --
the reference config with n_heads=4 alone lands well north of 3.5M. This
config defaults to the SMALLER dims (below) to satisfy the hard 2.0-3.5M
budget from Section 4, since that constraint has an explicit rationale
(epoch cycle time). Bump ENC_CHANNELS/SEP_D_MODEL/SEP_FF_DIM back up to
256/256/512 if you'd rather match Section 5 literally and accept slower
convergence / a bigger model (run count_params.py to check where you land).
"""

import os
import torch


class Config:
    # ---------------- paths ----------------
    DATA_ROOT = "/kaggle/input/datasets/emnabouhajeb/lrs2-v1/mvlrs_v1"  # contains main/ and pretrain/
    SPLITS = ["main", "pretrain"]          # traverse both, per Refactor.md Section 2
    # Optional split file lists (train.txt/val.txt/test.txt). If any of these
    # don't exist under DATA_ROOT, the loader falls back to a deterministic
    # random split over all discovered files (see data/lrs2_dataset.py).
    TRAIN_LIST = os.path.join(DATA_ROOT, "train.txt")
    VAL_LIST   = os.path.join(DATA_ROOT, "val.txt")
    TEST_LIST  = os.path.join(DATA_ROOT, "test.txt")

    CKPT_DIR = "/kaggle/working/checkpoints"
    LOG_DIR  = "/kaggle/working/logs"
    EVAL_DIR = "/kaggle/working/eval_results"

    # ---------------- audio ----------------
    SAMPLE_RATE   = 16000
    CHUNK_SECONDS = 3.0
    CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SECONDS)   # 48000
    N_SOURCES     = 2

    # ---------------- mixing ----------------
    MIX_SNR_DB_RANGE = (-2.5, 2.5)   # Refactor.md Section 2: tight SNR window
    MIX_PEAK_LIMIT   = 0.9           # normalize mixture so peak amplitude <= 0.9

    # ---------------- audio encoder/decoder (Conv-TasNet style) ----------------
    ENC_KERNEL   = 16
    ENC_STRIDE   = 8
    ENC_CHANNELS = 192     # see param-budget note above (Refactor.md Section 5 default: 256)

    # ---------------- separator (dual-path transformer, dual mask output) ----------------
    SEP_CHUNK_SIZE = 100    # K, per Refactor.md Section 4/5
    SEP_HOP_SIZE   = 50     # H, 50% overlap
    SEP_D_MODEL    = 192    # see param-budget note above (Refactor.md Section 5 default: 256)
    SEP_N_HEADS    = 4
    SEP_FF_DIM     = 384    # see param-budget note above (Refactor.md Section 5 default: 512)
    SEP_N_LAYERS   = 5       # dual-path blocks, each = intra + inter transformer
    SEP_DROPOUT    = 0.1

    # ---------------- training ----------------
    BATCH_SIZE       = 4
    GRAD_ACCUM_STEPS = 4     # effective batch = 16
    NUM_WORKERS      = 2
    EPOCHS            = 25
    LR                = 1.5e-4
    WEIGHT_DECAY      = 1e-5
    WARMUP_STEPS      = 1000
    GRAD_CLIP         = 5.0
    USE_AMP           = True
    USE_GRAD_CKPT     = True

    # Epoch-as-subset strategy (Refactor.md Section 4): sample a fresh random
    # subset of MIXTURES_PER_EPOCH mixtures each epoch instead of iterating
    # the full (huge) main+pretrain file list every epoch.
    MIXTURES_PER_EPOCH = 13000

    # ---------------- colab/kaggle session budget ----------------
    MAX_TRAIN_SECONDS = 7.5 * 3600   # Refactor.md Section 3/4: strict 7-8h limit
    CKPT_EVERY_STEPS  = 500
    VAL_EVERY_STEPS   = 1000

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42

    @staticmethod
    def make_dirs():
        os.makedirs(Config.CKPT_DIR, exist_ok=True)
        os.makedirs(Config.LOG_DIR, exist_ok=True)
        os.makedirs(Config.EVAL_DIR, exist_ok=True)


cfg = Config
