"""
count_params.py
Quick check that the current config.py dims land inside the Refactor.md
Section 4 budget of 2.0-3.5M total parameters. Run:
    python count_params.py
"""

from config import cfg
from models.rafravss import TwoSpeakerSeparator

if __name__ == "__main__":
    model = TwoSpeakerSeparator()
    n = model.num_params()
    print(f"Total trainable params: {n:,} ({n/1e6:.3f}M)")
    lo, hi = 2.0e6, 3.5e6
    if lo <= n <= hi:
        print(f"Within Refactor.md Section 4 budget (2.0M - 3.5M).")
    else:
        print(f"OUTSIDE Refactor.md Section 4 budget (2.0M - 3.5M) -- "
              f"adjust ENC_CHANNELS / SEP_D_MODEL / SEP_FF_DIM / SEP_N_LAYERS in config.py.")
