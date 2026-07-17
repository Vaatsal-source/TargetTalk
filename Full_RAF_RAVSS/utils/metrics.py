"""
utils/metrics.py

- si_sdr: scale-invariant SDR, used both as the training loss (negated) and
  as a reported metric. Computed directly in torch so it's differentiable
  and needs no extra dependency.
- sdr: classic (non scale-invariant) SDR, via the same projection formula
  without the scale-invariance normalization -- reported for comparison
  against SepFormer/RAVSS papers that quote plain SDR.
- pesq_batch / stoi_batch: perceptual quality metrics computed with the
  `pesq` and `pystoi` packages (CPU, numpy) -- only used for validation/
  evaluation, never inside the training loop (they're slow and non-
  differentiable).
"""

import numpy as np
import torch

try:
    from pesq import pesq as _pesq_fn
except ImportError:
    _pesq_fn = None

try:
    from pystoi import stoi as _stoi_fn
except ImportError:
    _stoi_fn = None


def si_sdr(est, target, eps=1e-8):
    """
    est, target: (B, T) torch tensors.
    Returns per-sample SI-SDR in dB, shape (B,).
    """
    target = target - target.mean(dim=-1, keepdim=True)
    est = est - est.mean(dim=-1, keepdim=True)

    dot = (est * target).sum(dim=-1, keepdim=True)
    target_energy = (target ** 2).sum(dim=-1, keepdim=True) + eps
    proj = dot * target / target_energy                  # scaled target (s_target)
    noise = est - proj                                    # e_noise

    ratio = (proj ** 2).sum(dim=-1) / ((noise ** 2).sum(dim=-1) + eps)
    return 10 * torch.log10(ratio + eps)


def si_sdr_loss(est, target):
    """Negative mean SI-SDR -> minimize this."""
    return -si_sdr(est, target).mean()


def sdr(est, target, eps=1e-8):
    """Plain (non scale-invariant) SDR in dB, per-sample, shape (B,)."""
    target = target - target.mean(dim=-1, keepdim=True)
    est = est - est.mean(dim=-1, keepdim=True)
    noise = est - target
    ratio = (target ** 2).sum(dim=-1) / ((noise ** 2).sum(dim=-1) + eps)
    return 10 * torch.log10(ratio + eps)


def pesq_batch(est, target, sr=16000, mode="wb"):
    """
    est, target: (B, T) numpy arrays or torch tensors, on CPU.
    Returns list of PESQ scores (float), skipping samples that fail
    (PESQ requires >= 0.25s audio and can throw on silence/clipping).
    """
    if _pesq_fn is None:
        raise ImportError("pip install pesq to compute PESQ scores")
    est = est.detach().cpu().numpy() if torch.is_tensor(est) else est
    target = target.detach().cpu().numpy() if torch.is_tensor(target) else target
    scores = []
    for e, t in zip(est, target):
        try:
            scores.append(_pesq_fn(sr, t, e, mode))
        except Exception:
            continue
    return scores


def stoi_batch(est, target, sr=16000, extended=False):
    """Returns list of STOI scores (float) in [0, 1] (or extended STOI)."""
    if _stoi_fn is None:
        raise ImportError("pip install pystoi to compute STOI scores")
    est = est.detach().cpu().numpy() if torch.is_tensor(est) else est
    target = target.detach().cpu().numpy() if torch.is_tensor(target) else target
    scores = []
    for e, t in zip(est, target):
        try:
            scores.append(_stoi_fn(t, e, sr, extended=extended))
        except Exception:
            continue
    return scores


def compute_all_metrics(est, target, sr=16000):
    """Convenience wrapper used in the validation loop. Returns a dict of
    means; gracefully degrades if pesq/pystoi aren't installed."""
    out = {
        "si_sdr": si_sdr(est, target).mean().item(),
        "sdr": sdr(est, target).mean().item(),
    }
    if _pesq_fn is not None:
        p = pesq_batch(est, target, sr)
        out["pesq"] = float(np.mean(p)) if p else float("nan")
    if _stoi_fn is not None:
        s = stoi_batch(est, target, sr)
        out["stoi"] = float(np.mean(s)) if s else float("nan")
    return out
