"""
utils/metrics.py

Metrics for blind multi-speaker separation.

- si_sdr / sdr: per-sample metrics for aligned source tensors.
- pit_si_sdr_loss: permutation-invariant SI-SDR loss for (B, S, T) tensors.
- reorder_sources_by_pit: reorder estimated sources to the best target
  permutation for saving audio and reporting metrics.
"""

import itertools
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


def _flatten_sources(x):
    if x.ndim == 2:
        return x.unsqueeze(1)
    return x


def si_sdr(est, target, eps=1e-8):
    est = _flatten_sources(est)
    target = _flatten_sources(target)
    est = est - est.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    dot = (est * target).sum(dim=-1, keepdim=True)
    target_energy = (target ** 2).sum(dim=-1, keepdim=True) + eps
    proj = dot * target / target_energy
    noise = est - proj
    ratio = (proj ** 2).sum(dim=-1) / ((noise ** 2).sum(dim=-1) + eps)
    return 10 * torch.log10(ratio + eps)


def sdr(est, target, eps=1e-8):
    est = _flatten_sources(est)
    target = _flatten_sources(target)
    est = est - est.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    noise = est - target
    ratio = (target ** 2).sum(dim=-1) / ((noise ** 2).sum(dim=-1) + eps)
    return 10 * torch.log10(ratio + eps)


def _pairwise_si_sdr(est_sources, target_sources, eps=1e-8):
    est_sources = est_sources - est_sources.mean(dim=-1, keepdim=True)
    target_sources = target_sources - target_sources.mean(dim=-1, keepdim=True)
    dot = torch.einsum("bst,bkt->bsk", est_sources, target_sources)
    target_energy = torch.sum(target_sources ** 2, dim=-1).unsqueeze(1) + eps
    proj = dot.unsqueeze(-1) * target_sources.unsqueeze(1) / target_energy.unsqueeze(-1)
    noise = est_sources.unsqueeze(2) - proj
    ratio = torch.sum(proj ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + eps)
    return 10 * torch.log10(ratio + eps)  # (B, S, S)


def pit_si_sdr_loss(est_sources, target_sources):
    """
    Permutation invariant SI-SDR loss for tensors shaped (B, S, T).
    Returns:
        loss, reordered_est, best_perm
    """
    if est_sources.ndim != 3 or target_sources.ndim != 3:
        raise ValueError("pit_si_sdr_loss expects (B, S, T) tensors")

    if est_sources.shape[1] != target_sources.shape[1]:
        raise ValueError("Estimated and target source counts must match after padding")

    n_src = est_sources.shape[1]
    pairwise = _pairwise_si_sdr(est_sources, target_sources)
    perms = list(itertools.permutations(range(n_src)))
    scores = []
    for perm in perms:
        idx = torch.tensor(perm, device=est_sources.device)
        perm_score = pairwise[:, torch.arange(n_src, device=est_sources.device), idx].mean(dim=1)
        scores.append(perm_score)
    scores = torch.stack(scores, dim=1)
    best_scores, best_idx = scores.max(dim=1)
    best_perm = torch.tensor(perms, device=est_sources.device)[best_idx]
    reordered = torch.stack([est_sources[b, best_perm[b]] for b in range(est_sources.size(0))], dim=0)
    return -best_scores.mean(), reordered, best_perm


def reorder_sources_by_pit(est_sources, target_sources):
    _, reordered, perm = pit_si_sdr_loss(est_sources, target_sources)
    return reordered, perm


def pesq_batch(est, target, sr=16000, mode="wb"):
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
