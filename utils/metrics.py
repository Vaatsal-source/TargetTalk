"""
utils/metrics.py

- si_sdr: implements the exact formula in Refactor.md Section 6
  (alpha = <e,s>/||s||^2, s_target = alpha*s, e_noise = e - s_target,
  SI-SDR = 10*log10(||s_target||^2 / ||e_noise||^2)).
- pit_si_sdr_loss: tries both permutations of a 2-source output against the
  2 targets, picks whichever is better PER SAMPLE, and averages the best
  loss across the batch (Refactor.md Section 6).
- resolve_permutation: given a batch of estimates/targets, returns the
  best-permutation-reordered estimates so validation metrics are computed
  against the correct speaker, not an arbitrary channel ordering.
- sdr, pesq_batch, stoi_batch: reported only after permutation resolution.
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


def si_sdr(est, target, eps=1e-8):
    """
    est, target: (..., T) torch tensors, last dim = time.
    Returns SI-SDR in dB with the same leading shape (...,).
    """
    target = target - target.mean(dim=-1, keepdim=True)
    est = est - est.mean(dim=-1, keepdim=True)

    dot = (est * target).sum(dim=-1, keepdim=True)
    target_energy = (target ** 2).sum(dim=-1, keepdim=True) + eps
    alpha = dot / target_energy
    s_target = alpha * target
    e_noise = est - s_target

    ratio = (s_target ** 2).sum(dim=-1) / ((e_noise ** 2).sum(dim=-1) + eps)
    return 10 * torch.log10(ratio + eps)


def sdr(est, target, eps=1e-8):
    """Plain (non scale-invariant) SDR in dB, same shape convention as si_sdr."""
    target = target - target.mean(dim=-1, keepdim=True)
    est = est - est.mean(dim=-1, keepdim=True)
    noise = est - target
    ratio = (target ** 2).sum(dim=-1) / ((noise ** 2).sum(dim=-1) + eps)
    return 10 * torch.log10(ratio + eps)


# The only 2 permutations of 2 sources: identity and swap.
_PERMS_2SRC = list(itertools.permutations(range(2)))  # [(0,1), (1,0)]


def pit_si_sdr_loss(est, target):
    """
    est, target: (B, 2, T).
    For each sample, evaluates both speaker orderings, takes the best
    (lowest negative-SI-SDR) per sample, and returns the batch mean.
    This is the ONLY loss used for training (Refactor.md Section 6).
    """
    B = est.shape[0]
    losses_per_perm = []
    for perm in _PERMS_2SRC:
        est_perm = est[:, perm, :]                    # reorder estimate channels
        pair_sdr = si_sdr(est_perm, target)             # (B, 2)
        losses_per_perm.append(-pair_sdr.mean(dim=-1))   # (B,) negative mean SI-SDR for this perm
    losses_per_perm = torch.stack(losses_per_perm, dim=0)   # (n_perms, B)
    best_loss, _ = losses_per_perm.min(dim=0)                # (B,)
    return best_loss.mean()


@torch.no_grad()
def resolve_permutation(est, target):
    """
    est, target: (B, 2, T).
    Returns (est_reordered, best_perm_indices) where est_reordered matches
    target's speaker ordering as closely as possible per-sample, chosen by
    whichever permutation maximizes total SI-SDR (i.e. minimizes PIT loss).
    Use this before computing SDR/PESQ/STOI so those metrics reflect actual
    separation quality, not an arbitrary channel assignment.
    """
    B = est.shape[0]
    scores_per_perm = []
    for perm in _PERMS_2SRC:
        est_perm = est[:, perm, :]
        pair_sdr = si_sdr(est_perm, target)          # (B, 2)
        scores_per_perm.append(pair_sdr.mean(dim=-1))   # (B,)
    scores_per_perm = torch.stack(scores_per_perm, dim=0)   # (n_perms, B)
    best_idx = scores_per_perm.argmax(dim=0)                  # (B,) which perm won, per sample

    est_reordered = torch.empty_like(est)
    for b in range(B):
        perm = _PERMS_2SRC[best_idx[b].item()]
        est_reordered[b] = est[b, perm, :]
    return est_reordered, best_idx


def pesq_batch(est, target, sr=16000, mode="wb"):
    """est, target: (N, T) numpy/torch, already permutation-resolved and
    flattened across the source dimension (i.e. call once per speaker slot,
    or flatten (B,2,T)->(B*2,T) before calling)."""
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
    """
    est, target: (B, 2, T). Resolves permutation FIRST, then computes all
    metrics on the correctly-ordered pairs, flattening the source dimension
    into the batch dimension for PESQ/STOI (so a batch of 4 mixtures with 2
    speakers each gives 8 PESQ/STOI scores).
    """
    est_resolved, _ = resolve_permutation(est, target)

    si_sdr_scores = si_sdr(est_resolved, target)       # (B, 2)
    sdr_scores = sdr(est_resolved, target)              # (B, 2)

    est_flat = est_resolved.reshape(-1, est_resolved.shape[-1])
    target_flat = target.reshape(-1, target.shape[-1])

    out = {
        "si_sdr": si_sdr_scores.mean().item(),
        "sdr": sdr_scores.mean().item(),
    }
    if _pesq_fn is not None:
        p = pesq_batch(est_flat, target_flat, sr)
        out["pesq"] = float(np.mean(p)) if p else float("nan")
    if _stoi_fn is not None:
        s = stoi_batch(est_flat, target_flat, sr)
        out["stoi"] = float(np.mean(s)) if s else float("nan")
    return out
