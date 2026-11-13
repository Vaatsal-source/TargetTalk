"""
metrics_eval.py

Standalone evaluation for the blind two-speaker separator. Resolves the
PIT permutation per sample BEFORE computing any metric (Refactor.md
Section 7), so SI-SDR/SDR/PESQ/STOI reflect true separation quality rather
than an arbitrary channel assignment.

Usage:
    python metrics_eval.py --ckpt checkpoints/step_5000.pt --split val
    python metrics_eval.py --ckpt checkpoints/step_5000.pt --split test --max_samples 500 --save_audio 3
"""

import os
import csv
import argparse
import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader

from config import cfg
from data.lrs2_dataset import LRS2PairMixtureDataset, collate_fn
from models.rafravss import TwoSpeakerSeparator
from utils.metrics import si_sdr, sdr, pesq_batch, stoi_batch, resolve_permutation


def load_model(ckpt_path, device):
    model = TwoSpeakerSeparator().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint {ckpt_path} (step {ckpt.get('step', 'unknown')})")
    return model


@torch.no_grad()
def run_eval(model, loader, device, max_samples, save_audio_n, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    audio_dir = os.path.join(out_dir, "audio_samples")
    os.makedirs(audio_dir, exist_ok=True)

    per_sample_rows = []
    n_seen = 0
    n_saved = 0

    for batch in loader:
        if n_seen >= max_samples:
            break
        mix = batch["mix_audio"].to(device)                 # (B, T)
        target = batch["target_audio"].to(device)             # (B, 2, T)

        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=cfg.USE_AMP):
            est = model(mix)                                    # (B, 2, T)
        est = est.float()
        target_f = target.float()

        # ---- resolve PIT permutation BEFORE any metric (Section 7) ----
        est_resolved, _ = resolve_permutation(est, target_f)     # (B, 2, T)

        si_sdr_pair = si_sdr(est_resolved, target_f).cpu().numpy()   # (B, 2)
        sdr_pair = sdr(est_resolved, target_f).cpu().numpy()          # (B, 2)

        est_flat = est_resolved.reshape(-1, est_resolved.shape[-1])
        target_flat = target_f.reshape(-1, target_f.shape[-1])
        pesq_flat = pesq_batch(est_flat, target_flat, sr=cfg.SAMPLE_RATE)
        stoi_flat = stoi_batch(est_flat, target_flat, sr=cfg.SAMPLE_RATE)
        # pad to keep alignment if a couple of pesq/stoi calls silently failed
        pesq_flat = pesq_flat + [float("nan")] * (est_flat.shape[0] - len(pesq_flat))
        stoi_flat = stoi_flat + [float("nan")] * (est_flat.shape[0] - len(stoi_flat))

        B = mix.shape[0]
        for b in range(B):
            if n_seen >= max_samples:
                break
            per_sample_rows.append({
                "index": n_seen,
                "si_sdr_spk1_db": float(si_sdr_pair[b, 0]),
                "si_sdr_spk2_db": float(si_sdr_pair[b, 1]),
                "si_sdr_avg_db": float(si_sdr_pair[b].mean()),
                "sdr_spk1_db": float(sdr_pair[b, 0]),
                "sdr_spk2_db": float(sdr_pair[b, 1]),
                "sdr_avg_db": float(sdr_pair[b].mean()),
                "pesq_spk1": float(pesq_flat[2 * b]),
                "pesq_spk2": float(pesq_flat[2 * b + 1]),
                "stoi_spk1": float(stoi_flat[2 * b]),
                "stoi_spk2": float(stoi_flat[2 * b + 1]),
            })

            if n_saved < save_audio_n:
                torchaudio.save(os.path.join(audio_dir, f"sample{n_seen}_input_mix.wav"),
                                 mix[b:b+1].cpu(), cfg.SAMPLE_RATE)
                torchaudio.save(os.path.join(audio_dir, f"sample{n_seen}_target_speaker1.wav"),
                                 target[b:b+1, 0].cpu(), cfg.SAMPLE_RATE)
                torchaudio.save(os.path.join(audio_dir, f"sample{n_seen}_target_speaker2.wav"),
                                 target[b:b+1, 1].cpu(), cfg.SAMPLE_RATE)
                torchaudio.save(os.path.join(audio_dir, f"sample{n_seen}_estimated_speaker1.wav"),
                                 est_resolved[b:b+1, 0].cpu(), cfg.SAMPLE_RATE)
                torchaudio.save(os.path.join(audio_dir, f"sample{n_seen}_estimated_speaker2.wav"),
                                 est_resolved[b:b+1, 1].cpu(), cfg.SAMPLE_RATE)
                n_saved += 1

            n_seen += 1

    csv_path = os.path.join(out_dir, "per_sample_metrics.csv")
    fieldnames = list(per_sample_rows[0].keys()) if per_sample_rows else [
        "index", "si_sdr_spk1_db", "si_sdr_spk2_db", "si_sdr_avg_db",
        "sdr_spk1_db", "sdr_spk2_db", "sdr_avg_db",
        "pesq_spk1", "pesq_spk2", "stoi_spk1", "stoi_spk2",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_sample_rows)

    def _agg(key):
        vals = [r[key] for r in per_sample_rows]
        return float(np.nanmean(vals)), float(np.nanstd(vals))

    summary = {}
    for key in ["si_sdr_avg_db", "sdr_avg_db", "pesq_spk1", "pesq_spk2", "stoi_spk1", "stoi_spk2"]:
        mean, std = _agg(key)
        summary[key] = {"mean": mean, "std": std}
    # convenience combined PESQ/STOI across both speaker slots
    combined_pesq = [r["pesq_spk1"] for r in per_sample_rows] + [r["pesq_spk2"] for r in per_sample_rows]
    combined_stoi = [r["stoi_spk1"] for r in per_sample_rows] + [r["stoi_spk2"] for r in per_sample_rows]
    summary["pesq_combined"] = {"mean": float(np.nanmean(combined_pesq)), "std": float(np.nanstd(combined_pesq))}
    summary["stoi_combined"] = {"mean": float(np.nanmean(combined_stoi)), "std": float(np.nanstd(combined_stoi))}

    summary_path = os.path.join(out_dir, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "std", "n_samples"])
        for key, v in summary.items():
            writer.writerow([key, v["mean"], v["std"], len(per_sample_rows)])

    return summary, csv_path, summary_path, audio_dir, n_saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--save_audio", type=int, default=3,
                         help="number of mix/target1/target2/est1/est2 wav quintets to save")
    parser.add_argument("--out_dir", type=str, default=cfg.EVAL_DIR)
    args = parser.parse_args()

    device = cfg.DEVICE
    if args.batch_size:
        cfg.BATCH_SIZE = args.batch_size

    ds = LRS2PairMixtureDataset(split=args.split)
    loader = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                         num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, drop_last=False)

    model = load_model(args.ckpt, device)
    summary, csv_path, summary_path, audio_dir, n_saved = run_eval(
        model, loader, device, args.max_samples, args.save_audio, args.out_dir
    )

    print(f"\n===== Evaluation summary ({args.split} split) =====")
    for k, v in summary.items():
        print(f"{k:16s} : {v['mean']:.3f} +/- {v['std']:.3f}")
    print(f"\nPer-sample metrics : {csv_path}")
    print(f"Summary CSV        : {summary_path}")
    print(f"Audio samples ({n_saved} quintets) : {audio_dir}")


if __name__ == "__main__":
    main()
