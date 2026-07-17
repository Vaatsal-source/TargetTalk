"""
metrics_eval.py

Standalone evaluation for multi-speaker separation checkpoints. It computes
PIT-aligned SI-SDR / SDR and optional PESQ / STOI, then saves a small set of
audio examples so you can listen to the separation quality directly.
"""

import argparse
import csv
import os

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader

from config import cfg
from data.lrs2_dataset import LRS2MixtureDataset, collate_fn
from models.rafravss import RAFRAVSS
from utils.metrics import si_sdr, sdr, pesq_batch, stoi_batch, reorder_sources_by_pit


def load_model(ckpt_path, device):
    model = RAFRAVSS().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    step = ckpt.get("step", "unknown")
    print(f"Loaded checkpoint {ckpt_path} (step {step})")
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

        mix = batch["mix_audio"].to(device)
        target = batch["target_sources"].to(device)

        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=cfg.USE_AMP):
            est = model(mix, None)

        est = est.float()
        target = target.float()
        reordered, _ = reorder_sources_by_pit(est, target)

        for b in range(mix.shape[0]):
            if n_seen >= max_samples:
                break

            est_b = reordered[b]
            tgt_b = target[b]
            valid = tgt_b.abs().sum(dim=-1) > 0
            est_valid = est_b[valid]
            tgt_valid = tgt_b[valid]

            si_sdr_scores = si_sdr(est_valid, tgt_valid).detach().cpu().numpy()
            sdr_scores = sdr(est_valid, tgt_valid).detach().cpu().numpy()
            pesq_scores = pesq_batch(est_valid, tgt_valid, sr=cfg.SAMPLE_RATE) if valid.any() else []
            stoi_scores = stoi_batch(est_valid, tgt_valid, sr=cfg.SAMPLE_RATE) if valid.any() else []

            for s_idx in range(len(si_sdr_scores)):
                per_sample_rows.append({
                    "mix_index": n_seen,
                    "source_index": s_idx,
                    "si_sdr_db": float(si_sdr_scores[s_idx]),
                    "sdr_db": float(sdr_scores[s_idx]),
                    "pesq": float(pesq_scores[s_idx]) if s_idx < len(pesq_scores) else float("nan"),
                    "stoi": float(stoi_scores[s_idx]) if s_idx < len(stoi_scores) else float("nan"),
                })

            if n_saved < save_audio_n:
                torchaudio.save(os.path.join(audio_dir, f"sample{n_seen}_mix.wav"),
                                mix[b:b+1].cpu(), cfg.SAMPLE_RATE)
                for s_idx in range(min(est_b.shape[0], cfg.MAX_SOURCES)):
                    torchaudio.save(os.path.join(audio_dir, f"sample{n_seen}_target{s_idx}.wav"),
                                    tgt_b[s_idx:s_idx+1].cpu(), cfg.SAMPLE_RATE)
                    torchaudio.save(os.path.join(audio_dir, f"sample{n_seen}_estimate{s_idx}.wav"),
                                    est_b[s_idx:s_idx+1].cpu(), cfg.SAMPLE_RATE)
                n_saved += 1

            n_seen += 1

    csv_path = os.path.join(out_dir, "per_sample_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mix_index", "source_index", "si_sdr_db", "sdr_db", "pesq", "stoi"])
        writer.writeheader()
        writer.writerows(per_sample_rows)

    def _agg(key):
        vals = [r[key] for r in per_sample_rows]
        return float(np.nanmean(vals)), float(np.nanstd(vals))

    summary = {}
    for key in ["si_sdr_db", "sdr_db", "pesq", "stoi"]:
        mean, std = _agg(key)
        summary[key] = {"mean": mean, "std": std}

    summary_path = os.path.join(out_dir, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "std", "n_rows"])
        for key, v in summary.items():
            writer.writerow([key, v["mean"], v["std"], len(per_sample_rows)])

    return summary, csv_path, summary_path, audio_dir, n_saved


def evaluate_checkpoint(ckpt_path, split="val", max_samples=None, save_audio_n=None, batch_size=None, out_dir=None):
    device = cfg.DEVICE
    ds = LRS2MixtureDataset(train=(split == "train"))
    loader = DataLoader(
        ds,
        batch_size=batch_size or cfg.FINAL_EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=collate_fn,
        drop_last=False,
    )
    model = load_model(ckpt_path, device)
    summary, csv_path, summary_path, audio_dir, n_saved = run_eval(
        model,
        loader,
        device,
        max_samples or cfg.FINAL_EVAL_MAX_SAMPLES,
        save_audio_n if save_audio_n is not None else cfg.FINAL_EVAL_SAVE_AUDIO,
        out_dir or cfg.EVAL_DIR,
    )
    return summary, csv_path, summary_path, audio_dir, n_saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["val", "train"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--save_audio", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    summary, csv_path, summary_path, audio_dir, n_saved = evaluate_checkpoint(
        args.ckpt, split=args.split, max_samples=args.max_samples, save_audio_n=args.save_audio,
        batch_size=args.batch_size, out_dir=args.out_dir
    )

    print(f"\n===== Evaluation summary ({args.split} split) =====")
    for k, v in summary.items():
        print(f"{k:10s} : {v['mean']:.3f} +/- {v['std']:.3f}")
    print(f"\nPer-sample metrics : {csv_path}")
    print(f"Summary CSV        : {summary_path}")
    print(f"Audio samples ({n_saved} mixes) : {audio_dir}")


if __name__ == "__main__":
    main()
