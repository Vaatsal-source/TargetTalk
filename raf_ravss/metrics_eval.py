"""
metrics_eval.py

Standalone evaluation script — run this AFTER training (or at any checkpoint)
to get aggregate SI-SDR / SDR / PESQ / STOI over a full split, plus a CSV of
per-sample scores and a handful of saved audio triplets (mixture / target /
estimate) so you can actually listen to how well separation is working.

Usage:
    python metrics_eval.py --ckpt checkpoints/step_5000.pt --split val
    python metrics_eval.py --ckpt checkpoints/step_5000.pt --split val --max_samples 200 --save_audio 10

Notes:
- This does NOT use gradient checkpointing / grad accumulation (no backward
  pass at all), so it's lighter on memory than training and can use a larger
  batch size — bump cfg.BATCH_SIZE via --batch_size if you want it faster.
- LRS2 has no official test split in this raw layout; "val" here is the same
  held-out slice `LRS2MixtureDataset(train=False)` uses. If you want a truly
  fixed test set for reporting final numbers, carve one out once and reuse it
  (see the --test_list option below).
"""

import os
import csv
import argparse
import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader

from config import cfg
from data.lrs2_dataset import LRS2MixtureDataset, collate_fn
from models.rafravss import RAFRAVSS
from utils.metrics import si_sdr, sdr, pesq_batch, stoi_batch


def load_model(ckpt_path, device):
    model = RAFRAVSS().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if "model" in ckpt else ckpt  # allow raw state_dict too
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
        target = batch["target_audio"].to(device)
        lips = batch["lip_frames"].to(device)

        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=cfg.USE_AMP):
            est = model(mix, lips)
        est = est.float()
        target_f = target.float()

        si_sdr_scores = si_sdr(est, target_f).cpu().numpy()
        sdr_scores = sdr(est, target_f).cpu().numpy()
        pesq_scores = pesq_batch(est, target_f, sr=cfg.SAMPLE_RATE)
        stoi_scores = stoi_batch(est, target_f, sr=cfg.SAMPLE_RATE)

        # pesq/stoi silently drop failed samples internally; pad with NaN so
        # rows stay aligned when a sample fails PESQ (e.g. too short/silent).
        pesq_scores = pesq_scores + [float("nan")] * (len(si_sdr_scores) - len(pesq_scores))
        stoi_scores = stoi_scores + [float("nan")] * (len(si_sdr_scores) - len(stoi_scores))

        for b in range(mix.shape[0]):
            if n_seen >= max_samples:
                break
            per_sample_rows.append({
                "index": n_seen,
                "si_sdr_db": float(si_sdr_scores[b]),
                "sdr_db": float(sdr_scores[b]),
                "pesq": float(pesq_scores[b]),
                "stoi": float(stoi_scores[b]),
            })

            if n_saved < save_audio_n:
                torchaudio.save(os.path.join(audio_dir, f"sample{n_seen}_mix.wav"),
                                 mix[b:b+1].cpu(), cfg.SAMPLE_RATE)
                torchaudio.save(os.path.join(audio_dir, f"sample{n_seen}_target.wav"),
                                 target[b:b+1].cpu(), cfg.SAMPLE_RATE)
                torchaudio.save(os.path.join(audio_dir, f"sample{n_seen}_estimate.wav"),
                                 est[b:b+1].cpu(), cfg.SAMPLE_RATE)
                n_saved += 1

            n_seen += 1

    # write per-sample CSV
    csv_path = os.path.join(out_dir, "per_sample_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "si_sdr_db", "sdr_db", "pesq", "stoi"])
        writer.writeheader()
        writer.writerows(per_sample_rows)

    # aggregate (nanmean so a few failed PESQ/STOI samples don't wipe the whole average)
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
        writer.writerow(["metric", "mean", "std", "n_samples"])
        for key, v in summary.items():
            writer.writerow([key, v["mean"], v["std"], len(per_sample_rows)])

    return summary, csv_path, summary_path, audio_dir, n_saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="path to a step_*.pt checkpoint")
    parser.add_argument("--split", type=str, default="val", choices=["val", "train"],
                         help="which held-out split to evaluate on (train=False slice is 'val')")
    parser.add_argument("--max_samples", type=int, default=500, help="cap on number of utterances evaluated")
    parser.add_argument("--batch_size", type=int, default=None, help="override cfg.BATCH_SIZE for eval")
    parser.add_argument("--save_audio", type=int, default=10, help="number of mix/target/estimate wav triplets to save for listening")
    parser.add_argument("--out_dir", type=str, default="/kaggle/working/eval_results", help="where to write CSVs and audio samples")
    args = parser.parse_args()

    device = cfg.DEVICE
    if args.batch_size:
        cfg.BATCH_SIZE = args.batch_size

    ds = LRS2MixtureDataset(train=(args.split == "train"))
    loader = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                         num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, drop_last=False)

    model = load_model(args.ckpt, device)
    summary, csv_path, summary_path, audio_dir, n_saved = run_eval(
        model, loader, device, args.max_samples, args.save_audio, args.out_dir
    )

    print("\n===== Evaluation summary (" + args.split + " split) =====")
    for k, v in summary.items():
        print(f"{k:10s} : {v['mean']:.3f} +/- {v['std']:.3f}")
    print(f"\nPer-sample metrics : {csv_path}")
    print(f"Summary CSV        : {summary_path}")
    print(f"Audio samples ({n_saved} triplets) : {audio_dir}")


if __name__ == "__main__":
    main()
