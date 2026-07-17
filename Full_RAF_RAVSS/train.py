"""
train.py

Training entry point for Kaggle-friendly blind multi-speaker separation.
It automatically resumes from the latest checkpoint and, when training ends,
runs a final evaluation pass on that checkpoint so you get metrics plus audio
samples from the last saved model.
"""

import glob
import os
import random
import time

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from config import cfg
from data.lrs2_dataset import LRS2MixtureDataset, collate_fn
from metrics_eval import evaluate_checkpoint
from models.rafravss import RAFRAVSS
from utils.metrics import pit_si_sdr_loss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_latest_ckpt():
    ckpts = sorted(
        glob.glob(os.path.join(cfg.CKPT_DIR, "step_*.pt")),
        key=lambda p: int(os.path.basename(p).split("_")[-1].split(".")[0]),
    )
    return ckpts[-1] if ckpts else None


def save_ckpt(model, optimizer, scheduler, scaler, step, epoch):
    cfg.make_dirs()
    path = os.path.join(cfg.CKPT_DIR, f"step_{step}.pt")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "epoch": epoch,
    }, path)
    all_ckpts = sorted(
        glob.glob(os.path.join(cfg.CKPT_DIR, "step_*.pt")),
        key=lambda p: int(os.path.basename(p).split("_")[-1].split(".")[0]),
    )
    for old in all_ckpts[:-3]:
        os.remove(old)
    return path


def lr_lambda(step):
    if step < cfg.WARMUP_STEPS:
        return step / max(1, cfg.WARMUP_STEPS)
    return (cfg.WARMUP_STEPS / max(1, step)) ** 0.5


@torch.no_grad()
def validate(model, val_loader, max_batches=20):
    model.eval()
    totals = {}
    n = 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        mix = batch["mix_audio"].to(cfg.DEVICE)
        target = batch["target_sources"].to(cfg.DEVICE)
        with autocast(enabled=cfg.USE_AMP):
            est = model(mix, None)
        loss, reordered, _ = pit_si_sdr_loss(est.float(), target.float())
        totals["pit_si_sdr_loss"] = totals.get("pit_si_sdr_loss", 0.0) + float(loss.item())
        totals["reordered_si_sdr"] = totals.get("reordered_si_sdr", 0.0) + float((-loss).item())
        n += 1
    model.train()
    return {k: v / max(1, n) for k, v in totals.items()}


def main():
    set_seed(cfg.SEED)
    cfg.make_dirs()

    train_ds = LRS2MixtureDataset(train=True)
    val_ds = LRS2MixtureDataset(train=False)
    print(f"Train mixtures: {len(train_ds)} | Val mixtures: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    model = RAFRAVSS().to(cfg.DEVICE)
    print(f"Model params: {model.num_params() / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler(enabled=cfg.USE_AMP)

    step = 0
    start_epoch = 0
    ckpt_path = get_latest_ckpt()
    if ckpt_path is not None:
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=cfg.DEVICE)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        step = ckpt["step"]
        start_epoch = ckpt["epoch"]

    train_start_time = time.time()
    stop_training = False
    last_saved_ckpt = ckpt_path
    model.train()

    for epoch in range(start_epoch, cfg.EPOCHS):
        if stop_training:
            break
        optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(train_loader):
            elapsed = time.time() - train_start_time
            if elapsed > cfg.MAX_TRAIN_SECONDS:
                print(f"Hit MAX_TRAIN_SECONDS budget ({elapsed:.0f}s) -> checkpointing and stopping cleanly.")
                last_saved_ckpt = save_ckpt(model, optimizer, scheduler, scaler, step, epoch)
                stop_training = True
                break

            mix = batch["mix_audio"].to(cfg.DEVICE, non_blocking=True)
            target = batch["target_sources"].to(cfg.DEVICE, non_blocking=True)

            with autocast(enabled=cfg.USE_AMP):
                est = model(mix, None)
                loss, _, _ = pit_si_sdr_loss(est.float(), target.float())
                loss = loss / cfg.GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if (i + 1) % cfg.GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if step % 50 == 0:
                    mem_gb = torch.cuda.memory_allocated() / 1e9 if cfg.DEVICE.type == "cuda" else 0.0
                    print(
                        f"epoch {epoch} step {step} | loss {loss.item() * cfg.GRAD_ACCUM_STEPS:.3f} "
                        f"| lr {scheduler.get_last_lr()[0]:.2e} | mem {mem_gb:.2f}GB"
                    )

                if step % cfg.CKPT_EVERY_STEPS == 0:
                    last_saved_ckpt = save_ckpt(model, optimizer, scheduler, scaler, step, epoch)

                if step % cfg.VAL_EVERY_STEPS == 0:
                    metrics = validate(model, val_loader)
                    print(f"[val @ step {step}] " + " ".join(f"{k}={v:.3f}" for k, v in metrics.items()))

        if not stop_training:
            last_saved_ckpt = save_ckpt(model, optimizer, scheduler, scaler, step, epoch + 1)

    if last_saved_ckpt is None:
        last_saved_ckpt = get_latest_ckpt()

    print("Training finished. Running final evaluation on:", last_saved_ckpt)
    if last_saved_ckpt is not None:
        summary, csv_path, summary_path, audio_dir, n_saved = evaluate_checkpoint(
            last_saved_ckpt,
            split="val",
            max_samples=cfg.FINAL_EVAL_MAX_SAMPLES,
            save_audio_n=cfg.FINAL_EVAL_SAVE_AUDIO,
            batch_size=cfg.FINAL_EVAL_BATCH_SIZE,
            out_dir=cfg.EVAL_DIR,
        )
        print("\n===== Final evaluation summary =====")
        for k, v in summary.items():
            print(f"{k:10s} : {v['mean']:.3f} +/- {v['std']:.3f}")
        print(f"Per-sample metrics : {csv_path}")
        print(f"Summary CSV        : {summary_path}")
        print(f"Audio samples ({n_saved} mixes) : {audio_dir}")
    else:
        print("No checkpoint was available for final evaluation.")


if __name__ == "__main__":
    main()
