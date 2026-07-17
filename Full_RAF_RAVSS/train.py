"""
train.py
Entry point. Run with:  python train.py
Resumes automatically from the latest checkpoint in cfg.CKPT_DIR if present,
so a Colab/Kaggle session that gets killed mid-training can just be re-run.
"""

import os
import time
import glob
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from config import cfg
from data.lrs2_dataset import LRS2MixtureDataset, collate_fn
from models.rafravss import RAFRAVSS
from utils.metrics import si_sdr_loss, compute_all_metrics


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_latest_ckpt():
    ckpts = sorted(glob.glob(os.path.join(cfg.CKPT_DIR, "step_*.pt")),
                    key=lambda p: int(p.split("_")[-1].split(".")[0]))
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
    # keep only the 3 most recent checkpoints to save disk
    all_ckpts = sorted(glob.glob(os.path.join(cfg.CKPT_DIR, "step_*.pt")),
                        key=lambda p: int(p.split("_")[-1].split(".")[0]))
    for old in all_ckpts[:-3]:
        os.remove(old)
    return path


def lr_lambda(step):
    if step < cfg.WARMUP_STEPS:
        return step / max(1, cfg.WARMUP_STEPS)
    # simple inverse-sqrt decay after warmup
    return (cfg.WARMUP_STEPS / max(1, step)) ** 0.5


@torch.no_grad()
def validate(model, val_loader, max_batches=20):
    model.eval()
    all_metrics = {}
    n = 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        mix = batch["mix_audio"].to(cfg.DEVICE)
        target = batch["target_audio"].to(cfg.DEVICE)
        lips = batch["lip_frames"].to(cfg.DEVICE)
        with autocast(enabled=cfg.USE_AMP):
            est = model(mix, lips)
        m = compute_all_metrics(est.float(), target.float(), sr=cfg.SAMPLE_RATE)
        for k, v in m.items():
            all_metrics[k] = all_metrics.get(k, 0.0) + v
        n += 1
    model.train()
    return {k: v / max(1, n) for k, v in all_metrics.items()}


def main():
    set_seed(cfg.SEED)
    cfg.make_dirs()

    train_ds = LRS2MixtureDataset(train=True)
    val_ds = LRS2MixtureDataset(train=False)
    print(f"Train utterances: {len(train_ds)} | Val utterances: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn,
        pin_memory=True, drop_last=True, persistent_workers=cfg.NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn,
        pin_memory=True, drop_last=True,
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
    model.train()
    stop_training = False

    for epoch in range(start_epoch, cfg.EPOCHS):
        if stop_training:
            break
        optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(train_loader):
            elapsed = time.time() - train_start_time
            if elapsed > cfg.MAX_TRAIN_SECONDS:
                print(f"Hit MAX_TRAIN_SECONDS budget ({elapsed:.0f}s) -> checkpointing and stopping cleanly.")
                save_ckpt(model, optimizer, scheduler, scaler, step, epoch)
                stop_training = True
                break

            mix = batch["mix_audio"].to(cfg.DEVICE, non_blocking=True)
            target = batch["target_audio"].to(cfg.DEVICE, non_blocking=True)
            lips = batch["lip_frames"].to(cfg.DEVICE, non_blocking=True)

            with autocast(enabled=cfg.USE_AMP):
                est = model(mix, lips)
                loss = si_sdr_loss(est.float(), target.float()) / cfg.GRAD_ACCUM_STEPS

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
                    print(f"epoch {epoch} step {step} | loss(-SI-SDR/accum) {loss.item()*cfg.GRAD_ACCUM_STEPS:.3f} "
                          f"| lr {scheduler.get_last_lr()[0]:.2e} | mem {torch.cuda.memory_allocated()/1e9:.2f}GB")

                if step % cfg.CKPT_EVERY_STEPS == 0:
                    save_ckpt(model, optimizer, scheduler, scaler, step, epoch)

                if step % cfg.VAL_EVERY_STEPS == 0:
                    metrics = validate(model, val_loader)
                    print(f"[val @ step {step}] " + " ".join(f"{k}={v:.3f}" for k, v in metrics.items()))

    print("Training finished (or paused for session limit). Final checkpoint saved in", cfg.CKPT_DIR)


if __name__ == "__main__":
    main()
