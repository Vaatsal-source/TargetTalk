"""
train.py
Entry point for blind two-speaker separation training. Run with:
    python train.py
Resumes automatically from the latest checkpoint in cfg.CKPT_DIR if present.
"""

import os
import time
import glob
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.amp import autocast, GradScaler

from config import cfg
from data.lrs2_dataset import LRS2PairMixtureDataset, collate_fn
from models.rafravss import TwoSpeakerSeparator
from utils.metrics import pit_si_sdr_loss, compute_all_metrics
from metrics_eval import run_eval


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
    all_ckpts = sorted(glob.glob(os.path.join(cfg.CKPT_DIR, "step_*.pt")),
                        key=lambda p: int(p.split("_")[-1].split(".")[0]))
    for old in all_ckpts[:-3]:
        os.remove(old)
    return path


def build_scheduler(optimizer, total_steps):
    """Cosine annealing with a brief linear warmup (Refactor.md Section 4)."""
    warmup = cfg.WARMUP_STEPS

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


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
        with autocast(device_type=cfg.DEVICE.type, enabled=cfg.USE_AMP):
            est = model(mix)
        m = compute_all_metrics(est.float(), target.float(), sr=cfg.SAMPLE_RATE)
        for k, v in m.items():
            all_metrics[k] = all_metrics.get(k, 0.0) + v
        n += 1
    model.train()
    return {k: v / max(1, n) for k, v in all_metrics.items()}


def main():
    set_seed(cfg.SEED)
    cfg.make_dirs()

    train_ds = LRS2PairMixtureDataset(split="train")
    val_ds = LRS2PairMixtureDataset(split="val")
    print(f"Train pool: {len(train_ds)} files | Val pool: {len(val_ds)} files")
    print(f"Sampling {cfg.MIXTURES_PER_EPOCH} mixtures/epoch (Refactor.md Section 4 subset strategy)")

    # Epoch-as-subset: a fresh random subset of MIXTURES_PER_EPOCH indices is
    # drawn each epoch (RandomSampler re-samples every time the DataLoader's
    # iterator is (re)created, i.e. every epoch, since num_samples < len(ds)
    # forces replacement=True internally when not explicitly disabled).
    train_sampler = RandomSampler(
        train_ds, replacement=True, num_samples=cfg.MIXTURES_PER_EPOCH
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, sampler=train_sampler,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn,
        pin_memory=True, drop_last=True, persistent_workers=cfg.NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn,
        pin_memory=True, drop_last=True,
    )

    model = TwoSpeakerSeparator().to(cfg.DEVICE)
    print(f"Model params: {model.num_params() / 1e6:.3f}M "
          f"(Refactor.md Section 4 target: 2.0-3.5M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    steps_per_epoch = cfg.MIXTURES_PER_EPOCH // (cfg.BATCH_SIZE * cfg.GRAD_ACCUM_STEPS)
    total_steps = steps_per_epoch * cfg.EPOCHS
    scheduler = build_scheduler(optimizer, total_steps)
    scaler = GradScaler(device=cfg.DEVICE.type, enabled=cfg.USE_AMP)

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

            with autocast(device_type=cfg.DEVICE.type, enabled=cfg.USE_AMP):
                est = model(mix)
                loss = pit_si_sdr_loss(est.float(), target.float()) / cfg.GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if (i + 1) % cfg.GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
                prev_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                # GradScaler skips the optimizer step (but not scheduler.step())
                # on iterations where it detects inf/NaN grads while calibrating
                # its loss scale -- that's visible as the scale value dropping.
                # Only advance the LR schedule when the step actually happened.
                if scaler.get_scale() >= prev_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if step % 50 == 0:
                    print(f"epoch {epoch} step {step} | PIT-SI-SDR loss {loss.item()*cfg.GRAD_ACCUM_STEPS:.3f} "
                          f"| lr {scheduler.get_last_lr()[0]:.2e} | mem {torch.cuda.memory_allocated()/1e9:.2f}GB")

                if step % cfg.CKPT_EVERY_STEPS == 0:
                    save_ckpt(model, optimizer, scheduler, scaler, step, epoch)

                if step % cfg.VAL_EVERY_STEPS == 0:
                    metrics = validate(model, val_loader)
                    print(f"[val @ step {step}] " + " ".join(f"{k}={v:.3f}" for k, v in metrics.items()))

    print("Training finished (or paused for session limit). Final checkpoint saved in", cfg.CKPT_DIR)

    if stop_training:
        print("[Auto-Eval] Skipped -- training stopped due to the time budget, not natural completion. "
              "Run metrics_eval.py manually against the saved checkpoint when convenient.")
        return

    print("\n" + "=" * 50)
    print("[Auto-Eval] Training completed naturally. Running full validation-split evaluation...")
    print("=" * 50)
    try:
        eval_ds = LRS2PairMixtureDataset(split="val")
        eval_loader = DataLoader(
            eval_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
            num_workers=cfg.NUM_WORKERS, collate_fn=collate_fn, drop_last=False
        )
        model.eval()
        summary, csv_path, summary_path, audio_dir, n_saved = run_eval(
            model=model, loader=eval_loader, device=cfg.DEVICE,
            max_samples=500, save_audio_n=3, out_dir=cfg.EVAL_DIR,
        )
        print("\n===== Evaluation summary =====")
        for k, v in summary.items():
            print(f"{k:10s} : {v['mean']:.3f} +/- {v['std']:.3f}")
        print(f"\nPer-sample metrics : {csv_path}")
        print(f"Summary CSV        : {summary_path}")
        print(f"Audio samples      : {audio_dir}")
    except Exception as e:
        print(f"\n[Auto-Eval Warning] Evaluation encountered an error: {e}")


if __name__ == "__main__":
    main()
