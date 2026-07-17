"""
data/lrs2_dataset.py

Builds blind 3- and 4-speaker mixtures from single-speaker LRS2 clips.
Each sample returns a fixed-length mixture and a padded stack of source
waveforms so the separator can be trained with PIT.
"""

import glob
import os
import random

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

from config import cfg


def _list_utterances(root, split):
    pattern = os.path.join(root, split, "*", "*.mp4")
    files = sorted(glob.glob(pattern))
    if len(files) == 0:
        raise RuntimeError(f"No .mp4 files found under {pattern}. Check DATA_ROOT/TRAIN_SPLIT in config.py.")
    return files


class LRS2MixtureDataset(Dataset):
    def __init__(self, root=None, split=None, train=True, val_fraction=0.05):
        self.root = root or cfg.DATA_ROOT
        self.split = split or cfg.TRAIN_SPLIT
        all_files = _list_utterances(self.root, self.split)

        n_val = max(1, int(len(all_files) * val_fraction))
        rng = random.Random(cfg.SEED)
        shuffled = all_files[:]
        rng.shuffle(shuffled)
        val_files = set(shuffled[:n_val])

        if train:
            self.files = [f for f in all_files if f not in val_files]
        else:
            self.files = [f for f in all_files if f in val_files]

        self.train = train

    def __len__(self):
        return len(self.files)

    def _load_audio(self, mp4_path):
        wav, sr = torchaudio.load(mp4_path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != cfg.SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, cfg.SAMPLE_RATE)
        wav = wav.squeeze(0)

        target_len = cfg.CHUNK_SAMPLES
        if wav.shape[0] >= target_len:
            start = random.randint(0, wav.shape[0] - target_len) if self.train else 0
            wav = wav[start:start + target_len]
        else:
            wav = torch.nn.functional.pad(wav, (0, target_len - wav.shape[0]))
        return wav

    def _sample_num_sources(self, idx):
        if self.train:
            return random.randint(cfg.MIN_SOURCES, cfg.MAX_SOURCES)
        return cfg.MAX_SOURCES if (idx % 2 == 0) else cfg.MIN_SOURCES

    def __getitem__(self, idx):
        num_sources = self._sample_num_sources(idx)
        chosen = [self.files[idx]]
        while len(chosen) < num_sources:
            candidate = self.files[random.randint(0, len(self.files) - 1)]
            if candidate not in chosen:
                chosen.append(candidate)

        source_audios = [self._load_audio(path) for path in chosen]
        target_sources = torch.stack(source_audios, dim=0)  # (S, T)

        snr_db = random.uniform(-5, 5)
        mix = target_sources[0].clone()
        for src in target_sources[1:]:
            src_power = src.pow(2).mean().clamp_min(1e-8)
            mix_power = mix.pow(2).mean().clamp_min(1e-8)
            scale = torch.sqrt(mix_power / (src_power * (10 ** (snr_db / 10))))
            mix = mix + scale * src

        norm = mix.abs().max().clamp_min(1e-6)
        mix = mix / norm
        target_sources = target_sources / norm

        if target_sources.shape[0] < cfg.MAX_SOURCES:
            pad = cfg.MAX_SOURCES - target_sources.shape[0]
            target_sources = torch.cat(
                [target_sources, torch.zeros(pad, target_sources.shape[1], dtype=target_sources.dtype)],
                dim=0,
            )

        return {
            "mix_audio": mix,
            "target_sources": target_sources,
            "num_sources": torch.tensor(num_sources, dtype=torch.long),
            "snr_db": torch.tensor(snr_db, dtype=torch.float32),
        }


def collate_fn(batch):
    return {
        "mix_audio": torch.stack([b["mix_audio"] for b in batch]),
        "target_sources": torch.stack([b["target_sources"] for b in batch]),
        "num_sources": torch.stack([b["num_sources"] for b in batch]),
        "snr_db": torch.stack([b["snr_db"] for b in batch]),
    }
