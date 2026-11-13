"""
data/lrs2_dataset.py

Audio-only, blind two-speaker mixture loader for LRS2. No video is read at
all -- per Refactor.md Section 1/2, torchaudio.load() pulls the audio stream
directly out of the .mp4 container via its FFmpeg backend, so no frame
decoding, no OpenCV, no face detection ever happens. This is the single
biggest CPU/memory win from dropping the visual pipeline.

Directory layout expected (per Refactor.md Section 2):
    DATA_ROOT/
        main/<speaker_id>/00001.mp4 (+ .txt)
        pretrain/<speaker_id>/00001.mp4 (+ .txt)

Speaker-disjoint mixing: Speaker B is always sampled from a DIFFERENT
speaker folder than Speaker A, so the model never gets a same-speaker
"mixture" by accident.

Optional split files: if DATA_ROOT/train.txt, val.txt, test.txt exist (one
relative mp4 path per line, as in the official LRS2 file lists), those are
used directly. If they don't exist, falls back to a deterministic random
95/5 split so the pipeline still runs out of the box.
"""

import os
import glob
import random
import torch
import torchaudio
from torch.utils.data import Dataset

from config import cfg


def _discover_all_files(root, splits):
    files = []
    for split in splits:
        pattern = os.path.join(root, split, "*", "*.mp4")
        files.extend(sorted(glob.glob(pattern)))
    if len(files) == 0:
        raise RuntimeError(
            f"No .mp4 files found under {root}/{{{','.join(splits)}}}/*/*.mp4. "
            f"Check config.DATA_ROOT and that the dataset is mounted correctly."
        )
    return files


def _read_list_file(path, root):
    """Split files list plain paths, one per line, relative to root. Returns
    None if the file doesn't exist so callers can fall back gracefully."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    resolved = []
    for line in lines:
        p = line if os.path.isabs(line) else os.path.join(root, line)
        if not p.endswith(".mp4"):
            p = p + ".mp4"
        if os.path.exists(p):
            resolved.append(p)
    return resolved if resolved else None


def _speaker_of(path):
    """Speaker id = parent folder name, used to keep mixtures speaker-disjoint."""
    return os.path.basename(os.path.dirname(path))


class LRS2PairMixtureDataset(Dataset):
    """
    Yields:
        mix_audio    : (T,)     float32, Speaker_A + Speaker_B, peak-normalized
        target_audio : (2, T)   float32, stacked clean [Speaker_A, Speaker_B]
    Order of the two targets is arbitrary (that's exactly what PIT is for).
    """

    def __init__(self, split="train", root=None):
        self.root = root or cfg.DATA_ROOT
        self.split = split

        list_path = {"train": cfg.TRAIN_LIST, "val": cfg.VAL_LIST, "test": cfg.TEST_LIST}[split]
        files = _read_list_file(list_path, self.root)

        if files is not None:
            self.files = files
        else:
            all_files = _discover_all_files(self.root, cfg.SPLITS)
            rng = random.Random(cfg.SEED)
            shuffled = all_files[:]
            rng.shuffle(shuffled)
            n_val = max(1, int(len(shuffled) * 0.05))
            n_test = max(1, int(len(shuffled) * 0.05))
            if split == "val":
                self.files = shuffled[:n_val]
            elif split == "test":
                self.files = shuffled[n_val:n_val + n_test]
            else:
                self.files = shuffled[n_val + n_test:]

        # group by speaker so we can reject same-speaker pairs in O(1)
        self.speaker_of = {f: _speaker_of(f) for f in self.files}

    def __len__(self):
        return len(self.files)

    def _load_audio_raw(self, mp4_path):
        """Raises on decode failure -- caller (_load_audio) handles retries."""
        wav, sr = torchaudio.load(mp4_path)          # native FFmpeg demux, audio-only
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)        # downmix to mono
        if sr != cfg.SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, cfg.SAMPLE_RATE)
        wav = wav.squeeze(0)
        if wav.numel() == 0:
            raise RuntimeError(f"Zero-length audio decoded from {mp4_path}")

        target_len = cfg.CHUNK_SAMPLES
        if wav.shape[0] >= target_len:
            start = random.randint(0, wav.shape[0] - target_len) if self.split == "train" else 0
            wav = wav[start:start + target_len]
        else:
            wav = torch.nn.functional.pad(wav, (0, target_len - wav.shape[0]))
        return wav

    def _load_audio(self, path, max_retries=10):
        """Corrupt/truncated .mp4s are common at LRS2 scale (bad packets,
        zero-byte files, incomplete downloads). Rather than crash the whole
        DataLoader worker on one bad file, retry with a freshly sampled
        random file up to max_retries times, then give up loudly."""
        attempt_path = path
        for attempt in range(max_retries):
            try:
                return self._load_audio_raw(attempt_path)
            except Exception as e:
                print(f"[dataset warning] failed to decode {attempt_path} ({e}); "
                      f"skipping (attempt {attempt + 1}/{max_retries})")
                attempt_path = self.files[random.randint(0, len(self.files) - 1)]
        raise RuntimeError(
            f"Could not load a valid audio file after {max_retries} attempts "
            f"(started from {path}). Dataset may have a large fraction of corrupt files."
        )

    def __getitem__(self, idx):
        a_path = self.files[idx]
        a_speaker = self.speaker_of[a_path]

        b_path = self.files[random.randint(0, len(self.files) - 1)]
        tries = 0
        while self.speaker_of[b_path] == a_speaker and tries < 20:
            b_path = self.files[random.randint(0, len(self.files) - 1)]
            tries += 1

        wav_a = self._load_audio(a_path)
        wav_b = self._load_audio(b_path)

        # equal-RMS normalize both, then apply a random SNR offset to B
        rms_a = wav_a.pow(2).mean().clamp_min(1e-8).sqrt()
        rms_b = wav_b.pow(2).mean().clamp_min(1e-8).sqrt()
        wav_a = wav_a / rms_a
        wav_b = wav_b / rms_b

        snr_db = random.uniform(*cfg.MIX_SNR_DB_RANGE)
        scale_b = 10 ** (-snr_db / 20)
        wav_b = wav_b * scale_b

        mix = wav_a + wav_b
        peak = mix.abs().max().clamp_min(1e-6)
        if peak > cfg.MIX_PEAK_LIMIT:
            norm = cfg.MIX_PEAK_LIMIT / peak
            mix = mix * norm
            wav_a = wav_a * norm
            wav_b = wav_b * norm

        target = torch.stack([wav_a, wav_b], dim=0)   # (2, T)
        return {"mix_audio": mix, "target_audio": target}


def collate_fn(batch):
    return {
        "mix_audio": torch.stack([b["mix_audio"] for b in batch]),
        "target_audio": torch.stack([b["target_audio"] for b in batch]),
    }
