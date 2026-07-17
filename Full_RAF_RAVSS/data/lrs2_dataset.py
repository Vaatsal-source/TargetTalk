"""
data/lrs2_dataset.py

Loads your LRS2-v2 layout:

    mvlrs_v1/
        main/
            <speaker_id>/
                00001.mp4
                00001.txt
                ...
        pretrain/
            ...

Each .mp4 has synced audio+video of ONE speaker talking. There is no
pre-made mixture in LRS2, so we build a 2-speaker mixture ON THE FLY:
  - pick an "anchor" utterance -> its video = the visual conditioning cue,
    its audio = the target we want the model to recover.
  - pick a random "interferer" utterance -> only its audio is used, mixed
    on top of the anchor's audio at a random SNR.
The model must use the anchor's lip video to pull the anchor's voice out
of the mixture (audio-visual target speaker extraction).
"""

import os
import glob
import random
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

try:
    import cv2
except ImportError:
    cv2 = None

from config import cfg


def _list_utterances(root, split):
    """Returns list of (mp4_path,) for every utterance under root/split/*/*.mp4"""
    pattern = os.path.join(root, split, "*", "*.mp4")
    files = sorted(glob.glob(pattern))
    if len(files) == 0:
        raise RuntimeError(
            f"No .mp4 files found under {pattern}. Check DATA_ROOT/TRAIN_SPLIT in config.py "
            f"and that the dataset is mounted (e.g. /kaggle/input/... is read-only)."
        )
    return files


class LRS2MixtureDataset(Dataset):
    """
    Yields:
        mix_audio   : (T,)         float32, mixture waveform, target + interferer
        target_audio: (T,)         float32, clean target waveform (training label)
        lip_frames  : (F, 1, H, W) float32, grayscale mouth-ROI crops of the target speaker
        snr_db      : float, the SNR used to build the mixture (useful for logging)
    """

    def __init__(self, root=None, split=None, train=True, val_fraction=0.05):
        self.root = root or cfg.DATA_ROOT
        self.split = split or cfg.TRAIN_SPLIT
        all_files = _list_utterances(self.root, self.split)

        # deterministic train/val split by index, not by speaker, kept simple.
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
        self._face_cascade = None  # lazy-loaded

    def __len__(self):
        return len(self.files)

    # ---------------- audio ----------------
    def _load_audio(self, mp4_path):
        wav, sr = torchaudio.load(mp4_path)  # torchaudio can demux mp4 audio via ffmpeg backend
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != cfg.SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, cfg.SAMPLE_RATE)
        wav = wav.squeeze(0)

        target_len = cfg.CHUNK_SAMPLES
        if wav.shape[0] >= target_len:
            if self.train:
                start = random.randint(0, wav.shape[0] - target_len)
            else:
                start = 0
            wav = wav[start:start + target_len]
            frame_start = start / cfg.SAMPLE_RATE
        else:
            pad = target_len - wav.shape[0]
            wav = torch.nn.functional.pad(wav, (0, pad))
            frame_start = 0.0
        return wav, frame_start

    # ---------------- video ----------------
    def _get_face_cascade(self):
        if self._face_cascade is None and cv2 is not None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._face_cascade = cv2.CascadeClassifier(cascade_path)
        return self._face_cascade

    def _mouth_crop(self, frame_bgr):
        """Very lightweight mouth ROI extractor: face-detect, take the lower
        third of the face box, resize to LIP_CROP_SIZE. If detection fails,
        fall back to a center crop. Swap this for landmark-based cropping
        (e.g. mediapipe FaceMesh) if you need tighter ROIs."""
        h, w = frame_bgr.shape[:2]
        cascade = self._get_face_cascade()
        box = None
        if cascade is not None:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))
            if len(faces) > 0:
                box = max(faces, key=lambda f: f[2] * f[3])  # largest face
        if box is not None:
            x, y, fw, fh = box
            y0 = y + int(fh * 0.55)
            y1 = min(h, y + fh)
            x0, x1 = x, min(w, x + fw)
        else:
            y0, y1 = int(h * 0.55), h
            x0, x1 = int(w * 0.25), int(w * 0.75)
        crop = frame_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            crop = frame_bgr
        crop = cv2.resize(crop, (cfg.LIP_CROP_SIZE, cfg.LIP_CROP_SIZE))
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return gray.astype(np.float32) / 255.0

    def _load_video(self, mp4_path, start_time_s):
        if cv2 is None:
            raise RuntimeError("opencv-python is required to read video frames: pip install opencv-python")
        cap = cv2.VideoCapture(mp4_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or cfg.VIDEO_FPS
        start_frame = int(start_time_s * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame))

        frames = []
        n_needed = cfg.FRAMES_PER_CHUNK
        while len(frames) < n_needed:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(self._mouth_crop(frame))
        cap.release()

        if len(frames) == 0:
            frames = [np.zeros((cfg.LIP_CROP_SIZE, cfg.LIP_CROP_SIZE), dtype=np.float32)]
        while len(frames) < n_needed:
            frames.append(frames[-1])  # pad by repeating last frame

        arr = np.stack(frames[:n_needed], axis=0)          # (F, H, W)
        arr = arr[:, None, :, :]                             # (F, 1, H, W)
        return torch.from_numpy(arr)

    # ---------------- mixing ----------------
    def __getitem__(self, idx):
        anchor_path = self.files[idx]
        interferer_path = self.files[random.randint(0, len(self.files) - 1)]
        while interferer_path == anchor_path and len(self.files) > 1:
            interferer_path = self.files[random.randint(0, len(self.files) - 1)]

        target_audio, start_t = self._load_audio(anchor_path)
        interferer_audio, _ = self._load_audio(interferer_path)

        snr_db = random.uniform(-5, 5)
        target_power = target_audio.pow(2).mean().clamp_min(1e-8)
        interf_power = interferer_audio.pow(2).mean().clamp_min(1e-8)
        scale = torch.sqrt(target_power / (interf_power * (10 ** (snr_db / 10))))
        mix_audio = target_audio + scale * interferer_audio

        # normalize to avoid clipping differences across samples
        norm = mix_audio.abs().max().clamp_min(1e-6)
        mix_audio = mix_audio / norm
        target_audio = target_audio / norm

        lip_frames = self._load_video(anchor_path, start_t)

        return {
            "mix_audio": mix_audio,
            "target_audio": target_audio,
            "lip_frames": lip_frames,
            "snr_db": torch.tensor(snr_db, dtype=torch.float32),
        }


def collate_fn(batch):
    return {
        "mix_audio": torch.stack([b["mix_audio"] for b in batch]),
        "target_audio": torch.stack([b["target_audio"] for b in batch]),
        "lip_frames": torch.stack([b["lip_frames"] for b in batch]),
        "snr_db": torch.stack([b["snr_db"] for b in batch]),
    }
