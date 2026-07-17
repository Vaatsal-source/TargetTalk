# RAF-RAVSS — Audio-Visual Target Speaker Extraction on LRS2

A from-scratch implementation of an audio-visual speech separation pipeline:
a **Conv-TasNet-style learnable audio codec** + a **SepFormer-style dual-path
transformer separator**, conditioned via **FiLM** on a lightweight
**AV-HuBERT-style visual frontend** built from your target speaker's lip
video. Task framing: given a 2-speaker audio mixture and the lip video of
one of the speakers, recover that speaker's clean speech.

## Project layout
```
raf_ravss/
  config.py                  <- every hyperparameter, paths, memory/time budget
  data/lrs2_dataset.py        <- LRS2 loading, mouth-ROI extraction, on-the-fly mixing
  models/
    audio_codec.py            <- learnable encoder/decoder (waveform <-> latent)
    visual_encoder.py         <- 3D-CNN + ResNet2D + temporal conv frontend
    separator.py               <- dual-path transformer + FiLM conditioning
    rafravss.py                <- wires everything together
  utils/metrics.py             <- SI-SDR (loss+metric), SDR, PESQ, STOI
  train.py                     <- training loop: AMP, grad accum, ckpt/resume, time budget
  requirements.txt
```

## Point it at your dataset
Your layout is:
```
lrs2-v1/mvlrs_v1/main/<speaker_id>/00001.mp4 (+ .txt)
lrs2-v1/mvlrs_v1/pretrain/...
```
Set in `config.py`:
```python
DATA_ROOT = "/kaggle/input/lrs2-v2/lrs2-v1/mvlrs_v1"
TRAIN_SPLIT = "main"     # switch to "pretrain" for more data once "main" works
```

## Why this fits Colab/Kaggle GPU + time limits
- **Audio chunking**: every sample is a fixed 3s crop (`CHUNK_SECONDS`), not
  full-length utterances — bounds both memory and per-step compute regardless
  of how long a raw LRS2 clip is.
- **Dual-path local attention** (chunk=100 frames) instead of full
  self-attention over the whole sequence — attention cost is O(K²·S) instead
  of O(L²), which is what lets SepFormer-style models run on a single GPU.
- **Gradient checkpointing** (`USE_GRAD_CKPT=True`) recomputes activations in
  the separator blocks during backward instead of storing them — trades
  ~20-30% more compute for a large activation-memory cut.
- **Mixed precision (AMP)** roughly halves activation memory and speeds up
  matmuls on T4/P100/A100.
- **Gradient accumulation** (`GRAD_ACCUM_STEPS=4`) lets you use a small
  physical batch (`BATCH_SIZE=4`, so 16 total 3-second audio+video samples in
  memory at once) while still getting an effective batch of 16 — tune the
  physical batch size first if you hit OOM, accumulation compensates.
- **A lightweight from-scratch visual frontend** (~3-4M params) instead of a
  full pretrained AV-HuBERT (~300M+ params) — see the docstring in
  `models/visual_encoder.py` for how to swap in a real pretrained (and
  frozen) AV-HuBERT checkpoint once you have the disk/time budget for it.
- **Time-budgeted training loop**: `MAX_TRAIN_SECONDS` (default 8.5h) is
  checked every step; when hit, the model checkpoints and exits cleanly
  instead of getting killed mid-write by the Colab/Kaggle session limit.
  Re-running `train.py` auto-resumes from the latest checkpoint in
  `CKPT_DIR` (optimizer + scheduler + scaler state included).

## Metrics
Reported at every validation step (`utils/metrics.py`):
- **SI-SDR** (scale-invariant SDR) — also the training loss (`si_sdr_loss`, the
  negative mean SI-SDR).
- **SDR** — classic, non scale-invariant version, for comparing against
  papers that report plain SDR.
- **PESQ** (`pip install pesq`) — perceptual speech quality, ITU-T P.862.
- **STOI** (`pip install pystoi`) — short-time objective intelligibility.

PESQ/STOI are numpy/CPU-only and non-differentiable, so they're computed only
in `validate()`, never inside the training step.

## Running
```bash
pip install -r requirements.txt
python train.py
```
First run starts fresh; every subsequent run picks up from the last saved
checkpoint automatically.

## Known simplifications (documented, not hidden)
1. **Visual conditioning is temporally pooled** before FiLM (one embedding
   per whole 3s clip, not per-audio-frame) — this keeps the audio/video rate
   mismatch (16kHz vs 25fps) trivial to handle. If your ablations show you
   need frame-synchronous conditioning, upsample `vis_embed_seq` to the
   audio encoder's frame rate with `nn.functional.interpolate` and feed it
   into FiLM per-position instead of pooling.
2. **Mouth ROI extraction** uses OpenCV Haar-cascade face detection + a fixed
   lower-face crop, not real landmark-based lip localization. Swap in
   mediapipe FaceMesh landmarks in `data/lrs2_dataset.py::_mouth_crop` for
   tighter crops if you see the frontend struggling.
3. **Mixtures are synthesized on the fly** from single-speaker LRS2 clips
   (no pre-built 2-speaker mixture set exists in raw LRS2) — this matches
   how LRS2-based separation papers (e.g. the original SepFormer/AV-HuBERT
   separation setups) construct training data.
