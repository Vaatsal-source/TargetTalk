# TargetTalk

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Research-Enabled-0A66C2?style=for-the-badge" alt="Research Enabled" />
</p>

<p align="center">
  <strong>A PyTorch research codebase for multi-speaker speech separation</strong><br>
  <em>A compact dual-path transformer separator for two-speaker mixtures, with a Conv-TasNet baseline.</em>
</p>

---

## Overview

TargetTalk is a research project on separating overlapping speakers from a single-channel audio mixture. The current code is an **audio-only, blind two-speaker separator** trained on LRS2 with Permutation Invariant Training (PIT) and an SI-SDR loss. It is designed to run on Kaggle or Colab within a small parameter budget (2.0M to 3.5M parameters).


---

## Demo

[`DemoRAVSS_2Speakers.mp4`](DemoRAVSS_2Speakers.mp4) is a screen recording of the Kaggle notebook ([`AASS.ipynb`](AASS.ipynb)) playing back results from the step 11354 checkpoint. For each sample it plays five clips:

- `input_mix`, the overlapping mixture
- `target_speaker1` and `target_speaker2`, the clean references
- `estimated_speaker1` and `estimated_speaker2`, the model outputs

---

## Model

The network has three stages:

1. **Audio encoder.** A Conv-TasNet-style 1D convolution that maps the time-domain mixture to a latent representation.
2. **Dual-path transformer separator.** It alternates intra-chunk and inter-chunk self-attention to capture short-range and long-range structure, and outputs two non-negative masks.
3. **Audio decoder.** A transposed 1D convolution that applies the masks to the encoded mixture and reconstructs one waveform per speaker.

### Training features

- **Parameter budget.** The default configuration stays between 2.0M and 3.5M parameters. [`count_params.py`](count_params.py) checks this.
- **Mixed precision and gradient accumulation** to fit small GPUs.
- **Time-budgeted training.** `MAX_TRAIN_SECONDS` makes the loop save a checkpoint and stop before the cloud session limit. Running `train.py` again resumes from the latest checkpoint.
- **Epoch-as-subset sampling.** Each epoch draws a fresh random subset of mixtures.
- **Permutation resolution.** The PIT channel swap is resolved before validation metrics are computed, so the scores reflect real separation quality.

---

## Repository Structure

```text
TargetTalk/
├── AASS.ipynb               # Kaggle notebook: train, evaluate, and listen to samples
├── Baseline/
│   └── librimix-convtasnet.ipynb   # Conv-TasNet baseline on LibriMix (Libri2Mix)
├── config.py                # Hyperparameters, dataset paths, time budget
├── count_params.py          # Checks the model fits the parameter budget
├── train.py                 # Training loop (AMP, grad accumulation, checkpoint/resume)
├── metrics_eval.py          # Standalone evaluation: SI-SDR, SDR, PESQ, STOI
├── data/
│   └── lrs2_dataset.py      # LRS2PairMixtureDataset: builds two-speaker mixtures from LRS2 audio
├── models/
│   ├── audio_codec.py       # AudioEncoder / AudioDecoder
│   ├── separator.py         # DualPathSeparator
│   └── rafravss.py          # TwoSpeakerSeparator: wires the three stages together
├── utils/
│   └── metrics.py           # SI-SDR, SDR, PIT loss, permutation resolution, PESQ, STOI
├── requirements.txt
├── README_AASS.md           # Detailed notes on the audio-only separator
└── DemoRAVSS_2Speakers.mp4  # Demo recording
```

---

## Installation

```bash
git clone https://github.com/Vaatsal-source/TargetTalk.git
cd TargetTalk
python -m venv venv
source venv/bin/activate   # Linux / macOS
# or
venv\Scripts\activate      # Windows
pip install -r requirements.txt
pip install pesq pystoi    # needed for PESQ and STOI during evaluation
```

A CUDA-capable GPU is recommended for training.

---

## Data

The code trains on **LRS2** (Lip Reading Sentences 2). By default `config.py` expects the dataset at `/kaggle/input/datasets/emnabouhajeb/lrs2-v1/mvlrs_v1`, laid out as:

```text
mvlrs_v1/
├── main/<speaker_id>/00001.mp4 (+ .txt)
└── pretrain/<speaker_id>/00001.mp4 (+ .txt)
```

The loader reads only the audio track from each mp4 and mixes two utterances from different speakers on the fly. You can supply optional `train.txt`, `val.txt` and `test.txt` file lists, one relative mp4 path per line. Without them the loader makes a deterministic random split.

The baseline notebook in `Baseline/` uses **LibriMix** instead.

---

## Usage

### 1. Check the parameter budget

```bash
python count_params.py
```

### 2. Train

```bash
python train.py
```

Training resumes automatically from the latest checkpoint in `checkpoints/`.

### 3. Evaluate

```bash
python metrics_eval.py --ckpt checkpoints/step_5000.pt --split val --max_samples 50 --save_audio 5 --out_dir eval_results/step_5000
```

This reports SI-SDR, SDR, PESQ and STOI, writes a `summary.csv`, and saves example WAV files (`sampleN_input_mix.wav`, `sampleN_target_speaker1.wav`, `sampleN_estimated_speaker1.wav`, and so on) under `audio_samples/`.

---

## Baseline

[`Baseline/librimix-convtasnet.ipynb`](Baseline/librimix-convtasnet.ipynb) trains a standard Conv-TasNet with a PIT SI-SDR loss on LibriMix. It gives reference numbers to compare the dual-path transformer against.

---

## Roadmap

- Add an inference script for single files.
- Compare the dual-path transformer against the Conv-TasNet baseline under the same budget.

---

## Contact

For questions or collaboration, open an issue on the GitHub repository.
