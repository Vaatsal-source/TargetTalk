# TargetTalk

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Research-Enabled-0A66C2?style=for-the-badge" alt="Research Enabled" />
</p>

<p align="center">
  <strong>A PyTorch research codebase for multi-speaker speech separation</strong><br>
  <em>Extracting a target speaker’s clean speech from mixed audio using visual cues from lip video.</em>
</p>

---

## Overview

TargetTalk is a research-oriented PyTorch project focused on solving the challenging problem of target speaker extraction in multi-speaker audio scenes. Instead of separating all speakers indiscriminately, the system focuses on one target speaker by leveraging visual information from their lip movements, enabling more robust and speaker-aware speech enhancement.

This repository is designed for experimentation and research workflows in:
- speech separation
- target speaker extraction
- audio-visual learning
- multimodal signal processing
- robust speech enhancement

The core idea is simple yet powerful: when multiple speakers are active simultaneously, the target speaker’s mouth motion provides a strong visual cue that helps isolate their voice from the mixture. By combining audio and visual streams, the model learns to emphasize the correct speaker while suppressing interference from others.

---

## Why TargetTalk?

Multi-speaker environments are inherently difficult because:
- overlapping voices create strong interference,
- standard speech separation may not preserve the identity of a desired speaker,
- target-specific extraction is often more useful in real applications than generic demixing.

TargetTalk addresses this by using:
- mixed audio signal as the primary input,
- lip video of the target speaker as a conditioning signal,
- a deep learning model trained to reconstruct the target clean speech waveform.

This makes the system especially relevant for:
- conferencing
- assistive listening
- human-computer interaction
- speech recognition preprocessing
- multimodal audio understanding

---

## Key Features

- Target-speaker speech extraction from mixed audio
- Lip-driven visual conditioning for speaker identity and timing
- PyTorch-based training pipeline
- Modular research codebase for experimentation
- Support for audio-visual multimodal models
- Designed for research and extension
- Easy to adapt for custom datasets and experiments

---

## Research Motivation

Speech separation has traditionally relied on audio-only approaches, which struggle when:
- speakers overlap heavily,
- source identities are ambiguous,
- interference is dynamic or non-stationary.

Visual information from lip motion helps resolve these ambiguities. The mouth region contains speech-related information strongly correlated with the actual audio signal. By conditioning the model on the target speaker’s visual features, the network learns a speaker-aware representation of the audio mixture and can more accurately reconstruct the clean target speech.

---

## System Concept

The project follows a common multimodal target extraction pipeline:

1. Input audio mixture contains multiple overlapping speakers.
2. The target speaker’s lip video is extracted and encoded.
3. The visual stream provides speaker identity and phonetic timing cues.
4. The audio encoder processes the mixed waveform.
5. A fusion module combines audio and visual representations.
6. The model predicts a clean target speech signal.

This framework is especially effective when the target speaker is known through the visual stream, allowing the model to focus on their voice rather than all possible sources.

---

## Repository Structure

```text
TargetTalk/
├── configs/                  # Experiment and training configuration files
├── data/                    # Dataset loaders, preprocessing utilities, and metadata
├── models/                  # Model architectures and multimodal modules
├── utils/                   # Helper functions, metrics, logging, and training utilities
├── scripts/                 # Training, evaluation, and inference entry points
├── notebooks/               # Exploratory experiments and analysis
├── checkpoints/             # Saved checkpoints and model weights
├── logs/                    # Training logs and TensorBoard outputs
├── README.md                # Project documentation
├── requirements.txt          # Python dependencies
├── setup.py                 # Optional packaging configuration
├── train.py                 # Main training script
├── evaluate.py              # Evaluation script
├── infer.py                 # Inference script
├── LICENSE                  # Project license
└── .gitignore               # Git ignore rules
```

> Note: The exact directory layout may vary depending on the development stage of the project. The above structure is representative of a typical research codebase for this task.

---

## Installation

### Prerequisites

- Python 3.8+
- PyTorch
- CUDA-capable GPU recommended for training
- FFmpeg (for audio processing pipelines)
- Optional: TensorBoard for monitoring

### Clone the repository

```bash
git clone https://github.com/Vaatsal-source/TargetTalk.git
cd TargetTalk
```

### Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
# or
venv\Scripts\activate      # Windows
```

### Install dependencies

```bash
pip install -r requirements.txt
```

If your environment is missing additional packages, install them as needed:

```bash
pip install torch torchvision torchaudio
pip install numpy scipy soundfile tqdm
pip install tensorboard
```

---

## Data Preparation

TargetTalk typically requires:
- mixed speech audio recordings,
- clean target speech reference signals,
- lip video corresponding to the target speaker.

### Typical dataset format

```text
data/
├── train/
│   ├── audio/
│   ├── video/
│   ├── target/
│   └── metadata.json
├── val/
│   ├── audio/
│   ├── video/
│   ├── target/
│   └── metadata.json
└── test/
    ├── audio/
    ├── video/
    ├── target/
    └── metadata.json
```

### Preprocessing steps

- resample audio to the target sampling rate,
- normalize waveform amplitude,
- extract video frames aligned with audio,
- crop the mouth region or lip-centered ROI,
- generate synchronized audio-video pairs,
- create training/validation/test splits.

---

## Training

Use the training script with a configuration file:

```bash
python train.py --config configs/default.yaml
```

Example config may include:
- learning rate
- batch size
- number of epochs
- audio sampling rate
- video frame rate
- optimizer settings
- augmentation settings
- checkpoint directory

### Typical training command

```bash
python train.py \
  --config configs/targettalk.yaml \
  --device cuda \
  --logdir logs/ \
  --checkpoint_dir checkpoints/
```

### Monitoring

You can monitor progress with TensorBoard:

```bash
tensorboard --logdir logs/
```

---

## Evaluation

To evaluate a trained model:

```bash
python evaluate.py --config configs/default.yaml --weights checkpoints/best_model.pth
```

Evaluation metrics often include:
- SI-SDR
- SDR
- PESQ
- STOI
- speech intelligibility metrics
- perceptual quality metrics

These metrics help assess how well the model reconstructs the target speaker’s clean speech from the mixed audio.

---

## Inference

For single-sample or batch inference:

```bash
python infer.py --config configs/default.yaml --checkpoint checkpoints/best_model.pth --input path/to/audio_mix.wav --video path/to/target_lip_video.mp4
```

The output is typically:
- enhanced target speech waveform,
- optional visualization of spectrograms,
- optional audio export in WAV format.

---

## Model Philosophy

The underlying objective is to learn a mapping from:
- mixed waveform + target visual cues
to:
- clean speech of the target speaker.

This is a form of speaker-conditioned source extraction. Rather than blindly separating all sources, the model learns to focus on the desired voice using the visual identity and timing cues from the target speaker’s lips.

This is particularly well-suited for:
- dynamic speaker scenarios,
- overlapping speech,
- multimodal speech processing,
- realistic conversations and meetings.

---

## Possible Extensions

This codebase is naturally extensible for research experiments such as:
- alternative multimodal fusion strategies,
- transformer-based audio-visual encoders,
- adversarial learning objectives,
- self-supervised pretraining,
- robust performance under noisy conditions,
- real-time streaming target extraction,
- multilingual datasets,
- speaker diarization integration.

---

## Example Use Cases

- separating a target participant’s voice in a meeting
- cleaning speech in cocktail-party scenarios
- improving ASR performance in overlapping speech
- enhancing hearing aid processing
- speech retrieval for target-speaker applications

---

## Citation

If you use this project in your research, please consider citing it.

```bibtex
@misc{targettalk2026,
  title={TargetTalk: Target Speaker Extraction with Visual Lip Cues},
  author={Your Name and Contributors},
  year={2026},
  howpublished={GitHub repository},
  url={https://github.com/Vaatsal-source/TargetTalk}
}
```

> Replace the author field with your actual contributors if needed.

---

## License

This project is licensed under the MIT License unless otherwise specified. See the `LICENSE` file for details.

---

## Acknowledgements

This project builds on the foundation of research in:
- speech separation,
- target speaker extraction,
- multimodal learning,
- audio-visual speech processing,
- deep learning for signal enhancement.

We thank the broader research community for developing the ideas and methods that make work like this possible.

---

## Contact

For questions, collaboration, or research discussions, feel free to reach out through the repository’s GitHub issues or project discussions.

---

## Final Note

TargetTalk is designed as a research-first, modular audio-visual speech separation codebase. It is intended to help researchers and developers explore the potential of using visual lip information as a strong conditioning signal for extracting a target speaker’s speech from complex acoustic mixtures.

If you are building on this project, the most valuable contribution is often a clean reproduction of the methodology, thoughtful evaluation, and careful ablation studies that show how visual information improves target extraction performance.

---

<p align="center">
  <strong>Built for research. Designed for extraction. Focused on the target speaker.</strong>
</p>

