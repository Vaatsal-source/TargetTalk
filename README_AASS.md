# Speaker Audio Separation

This repository implements a deep learning model for blind two-speaker audio-only separation. It utilizes Permutation Invariant Training (PIT) optimized with Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) loss to separate overlapping voices from a single audio mixture without any visual conditioning.

Designed with resource efficiency in mind, the default architecture fits within a strict 2.0M - 3.5M parameter budget, making it ideal for training on platforms like Kaggle and Google Colab with strict time constraints.

## 🧠 Model Architecture

The network is composed of three main stages:
* **Audio Encoder:** A Conv-TasNet-style 1D convolutional layer that transforms the time-domain mixture into a latent representation.
* **Dual-Path Transformer Separator:** The core of the model. It processes chunks of the encoded audio using an alternating intra-chunk and inter-chunk self-attention mechanism, to capture both short-range and long-range dependencies. It outputs two non-negative masks.
* **Audio Decoder:** A transposed 1D convolutional layer that reconstructs the separate time-domain waveforms for each speaker by applying the generated masks to the encoded mixture.

## ✨ Key Features

* **Strict Parameter Budget:** Optimized configuration (under 3.5M parameters) ensures fast epoch cycle times and efficient convergence.
* **Session Safety & Checkpointing:** Built-in limits (e.g., `MAX_TRAIN_SECONDS`) ensure the training loop gracefully saves a checkpoint and halts before cloud environments time out. Training automatically resumes from the latest checkpoint.
* **Epoch-as-Subset Strategy:** Instead of processing massive datasets per epoch, it samples a fresh random subset of mixtures per epoch for faster, dynamic iteration.
* **Robust Evaluation Metrics:** Evaluates model performance using SI-SDR, SDR, PESQ, and STOI.
* **Permutation Resolution:** Resolves the Permutation Invariant Training (PIT) channel swap before calculating validation metrics, ensuring scores reflect actual separation quality.

## 📂 Dataset Preparation

This code is configured to train on the LRS2 (Lip Reading Sentences 2) dataset.
* By default, the `config.py` expects the dataset at `/kaggle/input/datasets/emnabouhajeb/lrs2-v1/mvlrs_v1`.
* The dataset loader traverses both `main/` and `pretrain/` splits.
* You can supply optional `train.txt`, `val.txt`, and `test.txt` file lists; otherwise, the loader performs a deterministic random split.

## ⚙️ Installation & Requirements

Ensure you have Python and PyTorch installed. You will also need specific packages for audio processing and evaluation:

```bash
pip install torch torchaudio numpy
pip install pesq pystoi
```

*(Note: `pesq` and `pystoi` are required to compute evaluation metrics during validation and testing).*

## 🚀 Usage

### 1. Verify Model Budget
Before training, verify that your current configuration in `config.py` lands within the target 2.0M - 3.5M parameter budget.
```bash
python count_params.py
```

### 2. Training
Start the training process. The script will automatically pick up from the latest checkpoint if one exists in your `checkpoints/` directory.
```bash
python train.py
```
*Note: The training uses Mixed Precision (AMP) and Gradient Accumulation by default to maximize batch processing on limited VRAM.*

### 3. Evaluation
Run a standalone evaluation on a specific checkpoint to calculate SI-SDR, SDR, PESQ, and STOI.
```bash
python metrics_eval.py --ckpt checkpoints/step_5000.pt --split val
```

**Optional Arguments:**
* `--split`: Choose between `train`, `val`, or `test` (Default: `val`).
* `--max_samples`: Limit the number of samples evaluated (Default: 500).
* `--save_audio`: Number of mixture/target/estimated wav files to save to the disk for manual listening (Default: 3).

Evaluation outputs are saved to the `eval_results/` directory, including a per-sample CSV, a summary CSV, and the extracted audio waveforms.

## 🛠️ Configuration (`config.py`)

All major hyperparameters, paths, and model dimensions are centralized in `config.py`.

To increase the model's capacity (and accept slower convergence), you can increase `ENC_CHANNELS`, `SEP_D_MODEL`, and `SEP_FF_DIM` to match standard SepFormer-style dimensions
