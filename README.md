# XL-Diff Reproduction Package

## Paper

This repository contains code for the following paper:

**XL-ChannelDiff: An Efficient Diffusion-Based Multi-Domain Near-Field Channel Extrapolation Framework for XL-MIMO Systems**

Read or download the paper on [IEEE Xplore](https://ieeexplore.ieee.org/document/11631639).

## Citation

If you use this code in your research, please cite:

> M. Li, Y. Han, H. Xu, Y. Zhu, C.-K. Wen, and S. Jin, "XL-ChannelDiff: An Efficient Diffusion-Based Multi-Domain Near-Field Channel Extrapolation Framework for XL-MIMO Systems," *IEEE Transactions on Wireless Communications*, vol. 25, 2026, doi: [10.1109/TWC.2026.3716401](https://doi.org/10.1109/TWC.2026.3716401).

## Overview

This package contains the XL-Diff channel extrapolation implementation.
Dataset files and pretrained weights are required separately for training
and evaluation. Frequency-domain and time-domain experiments are excluded.

## Package contents

- `train.py`: fixed-seed training and checkpoint-resume entry.
- `evaluate.py`: NMSE evaluation with configurable DDIM sampling steps.
- `cgan_enhanced_ddim.py`, `ddpm.py`: conditional diffusion sampling implementation.
- `DiT/`: diffusion Transformer and its required layers.
- `GAN/`: time-conditioned discriminator used for training and sampling guidance.
- `ema.py`: exponential moving average state for training and resuming.
- `data/*seed1234.mat`: expected training set and normalization source.
- `data/*seed4321.mat`: expected test set.
- `checkpoints/paper_original/model_best.pt`: expected pretrained checkpoint path.

## Environment

Create a CUDA-enabled Python environment and install dependencies. The PyTorch,
CUDA, and xFormers builds must be mutually compatible.

```bash
pip install -r requirements.txt
```

Run every command from this package root because the data paths are relative.

## Train

```bash
python train.py --gpu 0 --train CDL-C --seed 42 \
  --epochs 2000 --output_root outputs/seed42
```

Training saves `model_best.pt` when the validation noise-reconstruction loss
improves and updates `model_latest.pt` for resuming. The bundled paper checkpoint
is named `model_best.pt`; its weights are unchanged.

The unused standalone GAN generator has been removed. This changes random-number
consumption during initialization, so a new training run is not expected to be
bitwise identical to historical runs with the same seed. The diffusion model,
discriminator architecture, active losses, and paper checkpoint are retained.

To resume while retaining the model, optimizer, discriminator, and EMA states:

```bash
python train.py --gpu 0 --train CDL-C --seed 42 \
  --epochs 2000 --output_root outputs/seed42 \
  --resume /absolute/path/to/model_latest.pt
```

## Evaluate

The default checkpoint is `checkpoints/paper_original/model_best.pt`.

```bash
python evaluate.py --train CDL-C --test CDL-C \
  --ddim_steps 5 --output_root results/paper_5steps

python evaluate.py --train CDL-C --test CDL-C \
  --ddim_steps 50 --output_root results/paper_50steps

python evaluate.py --train CDL-C --test CDL-C \
  --ddim_steps 100 --output_root results/paper_100steps
```

To evaluate a newly trained checkpoint, override the default path:

```bash
--checkpoint /absolute/path/to/model_best.pt
```
