# XL-ChannelDiff

### An Efficient Diffusion-Based Multi-Domain Near-Field Channel Extrapolation Framework for XL-MIMO Systems

**Mengyuan Li, Yu Han, Hao Xu, Yongxu Zhu, Chao-Kai Wen, and Shi Jin**  
*IEEE Transactions on Wireless Communications*, vol. 25, 2026

[Paper](https://ieeexplore.ieee.org/document/11631639) | [DOI](https://doi.org/10.1109/TWC.2026.3716401) | [Dataset](https://huggingface.co/datasets/lmyxxn/XL-Diff) | [Pretrained checkpoint](checkpoints/paper_original/model_best.pt) | [Citation](#citation)

XL-ChannelDiff reconstructs complete near-field channels from partial channel observations using a conditional diffusion model. The framework combines a physics-aware Transformer backbone, mask-guided attention, WGAN-based supervision and sampling guidance, and RePaint-style refinement.

**Release scope:** this repository provides the antenna-domain implementation, original pretrained checkpoint, and training/evaluation entry points. The paper also studies frequency- and spatial-domain extrapolation; those implementations are not included in this release.

## Framework

![Original paper architecture: WGAN-enhanced CDDIM channel extrapolation](assets/framework.png)

*Architecture reproduced from the original paper. The figure presents the multi-domain framework; this release uses its 2D channel representation.*

<p align="center">
  <img src="assets/channel_extrapolation.png" alt="Original paper illustration of antenna-domain channel extrapolation" width="520">
</p>

*Antenna-domain extrapolation illustrated in the original paper: recover the full channel from partially observed entries.*

## Installation

```bash
git clone https://github.com/Lmyxxn/XL-Diff.git
cd XL-Diff
pip install -r requirements.txt
```

Use a CUDA-enabled PyTorch environment. PyTorch, CUDA, and xFormers must be mutually compatible. The supplied evaluation script runs on CUDA; a CPU-only workflow is not provided. Run all commands from the repository root because dataset paths are relative.

## Dataset

Download the original training and test files from
[Hugging Face: lmyxxn/XL-Diff](https://huggingface.co/datasets/lmyxxn/XL-Diff/tree/main).
Create a `data/` directory in the project root and place both files there,
keeping their filenames unchanged:

| File | Use | Size |
| --- | --- | --- |
| [CDL-C_Nt1_Nr1024_QuaRIGA_UPA0.50_seed1234.mat](https://huggingface.co/datasets/lmyxxn/XL-Diff/resolve/main/CDL-C_Nt1_Nr1024_QuaRIGA_UPA0.50_seed1234.mat?download=true) | Training and normalization | Approximately 1.59 GB |
| [CDL-C_Nt1_Nr1024_QuaRIGA_UPA0.50_seed4321.mat](https://huggingface.co/datasets/lmyxxn/XL-Diff/resolve/main/CDL-C_Nt1_Nr1024_QuaRIGA_UPA0.50_seed4321.mat?download=true) | Evaluation; also used for validation by the supplied training script | Approximately 31.83 MB |

```text
XL-Diff/
  data/
    CDL-C_Nt1_Nr1024_QuaRIGA_UPA0.50_seed1234.mat
    CDL-C_Nt1_Nr1024_QuaRIGA_UPA0.50_seed4321.mat
  checkpoints/
    paper_original/
      model_best.pt
```

The loader reads the first subcarrier and represents each channel as a `32 x 32` complex-valued array. Evaluation uses normalization statistics computed from the training file, so **both files are required even when using the pretrained checkpoint**.

## Pretrained Evaluation

The original checkpoint is included at `checkpoints/paper_original/model_best.pt`.

```bash
python evaluate.py --train CDL-C --test CDL-C --ddim_steps 50 --output_root results/paper_50steps
```

Set `--ddim_steps` to `5`, `50`, or `100` to evaluate different sampling budgets. The script evaluates the configured masking ratios and writes metrics to the specified output directory.

To use a different checkpoint:

```bash
python evaluate.py --train CDL-C --test CDL-C --ddim_steps 50 --checkpoint /absolute/path/to/model_best.pt --output_root results/custom
```

## Repository Structure

- `train.py`: fixed-seed training and checkpoint-resume entry.
- `evaluate.py`: NMSE evaluation with configurable DDIM sampling steps.
- `loaders.py`: MATLAB loading, normalization, and observation masks.
- `cgan_enhanced_ddim.py`, `ddpm.py`: conditional diffusion sampling implementation.
- `DiT/`: diffusion Transformer and its required layers.
- `GAN/`: time-conditioned discriminator used for training and sampling guidance.
- `ema.py`: exponential moving average state for training and resuming.
- `data/*seed1234.mat`: expected training set and normalization source.
- `data/*seed4321.mat`: expected test set.
- `checkpoints/paper_original/model_best.pt`: included original pretrained checkpoint.
- `assets/`: figures from the original paper.

## Training

```bash
python train.py --gpu 0 --train CDL-C --seed 42 --output_root outputs/run
```

Training saves `model_best.pt` when the validation noise-reconstruction loss
improves and updates `model_latest.pt` for resuming. The bundled paper checkpoint
is named `model_best.pt`; its weights are unchanged. Checkpoints are written under a configuration-specific subdirectory of `--output_root`; use the saved file's actual path when resuming.

For an independent benchmark, use a separate held-out test split rather than the file used for validation during training.

To resume while retaining the model, optimizer, discriminator, and EMA states:

```bash
python train.py --gpu 0 --train CDL-C --seed 42 --output_root outputs/run --resume /absolute/path/to/model_latest.pt
```

## Citation

If you use this code or dataset, please cite:

> M. Li, Y. Han, H. Xu, Y. Zhu, C.-K. Wen, and S. Jin, "XL-ChannelDiff: An Efficient Diffusion-Based Multi-Domain Near-Field Channel Extrapolation Framework for XL-MIMO Systems," *IEEE Transactions on Wireless Communications*, vol. 25, 2026, doi: [10.1109/TWC.2026.3716401](https://doi.org/10.1109/TWC.2026.3716401).

```bibtex
@article{li2026xlchanneldiff,
  author = {Mengyuan Li and Yu Han and Hao Xu and Yongxu Zhu and Chao-Kai Wen and Shi Jin},
  title = {{XL-ChannelDiff}: An Efficient Diffusion-Based Multi-Domain Near-Field Channel Extrapolation Framework for {XL-MIMO} Systems},
  journal = {IEEE Transactions on Wireless Communications},
  volume = {25},
  year = {2026},
  doi = {10.1109/TWC.2026.3716401},
  url = {https://ieeexplore.ieee.org/document/11631639}
}
```

## Questions

Please open a [GitHub issue](https://github.com/Lmyxxn/XL-Diff/issues) for questions about this release. Include the command, relevant error message, and your Python/PyTorch/CUDA/xFormers versions when reporting a problem.
