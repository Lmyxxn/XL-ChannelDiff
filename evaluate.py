#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate channel reconstruction NMSE and optionally measure model complexity."""

import numpy as np
import torch, os, copy, argparse, time

from DiT.enhanced_models import DiT

from cgan_enhanced_ddim import DDIM as CGANDDIM

from loaders               import Channels

from torch.utils.data     import DataLoader
from GAN.models_enhanced_wcgan import Discriminator

# Args
parser = argparse.ArgumentParser()

parser.add_argument('--train', type=str, default='CDL-C')
parser.add_argument('--test', type=str, default='CDL-C')
parser.add_argument('--save_channels', type=int, default=0)
parser.add_argument('--ddim_steps', type=int, default=100)
parser.add_argument('--ddim_eta', type=float, default=0.0)
parser.add_argument('--checkpoint', type=str, default='checkpoints/paper_original/model_best.pt')
parser.add_argument('--guidance_scale', type=float, default=1.0)
parser.add_argument('--sample_mask_ratio_threshold', type=float, default=0.7)
parser.add_argument('--run_complexity_analysis', action='store_true', help='Run model complexity analysis (time, FLOPs, parameters)')
parser.add_argument('--output_root', type=str, required=True)
args = parser.parse_args()

# Disable TF32 due to potential precision issues
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32       = False
torch.backends.cudnn.benchmark        = True
# GPU
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

num_rx             = 1024

mask_ratios = [0.2,0.3,0.4,0.5,0.6,0.7,0.8]

# Target file
target_file = os.path.abspath(args.checkpoint)
contents    = torch.load(target_file, weights_only=False)
config      = contents['config']
config.log_path = args.output_root

# DDIM specific parameters
config.model.ddim_eta = args.ddim_eta  # DDIM sampling parameter
config.model.ddim_steps = args.ddim_steps  # Number of DDIM steps
config.device = 'cuda:0'

# Define a mapping from string to class
model_classes = {
    'DiT': DiT,

}

# Get the class from the string
model_class = model_classes.get(config.model.type)

# Instantiate the model
if model_class is not None:
    diffuser = model_class(
        input_size=config.data.image_size[0],
        patch_size=config.model.patch_size,
        hidden_size=config.model.hidden_size,
        depth=config.model.depth
    )
else:
    raise ValueError(f"Model type {config.model.type} is not recognized.")
diffuser = diffuser.cuda()
# Load weights
diffuser.load_state_dict(contents['diffuser_state']) 
diffuser.eval()

# Load the critic used for sampling guidance.
discriminator = Discriminator(channels=2).to(config.device)
discriminator.load_state_dict(contents['discriminator_state'])
discriminator.eval()

# Initialize DDIM with CGAN guidance
ddim = CGANDDIM(
    device=config.device,
    n_steps=config.model.num_timesteps,
    min_beta=config.model.min_beta,
    max_beta=config.model.max_beta,
    guidance_scale=args.guidance_scale
)

seed = 42
torch.manual_seed(seed)
# Seeds for train and validation datasets
train_seed, val_seed = 1234, 4321

# Get training dataset for normalization
config.data.channel = args.train
dataset = Channels(train_seed, config, norm=config.data.norm_channels, num_rx=num_rx, mask_ratios=[0.5])

# Number of validation channels
num_channels = 200

# Clear unused variables and cache
import gc

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()

def metric_to_float(x):
    """Convert scalar / numpy array metric to a Python float."""
    return float(np.mean(np.asarray(x)))

def nmse_to_db(x, eps=1e-12):
    """Convert linear NMSE to dB safely."""
    return 10.0 * np.log10(max(float(x), eps))

nmse_summary_records = []

latency_summary_records = []

for mask_idx, mask_ratio in enumerate(mask_ratios):
    print(f"\n{'='*50}")
    print(f"Testing mask_ratio: {mask_ratio}")
    print(f"{'='*50}")
    
    # Global results
    result_dir = os.path.join(args.output_root, f'{mask_ratio:.6f}')

    os.makedirs(config.log_path, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    # Get validation dataset
    val_config = copy.deepcopy(config)
    val_config.data.channel      = args.test
    val_config.data.spacing_list = [0.5]
    val_dataset = Channels(val_seed, val_config, norm=[dataset.mean, dataset.std], num_rx=num_rx, mask_ratios=[mask_ratio])
    val_loader  = DataLoader(val_dataset, batch_size=num_channels,
        shuffle=False, num_workers=0, drop_last=True)

    # Inference only: GradScaler is not needed.

    complex_data_dir = os.path.join(result_dir, 'complex_data')
    os.makedirs(complex_data_dir, exist_ok=True)

    all_current_complex = []
    all_oracle = []
    all_random_complex = []
    all_zero_complex = []
    all_partial_complex = []
    all_masks = []

    nmse_log_batches = []
    nmse_random_log_batches = []
    nmse_zero_log_batches = []
    nmse_partial_log_batches = []

    l1_log_batches = []
    l1_random_log_batches = []
    l1_zero_log_batches = []
    l1_partial_log_batches = []

    cosine_log_batches = []
    cosine_random_log_batches = []
    cosine_zero_log_batches = []
    cosine_partial_log_batches = []

    total_samples = 0
    total_time = 0.0  # milliseconds
    latency_batch_times = []

    if mask_idx == 0 and args.run_complexity_analysis:
        print(f"\n{'='*60}")
        print("MODEL COMPLEXITY ANALYSIS")
        print(f"{'='*60}")

        def count_parameters(model):
            total_params = 0
            trainable_params = 0
            for param in model.parameters():
                param_count = param.numel()
                total_params += param_count
                if param.requires_grad:
                    trainable_params += param_count
            return total_params, trainable_params

        total_params_diffuser, trainable_params_diffuser = count_parameters(diffuser)

        total_params_discriminator, trainable_params_discriminator = count_parameters(discriminator)

        total_params = total_params_diffuser + total_params_discriminator
        trainable_params = trainable_params_diffuser + trainable_params_discriminator
        
        print(f"Network Parameters:")
        print(f"  DiT Model Parameters: {total_params_diffuser:,}")
        print(f"  Discriminator Parameters: {total_params_discriminator:,}")
        print(f"  Total Parameters: {total_params:,}")
        print(f"  Trainable Parameters: {trainable_params:,}")
        print(f"  Model Size: {total_params * 4 / 1024 / 1024:.2f} MB")

        def calculate_flops_detailed(model, input_shape, mac_to_flop=2):
            """Estimate DiT diffuser FLOPs per sample over all DDIM steps."""
            batch_size, channels, height, width = input_shape

            def first_int(value):
                if isinstance(value, (list, tuple)):
                    return int(value[0])
                if hasattr(value, "item"):
                    return int(value.item())
                return int(value)

            # Read dimensions from the instantiated model, not only from config.
            patch_size = first_int(model.x_embedder.patch_size)
            hidden_size = int(model.pos_embed.shape[-1])
            depth = len(model.blocks)
            ddim_steps = int(config.model.ddim_steps)
            out_channels = int(model.out_channels)
            seq_len = int(model.x_embedder.num_patches)
            num_patches = seq_len

            if seq_len <= 0:
                seq_len = (height // patch_size) * (width // patch_size)
                num_patches = seq_len
            if patch_size <= 0 or hidden_size <= 0 or depth <= 0 or ddim_steps <= 0:
                raise ValueError(
                    f"Invalid FLOPs config: patch_size={patch_size}, "
                    f"hidden_size={hidden_size}, depth={depth}, ddim_steps={ddim_steps}"
                )

            d = hidden_size
            first_block = model.blocks[0]
            mlp_hidden = int(first_block.mlp.fc1.out_features)
            mlp_ratio = float(mlp_hidden / d)

            # Patch embedding is called twice in DiT.forward(): x and h_partial.
            patch_embed_macs = (
                2 * batch_size * num_patches
                * (patch_size * patch_size * channels) * d
            )

            # Self-attention: qkv projection + output projection + QK^T + AV.
            self_attn_macs = batch_size * (
                4 * seq_len * d * d + 2 * seq_len * seq_len * d
            )

            # Cross-attention: q projection + kv projection + output projection + QK^T + AV.
            cross_attn_macs = batch_size * (
                4 * seq_len * d * d + 2 * seq_len * seq_len * d
            )

            # MLP: D -> mlp_hidden -> D.
            mlp_macs = batch_size * (
                seq_len * d * mlp_hidden + seq_len * mlp_hidden * d
            )

            # LayerNorm / modulation / elementwise gates, approximate but nonzero.
            norm_mod_macs = batch_size * seq_len * d * 8
            block_macs = self_attn_macs + cross_attn_macs + mlp_macs + norm_mod_macs
            all_blocks_macs = depth * block_macs

            # t_embedder D->D->D, t_block D->6D, t_block_final D->2D,
            # and final adaLN modulation D->2D.
            time_macs = batch_size * (2 * d * d + 6 * d * d + 2 * d * d + 2 * d * d)

            # Final token projection: D -> patch_size^2 * out_channels.
            final_proj_macs = (
                batch_size * num_patches * d
                * (patch_size * patch_size * out_channels)
            )

            single_forward_macs = (
                patch_embed_macs + time_macs + all_blocks_macs + final_proj_macs
            )
            single_forward_flops = single_forward_macs * mac_to_flop
            total_flops = single_forward_flops * ddim_steps

            breakdown = {
                'patch_size': int(patch_size),
                'hidden_size': int(hidden_size),
                'depth': int(depth),
                'mlp_ratio': float(mlp_ratio),
                'out_channels': int(out_channels),
                'num_patches': int(num_patches),
                'mac_to_flop': int(mac_to_flop),
                'patch_embed_flops_per_forward': int(patch_embed_macs * mac_to_flop),
                'time_embed_flops_per_forward': int(time_macs * mac_to_flop),
                'self_attn_flops_per_block': int(self_attn_macs * mac_to_flop),
                'cross_attn_flops_per_block': int(cross_attn_macs * mac_to_flop),
                'mlp_flops_per_block': int(mlp_macs * mac_to_flop),
                'norm_mod_flops_per_block': int(norm_mod_macs * mac_to_flop),
                'blocks_flops_per_forward': int(all_blocks_macs * mac_to_flop),
                'final_proj_flops_per_forward': int(final_proj_macs * mac_to_flop),
                'single_forward_flops': int(single_forward_flops),
                'total_sampling_flops': int(total_flops),
                'patch_embed_gflops_per_forward': float(patch_embed_macs * mac_to_flop / 1e9),
                'time_embed_gflops_per_forward': float(time_macs * mac_to_flop / 1e9),
                'self_attn_gflops_per_block': float(self_attn_macs * mac_to_flop / 1e9),
                'cross_attn_gflops_per_block': float(cross_attn_macs * mac_to_flop / 1e9),
                'mlp_gflops_per_block': float(mlp_macs * mac_to_flop / 1e9),
                'norm_mod_gflops_per_block': float(norm_mod_macs * mac_to_flop / 1e9),
                'blocks_gflops_per_forward': float(all_blocks_macs * mac_to_flop / 1e9),
                'final_proj_gflops_per_forward': float(final_proj_macs * mac_to_flop / 1e9),
                'single_forward_gflops': float(single_forward_flops / 1e9),
                'total_sampling_gflops': float(total_flops / 1e9),
                'ddim_steps': int(ddim_steps),
                'note': (
                    'Estimated DiT diffuser FLOPs only; discriminator-guidance '
                    'autograd overhead is excluded and reflected by measured latency.'
                ),
            }

            return int(total_flops), breakdown

        input_shape = (1, 2, 32, 32)
        detailed_flops, flops_breakdown = calculate_flops_detailed(diffuser, input_shape)
        print(f"\nComputational Complexity (FLOPs):")
        print(
            "  FLOPs config: "
            f"patch={flops_breakdown.get('patch_size', 'NA')}, "
            f"hidden={flops_breakdown.get('hidden_size', 'NA')}, "
            f"depth={flops_breakdown.get('depth', 'NA')}, "
            f"tokens={flops_breakdown.get('num_patches', 'NA')}, "
            f"mlp_ratio={flops_breakdown.get('mlp_ratio', 'NA')}"
        )
        print(f"  Estimated DiT FLOPs per forward: {flops_breakdown['single_forward_flops']:,}")
        print(f"  Estimated DiT GFLOPs per forward: {flops_breakdown['single_forward_gflops']:.6f}")
        print(f"  Estimated DiT FLOPs per sample over {config.model.ddim_steps} DDIM steps: {detailed_flops:,}")
        print(f"  Estimated DiT GFLOPs per sample over {config.model.ddim_steps} DDIM steps: {detailed_flops / 1e9:.6f}")
        print("  FLOPs breakdown per forward:")
        print(f"    patch embedding: {flops_breakdown['patch_embed_gflops_per_forward']:.6f} GFLOPs")
        print(f"    timestep/final modulation: {flops_breakdown['time_embed_gflops_per_forward']:.6f} GFLOPs")
        print(f"    transformer blocks total: {flops_breakdown['blocks_gflops_per_forward']:.6f} GFLOPs")
        print(f"    final projection: {flops_breakdown['final_proj_gflops_per_forward']:.6f} GFLOPs")
        print(f"  FLOPs note: {flops_breakdown.get('note', '')}")

        print(f"\nMeasuring latency during testing...")
        total_samples = 0
        total_time = 0.0  # milliseconds
        latency_batch_times = []

    st_time = time.time()
    for batch_idx, val_sample in enumerate(val_loader):
        H_full = val_sample['H_full'].cuda()
        H_partial = val_sample['H_partial'].cuda()
        mask1 = val_sample['mask'].cuda()
        mask = mask1.unsqueeze(1).repeat(1, 2, 1, 1).cuda()
        oracle = H_full[:, 0, :, :] + 1j * H_full[:, 1, :, :]

        mean_tensor = torch.tensor(dataset.mean, device=H_full.device, dtype=H_full.dtype)
        std_tensor = torch.tensor(dataset.std, device=H_full.device, dtype=H_full.dtype)
        
        if torch.is_complex(mean_tensor) or torch.is_complex(std_tensor):

            mean_real = torch.real(mean_tensor)
            std_real = torch.real(std_tensor)
        else:
            mean_real = mean_tensor
            std_real = std_tensor
        
        init_val_H = mean_real + std_real * torch.randn_like(H_full)

        # ---------------------------------------------------------
        # Model-only latency measurement
        # ---------------------------------------------------------
        # Scope: prepared tensors already on GPU -> sample_backward() starts -> current is returned.
        # Excluded: data loading, CPU-to-GPU transfer, input initialization, random/zero baselines,
        #           metric computation, CPU copies, and file saving.
        measure_latency = bool(args.run_complexity_analysis)
        if measure_latency:
            torch.cuda.synchronize()
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            start_time.record()

        with torch.cuda.amp.autocast():
            current = ddim.sample_backward(
                init_val_H, H_partial, diffuser, discriminator, mask1, mask,
                mask_ratio, args.sample_mask_ratio_threshold,
                device=config.device, simple_var=True,
                ddim_step=config.model.ddim_steps, eta=config.model.ddim_eta
            )

        if measure_latency:
            end_time.record()
            torch.cuda.synchronize()
            batch_time = start_time.elapsed_time(end_time)  # milliseconds for the whole batch
            batch_samples = H_full.shape[0]
            total_time += batch_time
            total_samples += batch_samples
            latency_batch_times.append(batch_time / batch_samples)
            print(
                f"Batch {batch_idx+1}: {batch_samples} samples, "
                f"model-only latency {batch_time:.3f} ms "
                f"({batch_time / batch_samples:.3f} ms/sample)"
            )

        boolean_mask = mask.bool()
        H_random_real = mean_real + std_real * torch.randn_like(H_full)
        random_generate = torch.where(boolean_mask, H_partial, H_random_real)
        
        current_complex = current[:, 0, :, :] + 1j * current[:, 1, :, :]
        random_complex = random_generate[:, 0, :, :] + 1j * random_generate[:, 1, :, :]
        H_partial_complex = H_partial[:, 0, :, :] + 1j * H_partial[:, 1, :, :]

        H_zero_complex = torch.zeros_like(oracle, dtype=torch.complex64)

        H_zero_complex = torch.where(mask1.bool(), oracle, H_zero_complex)

        all_current_complex.append(current_complex.detach().cpu())
        all_oracle.append(oracle.detach().cpu())
        all_random_complex.append(random_complex.detach().cpu())
        all_zero_complex.append(H_zero_complex.detach().cpu())
        all_partial_complex.append(H_partial_complex.detach().cpu())
        all_masks.append(mask1.detach().cpu())

        nmse = (torch.sum(torch.square(torch.abs(current_complex - oracle)), dim=(-1, -2)) /
                torch.sum(torch.square(torch.abs(oracle)), dim=(-1, -2))).detach().cpu().numpy()
        nmse_log_batches.append(nmse)

        nmse_random = (torch.sum(torch.square(torch.abs(random_complex - oracle)), dim=(-1, -2)) /
                torch.sum(torch.square(torch.abs(oracle)), dim=(-1, -2))).detach().cpu().numpy()
        nmse_random_log_batches.append(nmse_random)

        nmse_zero = (torch.sum(torch.square(torch.abs(H_zero_complex - oracle)), dim=(-1, -2)) /
                torch.sum(torch.square(torch.abs(oracle)), dim=(-1, -2))).detach().cpu().numpy()
        nmse_zero_log_batches.append(nmse_zero)

        nmse_partial = (torch.sum(torch.square(torch.abs(H_partial_complex - oracle)), dim=(-1, -2)) /
                torch.sum(torch.square(torch.abs(oracle)), dim=(-1, -2))).detach().cpu().numpy()
        nmse_partial_log_batches.append(nmse_partial)

        def calculate_l1_distance(pred, target):
            return torch.sum(torch.abs(pred - target), dim=(-1, -2)) / torch.sum(torch.abs(target), dim=(-1, -2))

        l1 = calculate_l1_distance(current_complex, oracle).detach().cpu().numpy()
        l1_random = calculate_l1_distance(random_complex, oracle).detach().cpu().numpy()
        l1_zero = calculate_l1_distance(H_zero_complex, oracle).detach().cpu().numpy()
        l1_partial = calculate_l1_distance(H_partial_complex, oracle).detach().cpu().numpy()

        l1_log_batches.append(l1)
        l1_random_log_batches.append(l1_random)
        l1_zero_log_batches.append(l1_zero)
        l1_partial_log_batches.append(l1_partial)

        def calculate_cosine_distance(pred, target):

            pred_flat = pred.reshape(pred.shape[0], -1)
            target_flat = target.reshape(target.shape[0], -1)

            inner_product = torch.sum(pred_flat * torch.conj(target_flat), dim=1)
            pred_norm = torch.sqrt(torch.sum(torch.abs(pred_flat)**2, dim=1))
            target_norm = torch.sqrt(torch.sum(torch.abs(target_flat)**2, dim=1))

            cosine_sim = torch.abs(inner_product) / (pred_norm * target_norm)

            cosine_dist = 1 - cosine_sim
            return cosine_dist

        cosine = calculate_cosine_distance(current_complex, oracle).detach().cpu().numpy()
        cosine_random = calculate_cosine_distance(random_complex, oracle).detach().cpu().numpy()
        cosine_zero = calculate_cosine_distance(H_zero_complex, oracle).detach().cpu().numpy()
        cosine_partial = calculate_cosine_distance(H_partial_complex, oracle).detach().cpu().numpy()

        cosine_log_batches.append(cosine)
        cosine_random_log_batches.append(cosine_random)
        cosine_zero_log_batches.append(cosine_zero)
        cosine_partial_log_batches.append(cosine_partial)

        del current, random_generate, boolean_mask, H_full, H_partial, mask, oracle, init_val_H, H_random_real, H_zero_complex
        clear_memory()

    ed_time = time.time()
    print(f"Time taken for mask ratio {mask_ratio}: {ed_time - st_time:.2f} seconds")

    if args.run_complexity_analysis:
        if total_samples > 0:
            avg_latency = total_time / total_samples
            latency_batch_times_np = np.array(latency_batch_times, dtype=np.float64)
            latency_std = float(np.std(latency_batch_times_np)) if latency_batch_times_np.size > 0 else 0.0
            latency_min = float(np.min(latency_batch_times_np)) if latency_batch_times_np.size > 0 else avg_latency
            latency_max = float(np.max(latency_batch_times_np)) if latency_batch_times_np.size > 0 else avg_latency

            print(f"\nLatency Statistics (model-only):")
            print(f"  Scope: prepared GPU tensors -> sample_backward() -> current output")
            print(f"  Excluded: data loading/transfer, initialization, baselines, metrics, CPU copies, saving")
            print(f"  Total samples: {total_samples}")
            print(f"  Total model time: {total_time:.3f} ms")
            print(f"  Average latency per sample: {avg_latency:.3f} ms")
            print(f"  Batch-level ms/sample: min {latency_min:.3f}, max {latency_max:.3f}, std {latency_std:.3f}")

            latency_summary_records.append({
                'mask_ratio': float(mask_ratio),
                'avg_latency': float(avg_latency),
                'latency_std': float(latency_std),
                'latency_min': float(latency_min),
                'latency_max': float(latency_max),
                'total_samples': int(total_samples),
                'total_model_time_ms': float(total_time),
            })

            if mask_idx == 0:
                complexity_results = {
                    'avg_latency': avg_latency,
                    'latency_std': latency_std,
                    'latency_min': latency_min,
                    'latency_max': latency_max,
                    'latency_scope': 'prepared GPU tensors -> sample_backward() -> current output; excludes data loading/transfer, initialization, baselines, metrics, CPU copies, and saving',
                    'GFLOPs': detailed_flops / 1e9,
                    'GFLOPs_scope': 'estimated DiT diffuser FLOPs over the complete DDIM sampling trajectory',
                    'flops_breakdown': flops_breakdown,
                    'Params': total_params,
                    'Params_diffuser': total_params_diffuser,
                    'Params_discriminator': total_params_discriminator
                }

                complexity_save_path = './models_complexity_analysis/WCGAN_Enhanced_DDIM.pt'
                os.makedirs(os.path.dirname(complexity_save_path), exist_ok=True)
                torch.save(complexity_results, complexity_save_path)
                print(f"\nComplexity analysis results saved to: {complexity_save_path}")

                print(f"\n{'='*60}")
                print("MODEL COMPLEXITY ANALYSIS STATIC INFO SAVED!")
                print(f"{'='*60}")

    all_current_complex = torch.cat(all_current_complex, dim=0)
    all_oracle = torch.cat(all_oracle, dim=0)
    all_random_complex = torch.cat(all_random_complex, dim=0)
    all_zero_complex = torch.cat(all_zero_complex, dim=0)
    all_partial_complex = torch.cat(all_partial_complex, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    if args.save_channels:
        save_path = os.path.join(complex_data_dir, 'complex_data.pt')
        torch.save({
            'current_complex': all_current_complex,
            'oracle': all_oracle,
            'random_complex': all_random_complex,
            'zero_complex': all_zero_complex,
            'partial_complex': all_partial_complex,
            'mask': all_masks,
            'checkpoint': args.checkpoint,
            'mask_ratio': mask_ratio
        }, save_path)
        print(f"Saved all complex data to {save_path}")

    nmse_log = np.concatenate(nmse_log_batches, axis=0)
    avg_nmse  = np.mean(nmse_log, axis=0)
    best_nmse = np.min(avg_nmse, axis=0)

    nmse_random_log = np.concatenate(nmse_random_log_batches, axis=0)
    avg_nmse_random = np.mean(nmse_random_log, axis=0)
    best_nmse_random = np.min(avg_nmse_random, axis=0)
                   
    nmse_zero_log = np.concatenate(nmse_zero_log_batches, axis=0)
    avg_nmse_zero = np.mean(nmse_zero_log, axis=0)
    best_nmse_zero = np.min(avg_nmse_zero, axis=0)

    nmse_partial_log = np.concatenate(nmse_partial_log_batches, axis=0)
    avg_nmse_partial = np.mean(nmse_partial_log, axis=0)
    best_nmse_partial = np.min(avg_nmse_partial, axis=0)

    print(f"Mask Ratio {mask_ratio} - NMSE Results:")
    print(f"Average NMSE (DDIM): {avg_nmse:.4f}")
    print(f"Average NMSE (Random): {avg_nmse_random:.4f}")
    print(f"Average NMSE (Zero): {avg_nmse_zero:.4f}")
    print(f"Average NMSE (Partial): {avg_nmse_partial:.4f}")

    avg_nmse_scalar = metric_to_float(avg_nmse)
    avg_nmse_random_scalar = metric_to_float(avg_nmse_random)
    avg_nmse_zero_scalar = metric_to_float(avg_nmse_zero)
    avg_nmse_partial_scalar = metric_to_float(avg_nmse_partial)
    nmse_summary_records.append({
        'mask_ratio': float(mask_ratio),
        'avg_nmse': avg_nmse_scalar,
        'avg_nmse_db': nmse_to_db(avg_nmse_scalar),
        'avg_nmse_random': avg_nmse_random_scalar,
        'avg_nmse_random_db': nmse_to_db(avg_nmse_random_scalar),
        'avg_nmse_zero': avg_nmse_zero_scalar,
        'avg_nmse_zero_db': nmse_to_db(avg_nmse_zero_scalar),
        'avg_nmse_partial': avg_nmse_partial_scalar,
        'avg_nmse_partial_db': nmse_to_db(avg_nmse_partial_scalar),
    })

    l1_log = np.concatenate(l1_log_batches, axis=0)
    l1_random_log = np.concatenate(l1_random_log_batches, axis=0)
    l1_zero_log = np.concatenate(l1_zero_log_batches, axis=0)
    l1_partial_log = np.concatenate(l1_partial_log_batches, axis=0)

    avg_l1 = np.mean(l1_log, axis=0)
    avg_l1_random = np.mean(l1_random_log, axis=0)
    avg_l1_zero = np.mean(l1_zero_log, axis=0)
    avg_l1_partial = np.mean(l1_partial_log, axis=0)

    print(f"Mask Ratio {mask_ratio} - L1 Distance Results:")
    print(f"Average L1 Distance (DDIM): {avg_l1:.4f}")
    print(f"Average L1 Distance (Random): {avg_l1_random:.4f}")
    print(f"Average L1 Distance (Zero): {avg_l1_zero:.4f}")
    print(f"Average L1 Distance (Partial): {avg_l1_partial:.4f}")

    cosine_log = np.concatenate(cosine_log_batches, axis=0)
    cosine_random_log = np.concatenate(cosine_random_log_batches, axis=0)
    cosine_zero_log = np.concatenate(cosine_zero_log_batches, axis=0)
    cosine_partial_log = np.concatenate(cosine_partial_log_batches, axis=0)

    avg_cosine = np.mean(cosine_log, axis=0)
    avg_cosine_random = np.mean(cosine_random_log, axis=0)
    avg_cosine_zero = np.mean(cosine_zero_log, axis=0)
    avg_cosine_partial = np.mean(cosine_partial_log, axis=0)

    print(f"Mask Ratio {mask_ratio} - Cosine Distance Results:")
    print(f"Average Cosine Distance (DDIM): {avg_cosine:.4f}")
    print(f"Average Cosine Distance (Random): {avg_cosine_random:.4f}")
    print(f"Average Cosine Distance (Zero): {avg_cosine_zero:.4f}")
    print(f"Average Cosine Distance (Partial): {avg_cosine_partial:.4f}")

    # Save results to file for this mask ratio
    save_dict = {
        'nmse_log': nmse_log,
        'avg_nmse': avg_nmse,
        'best_nmse': best_nmse,
        'nmse_random_log': nmse_random_log,
        'avg_nmse_random': avg_nmse_random,
        'best_nmse_random': best_nmse_random,
        'nmse_zero_log': nmse_zero_log,
        'avg_nmse_zero': avg_nmse_zero,
        'best_nmse_zero': best_nmse_zero,
        'nmse_partial_log': nmse_partial_log,
        'avg_nmse_partial': avg_nmse_partial,
        'best_nmse_partial': best_nmse_partial,
        'val_config': val_config,
        'ddim_steps': config.model.ddim_steps,
        'ddim_eta': config.model.ddim_eta,
        'mask_ratio': mask_ratio,
        'l1_log': l1_log,
        'avg_l1': avg_l1,
        'l1_random_log': l1_random_log,
        'avg_l1_random': avg_l1_random,
        'l1_zero_log': l1_zero_log,
        'avg_l1_zero': avg_l1_zero,
        'l1_partial_log': l1_partial_log,
        'avg_l1_partial': avg_l1_partial,
        'cosine_log': cosine_log,
        'avg_cosine': avg_cosine,
        'cosine_random_log': cosine_random_log,
        'avg_cosine_random': avg_cosine_random,
        'cosine_zero_log': cosine_zero_log,
        'avg_cosine_zero': avg_cosine_zero,
        'cosine_partial_log': cosine_partial_log,
        'avg_cosine_partial': avg_cosine_partial
    }
    torch.save(save_dict, os.path.join(result_dir, 'results.pt'))
    print(f"Saved results for mask_ratio {mask_ratio} to {os.path.join(result_dir, 'results.pt')}")

if len(latency_summary_records) > 0:
    print(f"\n{'='*96}")
    print(f"MODEL-ONLY LATENCY SUMMARY OVER MASK RATIOS (DDIM steps = {config.model.ddim_steps})")
    print(f"{'='*96}")
    print(
        f"{'Mask':>6} | {'Avg Lat.':>10} | {'Min':>10} | {'Max':>10} | {'Std':>10} | {'Samples':>8} | {'Total ms':>12}"
    )
    print('-' * 96)
    for rec in latency_summary_records:
        print(
            f"{rec['mask_ratio']:6.1f} | "
            f"{rec['avg_latency']:10.3f} | {rec['latency_min']:10.3f} | "
            f"{rec['latency_max']:10.3f} | {rec['latency_std']:10.3f} | "
            f"{rec['total_samples']:8d} | {rec['total_model_time_ms']:12.3f}"
        )

    latency_vals = np.array([rec['avg_latency'] for rec in latency_summary_records], dtype=np.float64)
    total_model_time_all = np.sum([rec['total_model_time_ms'] for rec in latency_summary_records], dtype=np.float64)
    total_samples_all = np.sum([rec['total_samples'] for rec in latency_summary_records], dtype=np.float64)
    mean_latency_unweighted = float(np.mean(latency_vals))
    mean_latency_weighted = float(total_model_time_all / max(total_samples_all, 1.0))
    print('-' * 96)
    print(f"Mean of per-mask latency: {mean_latency_unweighted:.3f} ms/sample")
    print(f"Weighted overall latency: {mean_latency_weighted:.3f} ms/sample")
    print(f"Latency scope: prepared GPU tensors -> sample_backward() -> current output")

    latency_summary_save_path = (
        f"./results/latency_summaries/"
        f"WCGAN_Enhanced_DDIM_{config.model.ddim_steps}steps_"
        f"train-{args.train}_test-{args.test}.pt"
    )
    os.makedirs(os.path.dirname(latency_summary_save_path), exist_ok=True)
    torch.save({
        'ddim_steps': config.model.ddim_steps,
        'checkpoint': args.checkpoint,
        'train': args.train,
        'test': args.test,
        'records': latency_summary_records,
        'mean_latency_unweighted': mean_latency_unweighted,
        'mean_latency_weighted': mean_latency_weighted,
        'latency_scope': 'prepared GPU tensors -> sample_backward() -> current output; excludes data loading/transfer, initialization, baselines, metrics, CPU copies, and saving',
    }, latency_summary_save_path)
    print(f"Saved latency summary to: {latency_summary_save_path}")

    complexity_save_path = './models_complexity_analysis/WCGAN_Enhanced_DDIM.pt'
    if os.path.exists(complexity_save_path):
        complexity_results = torch.load(complexity_save_path, map_location='cpu')
        complexity_results.update({
            'avg_latency': mean_latency_weighted,
            'mean_latency_unweighted': mean_latency_unweighted,
            'mean_latency_weighted': mean_latency_weighted,
            'latency_by_mask': latency_summary_records,
        })
        torch.save(complexity_results, complexity_save_path)
        print(f"Updated complexity analysis with all-mask average latency: {complexity_save_path}")

if len(nmse_summary_records) > 0:
    print(f"\n{'='*86}")
    print(f"NMSE SUMMARY OVER MASK RATIOS (DDIM steps = {config.model.ddim_steps})")
    print(f"{'='*86}")
    print(
        f"{'Mask':>6} | {'DDIM':>12} | {'DDIM(dB)':>10} | "
        f"{'Random':>12} | {'Random(dB)':>10} | {'Zero':>12} | {'Partial':>12}"
    )
    print('-' * 86)
    for rec in nmse_summary_records:
        print(
            f"{rec['mask_ratio']:6.1f} | "
            f"{rec['avg_nmse']:12.6e} | {rec['avg_nmse_db']:10.3f} | "
            f"{rec['avg_nmse_random']:12.6e} | {rec['avg_nmse_random_db']:10.3f} | "
            f"{rec['avg_nmse_zero']:12.6e} | {rec['avg_nmse_partial']:12.6e}"
        )

    ddim_nmse_linear = np.array([rec['avg_nmse'] for rec in nmse_summary_records], dtype=np.float64)
    ddim_nmse_db = np.array([rec['avg_nmse_db'] for rec in nmse_summary_records], dtype=np.float64)
    print('-' * 86)
    print(f"Mean over mask ratios, linear NMSE: {np.mean(ddim_nmse_linear):.6e}")
    print(f"10log10(mean linear NMSE): {nmse_to_db(np.mean(ddim_nmse_linear)):.3f} dB")
    print(f"Mean of per-mask NMSE(dB): {np.mean(ddim_nmse_db):.3f} dB")

    nmse_summary_save_path = (
        f"./results/nmse_summaries/"
        f"WCGAN_Enhanced_DDIM_{config.model.ddim_steps}steps_"
        f"train-{args.train}_test-{args.test}.pt"
    )
    os.makedirs(os.path.dirname(nmse_summary_save_path), exist_ok=True)
    torch.save({
        'ddim_steps': config.model.ddim_steps,
        'checkpoint': args.checkpoint,
        'train': args.train,
        'test': args.test,
        'records': nmse_summary_records,
        'mean_linear_nmse': float(np.mean(ddim_nmse_linear)),
        'mean_linear_nmse_db': float(nmse_to_db(np.mean(ddim_nmse_linear))),
        'mean_per_mask_nmse_db': float(np.mean(ddim_nmse_db)),
    }, nmse_summary_save_path)
    print(f"Saved NMSE summary to: {nmse_summary_save_path}")

print(f"\n{'='*50}")
print("All mask ratios testing completed!")
print(f"{'='*50}") 
