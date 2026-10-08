#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch, os, copy, argparse, random
import tempfile
from cgan_enhanced_ddim import DDIM

from ema import EMAHelper

from DiT.enhanced_models import DiT

from loaders              import Channels
from torch.utils.data import DataLoader
from dotmap           import DotMap
from tqdm import tqdm
import torch.optim as optim
from tensorboardX import SummaryWriter
from GAN.models_enhanced_wcgan import Discriminator

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0,1,2,3')
parser.add_argument('--train', type=str, default='CDL-C')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--max_batches', type=int, default=0)
parser.add_argument('--output_root', type=str, default='./repro_models')
parser.add_argument('--resume', type=str, default='')
parser.add_argument('--data_root', type=str, default='./data')
parser.add_argument('--validation_file', type=str, default='')
parser.add_argument('--lr_g', type=float, default=0.0002)
parser.add_argument('--lr_d', type=float, default=0.0002)
parser.add_argument('--warmup_steps', type=int, default=0)
parser.add_argument('--loss_type', choices=('l1', 'l2'), default='l1')
args = parser.parse_args()
if args.lr_g <= 0 or args.lr_d <= 0 or args.warmup_steps < 0:
    parser.error('Learning rates must be positive and warmup_steps must be nonnegative.')

# Disable TF32 due to potential precision issues
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32       = False
torch.backends.cudnn.benchmark        = True
# GPU
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) 

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# Model config
config          = DotMap()
# Set the device
config.device = 'cuda:0'
# Inner model
config.model.ema           = True
config.model.ema_rate      = 0.999
config.model.num_timesteps = 1000
config.model.min_beta      = 0.0001
config.model.max_beta      = 0.02
config.model.type          = 'DiT'
config.model.patch_size    = 8
config.model.hidden_size   = 256
config.model.depth         = 3

# Optimizer
config.optim.weight_decay  = 0.000 # No weight decay
config.optim.optimizer     = 'AdamW'
config.optim.lr            = 0.0002
config.optim.lr_g          = args.lr_g
config.optim.lr_d          = args.lr_d
config.optim.warmup_steps  = args.warmup_steps
config.optim.beta1         = 0.5
config.optim.beta2         = 0.999
config.optim.amsgrad       = False
config.optim.eps           = 0.001
config.training.lambda_channel = 100
config.training.lambda_recon = 500
config.training.critic_steps = 5
config.training.critic_clip = 0.01
config.training.reconstruction_loss = args.loss_type + '_mean'

# Training
config.training.batch_size     = 256
config.training.num_workers    = 4
config.training.n_epochs       = args.epochs

# Data
config.data.channel        = args.train
config.data.root           = args.data_root
config.data.validation_file = args.validation_file
config.data.split_protocol = 'independent validation file' if args.validation_file else 'seed42 training-source holdout of 1000 samples'
config.data.channels       = 2 # {Re, Im}
config.data.num_rx        = 1024
config.data.image_size     = [32, 32] # [Nx, Ny] for the transposed UPA channel
config.data.mask_ratios     = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

config.data.norm_channels  = 'global'
config.data.spacing_list   = [0.5]

# Seeds for train and test datasets
train_seed, val_seed = 1234, 4321

# Get datasets and loaders for channels

dataset     = Channels(train_seed, config, norm=config.data.norm_channels, num_rx=config.data.num_rx, mask_ratios=config.data.mask_ratios, split='train', validation_file=args.validation_file)
config.data.normalization = [complex(dataset.mean), float(dataset.std)]
config.data.train_samples = len(dataset)
dataloader  = DataLoader(dataset, batch_size=config.training.batch_size, 
         shuffle=True, num_workers=config.training.num_workers, drop_last=True)

# Validation data
val_datasets, val_loaders, val_iters = [], [], []
for idx in range(len(config.data.spacing_list)):
    # Validation config
    val_config = copy.deepcopy(config)
    val_config.data.spacing_list = [config.data.spacing_list[idx]]
    # Create locals
    val_datasets.append(Channels(train_seed, val_config, norm=[dataset.mean, dataset.std], num_rx=config.data.num_rx, mask_ratios=config.data.mask_ratios, split='validation', validation_file=args.validation_file))
    val_loaders.append(DataLoader(
        val_datasets[-1], batch_size=len(val_datasets[-1]),
        shuffle=False, num_workers=0, drop_last=True))
    val_iters.append(iter(val_loaders[-1])) # For validation

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

diffuser = diffuser.to(config.device)

# The diffusion Transformer is the generator; only the critic is needed here.
discriminator = Discriminator(channels=2).to(config.device)

# Loss function
def reconstruction_loss(pred, target):
    error = pred - target
    return error.abs().mean() if args.loss_type == 'l1' else error.square().mean()

# Initialize EMA helper
ema_helper = EMAHelper(mu=config.model.ema_rate)
ema_helper.register(diffuser)

# Load DDIM
ddim = DDIM(config.device, config.model.num_timesteps, config.model.min_beta, config.model.max_beta)

# Sample fixed validation data
validation_numpy_state = np.random.get_state()
np.random.seed(123456)
val_H_full_list, val_H_partial_list, mask_list = [], [], []
for idx in range(len(config.data.spacing_list)):
    val_sample = next(val_iters[idx])
    val_H_full_list.append(val_sample['H_full'].cuda())
    val_H_partial_list.append(val_sample['H_partial'].cuda())
    mask_list.append(val_sample['mask'].cuda())
np.random.set_state(validation_numpy_state)

# Logging
config.log_path = os.path.join(args.output_root, 'seed%d' % args.seed, 'WCGAN_enhanced_DDIM_l1loss_with_time_%s_rx_%d_mixedMask_ratio_dimension%d_depth%d_patch%d' % (
    config.model.type, 
    config.data.num_rx, 
    config.model.hidden_size, 
    config.model.depth, 
    config.model.patch_size
), args.train)
os.makedirs(config.log_path, exist_ok=True)

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir=config.log_path)

def save_checkpoint_atomic(checkpoint, path):
    fd, temporary = tempfile.mkstemp(prefix='.checkpoint-', suffix='.tmp', dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(checkpoint, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

def get_optimizer(config, parameters, lr):
    return optim.AdamW(parameters, lr=lr,
                       weight_decay=config.optim.weight_decay,
                       betas=(config.optim.beta1, 0.999),
                       amsgrad=config.optim.amsgrad, eps=config.optim.eps)

# Instantiate optimizer
optimizer_G = get_optimizer(config, diffuser.parameters(), config.optim.lr_g)
optimizer_D = get_optimizer(config, discriminator.parameters(), config.optim.lr_d)

def warmup_factor(step):
    if config.optim.warmup_steps == 0:
        return 1.0
    return min((step + 1) / config.optim.warmup_steps, 1.0)

scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, warmup_factor)
global_step = 0
print('%s experiment: batch=%d, G lr=%g, D lr=%g, G warmup=%d updates' % (
    config.training.reconstruction_loss, config.training.batch_size,
    config.optim.lr_g, config.optim.lr_d, config.optim.warmup_steps))

train_loss, val_loss  = [], []
iter = 0
start_epoch = 0
best_val_loss = float('inf')

if args.resume:
    checkpoint = torch.load(args.resume, map_location=config.device, weights_only=False)
    diffuser.load_state_dict(checkpoint['diffuser_state'])
    discriminator.load_state_dict(checkpoint['discriminator_state'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state'])
    if 'scheduler_G_state' in checkpoint:
        scheduler_G.load_state_dict(checkpoint['scheduler_G_state'])
    elif config.optim.warmup_steps:
        raise ValueError('Warmup resume requires a checkpoint containing scheduler state.')
    global_step = int(checkpoint.get('global_step', 0))
    ema_helper.load_state_dict(checkpoint['ema_state'])
    start_epoch = int(checkpoint['epoch']) + 1
    best_val_loss = float(checkpoint.get('best_val_loss', checkpoint.get('val_loss', float('inf'))))
    print('Resuming from epoch %d using %s' % (start_epoch, args.resume))

# Training loop
for epoch in range(start_epoch, config.training.n_epochs):
    total_loss = 0
    total_g_loss = 0
    total_d_loss = 0
    iter = 0
    for i, sample in tqdm(enumerate(dataloader)):
        diffuser.train()
        discriminator.train()
        iter += 1
        current_batch_size = sample['H_full'].shape[0]
        h_full = sample['H_full'].to(config.device)
        h_partial = sample['H_partial'].to(config.device)
        mask = sample['mask'].to(config.device)
        
        for _ in range(config.training.critic_steps):
            optimizer_D.zero_grad(set_to_none=True)
            t = torch.randint(config.model.num_timesteps, (current_batch_size,), device=config.device)
            eps = torch.randn_like(h_full)
            x_t = ddim.sample_forward(h_full, t, eps)
            with torch.no_grad():
                fake_h = ddim.sample_backward_cgan(x_t, t, diffuser(x_t, t, h_partial, mask))
            d_loss = discriminator(fake_h).mean() - discriminator(h_full).mean()
            d_loss.backward()
            optimizer_D.step()
            with torch.no_grad():
                for parameter in discriminator.parameters():
                    parameter.clamp_(-config.training.critic_clip, config.training.critic_clip)

        for parameter in discriminator.parameters():
            parameter.requires_grad_(False)
        optimizer_G.zero_grad(set_to_none=True)
        t = torch.randint(config.model.num_timesteps, (current_batch_size,), device=config.device)
        eps = torch.randn_like(h_full)
        x_t = ddim.sample_forward(h_full, t, eps)
        eps_theta = diffuser(x_t, t, h_partial, mask)
        fake_h = ddim.sample_backward_cgan(x_t, t, eps_theta)
        g_loss_recon = reconstruction_loss(eps_theta, eps) * config.training.lambda_recon
        g_loss_channel = reconstruction_loss(fake_h, h_full) * config.training.lambda_channel
        g_loss_gan = -discriminator(fake_h).mean()
        if epoch == start_epoch and i == 0:
            adv_grad = torch.autograd.grad(g_loss_gan, eps_theta, retain_graph=True)[0]
            if not torch.isfinite(adv_grad).all() or not adv_grad.abs().sum() > 0:
                raise RuntimeError('Adversarial gradients do not reach the diffuser.')
        g_loss = g_loss_recon + g_loss_channel + g_loss_gan
        if not torch.isfinite(g_loss):
            raise FloatingPointError('Nonfinite generator loss.')
        g_loss.backward()
        optimizer_G.step()
        scheduler_G.step()
        global_step += 1
        for parameter in discriminator.parameters():
            parameter.requires_grad_(True)
        
        # EMA update
        ema_helper.update(diffuser)

        total_loss += (g_loss + d_loss).item()
        total_g_loss += g_loss.item()
        total_d_loss += d_loss.item()

        if args.max_batches > 0 and iter >= args.max_batches:
            break

    total_loss = total_loss * 100 / iter
    total_g_loss = total_g_loss * 100 / iter
    total_d_loss = total_d_loss * 100 / iter

    val_loss = 0
    diffuser.eval()
    validation_rng = torch.cuda.get_rng_state()
    torch.cuda.manual_seed(123456)
    with torch.no_grad():
        for idx in range(len(config.data.spacing_list)):
            h_full = val_H_full_list[idx].to(config.device)
            h_partial = val_H_partial_list[idx].to(config.device)
            mask = mask_list[idx].to(config.device)
            current_batch_size = h_full.shape[0]
            t = torch.randint(0, config.model.num_timesteps, (current_batch_size,)).to(config.device)
            eps = torch.randn_like(h_full).to(config.device)
            x_t = ddim.sample_forward(h_full, t, eps)
            eps_theta = diffuser(x_t, t, h_partial, mask)
            fake_h = ddim.sample_backward_cgan(x_t, t, eps_theta)

            loss = reconstruction_loss(eps, eps_theta) * config.training.lambda_recon
            val_loss += loss.item()
    val_loss = val_loss * 100 / len(config.data.spacing_list)
    torch.cuda.set_rng_state(validation_rng)

    writer.add_scalar('Loss/total', total_loss, epoch)
    writer.add_scalar('Loss/generator', total_g_loss, epoch)
    writer.add_scalar('Loss/discriminator', total_d_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('LR/generator', optimizer_G.param_groups[0]['lr'], global_step)
    writer.add_scalar('LR/discriminator', optimizer_D.param_groups[0]['lr'], global_step)

    print('Epoch %d, Total Loss %.3f, G Loss %.3f, D Loss %.3f, Val Loss %.3f' % 
          (epoch, total_loss, total_g_loss, total_d_loss, val_loss))
    print('Optimizer updates %d, next G lr %.8g, D lr %.8g' % (
        global_step, optimizer_G.param_groups[0]['lr'], optimizer_D.param_groups[0]['lr']))

    # Select the best checkpoint by validation noise-reconstruction loss.
    is_best = val_loss < best_val_loss
    if is_best:
        best_val_loss = val_loss
    checkpoint = {
        'diffuser_state': diffuser.state_dict(),
        'ema_state': ema_helper.state_dict(),
        'ddim_config': {
            'num_timesteps': config.model.num_timesteps,
            'min_beta': config.model.min_beta,
            'max_beta': config.model.max_beta,
            'device': config.device
        },
        'discriminator_state': discriminator.state_dict(),
        'optimizer_G_state': optimizer_G.state_dict(),
        'optimizer_D_state': optimizer_D.state_dict(),
        'scheduler_G_state': scheduler_G.state_dict(),
        'global_step': global_step,
        'config': config,
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'best_val_loss': best_val_loss,
        'total_loss': total_loss,
        'g_loss': total_g_loss,
        'd_loss': total_d_loss
    }
    if is_best:
        save_checkpoint_atomic(checkpoint, os.path.join(config.log_path, 'model_best.pt'))
    save_checkpoint_atomic(checkpoint, os.path.join(config.log_path, 'model_latest.pt'))

# Close the TensorBoard writer
writer.close()
