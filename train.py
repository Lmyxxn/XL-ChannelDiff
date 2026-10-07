#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch, os, copy, argparse, random
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
args = parser.parse_args()

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
config.optim.beta1         = 0.5
config.optim.beta2         = 0.999
config.optim.amsgrad       = False
config.optim.eps           = 0.001
config.training.lambda_l1 = 10  # L1 loss weight
config.training.lambda_recon = 50  # predicted noise loss weight

# Training
config.training.batch_size     = 32
config.training.num_workers    = 4
config.training.n_epochs       = args.epochs

# Data
config.data.channel        = args.train
config.data.channels       = 2 # {Re, Im}
config.data.num_rx        = 1024
config.data.image_size     = [32, 32] # [Nx, Ny] for the transposed UPA channel
config.data.mask_ratios     = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

config.data.norm_channels  = 'global'
config.data.spacing_list   = [0.5]

# Seeds for train and test datasets
train_seed, val_seed = 1234, 4321

# Get datasets and loaders for channels

dataset     = Channels(train_seed, config, norm=config.data.norm_channels, num_rx=config.data.num_rx, mask_ratios=config.data.mask_ratios)
dataloader  = DataLoader(dataset, batch_size=config.training.batch_size, 
         shuffle=True, num_workers=config.training.num_workers, drop_last=True)

# Validation data
val_datasets, val_loaders, val_iters = [], [], []
for idx in range(len(config.data.spacing_list)):
    # Validation config
    val_config = copy.deepcopy(config)
    val_config.data.spacing_list = [config.data.spacing_list[idx]]
    # Create locals
    val_datasets.append(Channels(val_seed, val_config, norm=[dataset.mean, dataset.std], num_rx=config.data.num_rx, mask_ratios=config.data.mask_ratios))
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
criterion_L1 = torch.nn.L1Loss()

# Initialize EMA helper
ema_helper = EMAHelper(mu=config.model.ema_rate)
ema_helper.register(diffuser)

# Load DDIM
ddim = DDIM(config.device, config.model.num_timesteps, config.model.min_beta, config.model.max_beta)

# Sample fixed validation data
val_H_full_list, val_H_partial_list, mask_list = [], [], []
for idx in range(len(config.data.spacing_list)):
    val_sample = next(val_iters[idx])
    val_H_full_list.append(val_sample['H_full'].cuda())
    val_H_partial_list.append(val_sample['H_partial'].cuda())
    mask_list.append(val_sample['mask'].cuda())

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

def get_optimizer(config, parameters):
    return optim.AdamW(parameters, lr=config.optim.lr,
                       weight_decay=config.optim.weight_decay,
                       betas=(config.optim.beta1, 0.999),
                       amsgrad=config.optim.amsgrad, eps=config.optim.eps)

# Instantiate optimizer
optimizer_G = get_optimizer(config, diffuser.parameters())
optimizer_D = get_optimizer(config, discriminator.parameters())

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
        
        # Update the time-conditioned critic.
        optimizer_D.zero_grad()
        
        t = torch.randint(0, config.model.num_timesteps, (current_batch_size,)).to(config.device)
        t_embedding = t.view(-1, 1, 1, 1).expand(h_full.shape[0], 1, h_full.shape[2], h_full.shape[3])

        eps = torch.randn_like(h_full).to(config.device)
        x_t = ddim.sample_forward(h_full, t, eps)
        eps_theta = diffuser(x_t, t, h_partial, mask)
        fake_h = ddim.sample_backward_cgan(x_t, t, eps_theta)
        
        real_pred = discriminator(h_full, t_embedding)
        fake_pred = discriminator(fake_h.detach(), t_embedding)
        
        d_loss = -(real_pred.mean() - fake_pred.mean())
        d_loss.backward()
        optimizer_D.step()
        optimizer_G.zero_grad()

        real_pred = discriminator(h_full, t_embedding)
        fake_pred = discriminator(fake_h.detach(), t_embedding)

        # Retain the historical noise and channel reconstruction losses.
        g_loss_recon = criterion_L1(eps_theta, eps) * config.training.lambda_recon
        
        g_loss_l1 = criterion_L1(fake_h, h_full) * config.training.lambda_l1

        g_loss_gan = -fake_pred.mean()

        g_loss = g_loss_recon + g_loss_l1 + g_loss_gan

        g_loss.backward()
        optimizer_G.step()
        
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

            loss = criterion_L1(eps, eps_theta) * config.training.lambda_recon
            val_loss += loss.item()
    val_loss = val_loss * 100 / len(config.data.spacing_list)

    writer.add_scalar('Loss/total', total_loss, epoch)
    writer.add_scalar('Loss/generator', total_g_loss, epoch)
    writer.add_scalar('Loss/discriminator', total_d_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)

    print('Epoch %d, Total Loss %.3f, G Loss %.3f, D Loss %.3f, Val Loss %.3f' % 
          (epoch, total_loss, total_g_loss, total_d_loss, val_loss))

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
        torch.save(checkpoint, os.path.join(config.log_path, 'model_best.pt'))
    torch.save(checkpoint, os.path.join(config.log_path, 'model_latest.pt'))

# Close the TensorBoard writer
writer.close()
