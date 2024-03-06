import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.parallel import DistributedDataParallel, DataParallel
from utils import get_sigma_time, get_sample_time, VESDE, get_config
from model import UNet3DModel
import matplotlib.pyplot as plt
from torch_ema import ExponentialMovingAverage
from dataloader import GalaxyDataset
import logging
import os
import sys

import argparse

parser = argparse.ArgumentParser(description='Train the model')
parser.add_argument('-j', default='galpure', type=str, help='job type')
parser.add_argument('--num_workers', default=1, type=int, help='number of workers for dataloader')
parser.add_argument('--task_id', default='training', type=str, help='task id')
parser.add_argument('--num_bins', default=1, type=int, help='number of bins')
args = parser.parse_args()

task_id = args.task_id

config = get_config('./config.json')
num_bins = args.num_bins
cosmo_dir = f'fiducial/galbin_{num_bins}/'
config.data.num_input_channels = int(num_bins * 2 + 1)
Nside = config.data.image_size
#DEVICE = config.device
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# Create directory structure
checkpoint_dir = os.path.join(config.model.workdir, f'checkpoints/galbin_{num_bins}')
data_path = config.model.workdir + cosmo_dir
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(data_path, exist_ok=True)

sigma_time = get_sigma_time(config.model.sigma_min, config.model.sigma_max)
sample_time = get_sample_time(config.model.sampling_eps, config.model.T)

# Build pytorch dataloaders and apply data preprocessing
# validation_dataset = GalaxyDataset(datadir='../diff_data/galaxies/', job_type='galsmear', train_or_val='validation')
# validation_loader = DataLoader(validation_dataset, config.training.batch_size, shuffle=True, num_workers=1)

scratch_ddir = f'/leonardo_scratch/large/userexternal/sgagnonh/diff_data/galbin_{num_bins}/'
if task_id == 'training':
    validation_dataset = GalaxyDataset(datadir=scratch_ddir, job_type=args.j, train_or_val='training', single_nf=0.4)
    validation_loader = DataLoader(validation_dataset, config.sampling.batch_size, shuffle=False, num_workers=args.num_workers)
elif task_id == 'validation':
    validation_dataset = GalaxyDataset(datadir=scratch_ddir, job_type=args.j, train_or_val='validation', single_nf=0.4)
    validation_loader = DataLoader(validation_dataset, config.sampling.batch_size, shuffle=False, num_workers=args.num_workers)
else:
    print('Invalid task_id')
    sys.exit(1)

# Initialize score model
model = DataParallel(UNet3DModel(config))
model = model.to(DEVICE)

# Define optimizer
optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.optim.lr,
        betas=(config.optim.beta1, 0.999),
        eps=config.optim.eps,
        weight_decay=config.optim.weight_decay                   
        )
ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)

sde = VESDE(config.model.sigma_min, config.model.sigma_max, config.model.num_scales, config.model.T, config.model.sampling_eps)

# Check for existing checkpoint
checkpoint_path = os.path.join(checkpoint_dir, 'best_checkpoint.pth')
if os.path.isfile(checkpoint_path):
    loaded_state = torch.load(checkpoint_path, map_location=DEVICE)
    optimizer.load_state_dict(loaded_state['optimizer'])
    model.load_state_dict(loaded_state['model'], strict=False)
    ema.load_state_dict(loaded_state['ema'])
    init_epoch = int(loaded_state['epoch'])
    logging.warning(f"Loaded checkpoint from {checkpoint_path}.")
else:
    logging.warning(f"No checkpoint found at {checkpoint_path}. Starting from scratch.")

model.eval()

def one_step(x, t):
    t_vec = torch.ones(shape[0], device=DEVICE) * t
    model_output = model(torch.cat([x, input_data], dim=1), t_vec)
    x, x_mean = sde.update_fn(x, t_vec, model_output=model_output)
    return x, x_mean

input_data = validation_dataset[0][0].to(DEVICE)
np.save(data_path + f'input_data_{task_id}.npy', np.array(input_data.clone().detach().cpu().numpy()))
label_data = validation_dataset[0][1].to(DEVICE)
np.save(data_path + f'label_data_{task_id}.npy', np.array(label_data.clone().detach().cpu().numpy()))
input_data = torch.tile(input_data, dims=(config.sampling.batch_size, 1, 1, 1, 1))
shape = (config.sampling.batch_size, 1, Nside, Nside, Nside)

samples = []
print('Sampling begins.')
for j in tqdm(range(config.sampling.num_samples//config.sampling.batch_size)):
    with torch.no_grad(), ema.average_parameters():
        x = sde.prior_sampling(shape).to(DEVICE)
        timesteps = sde.timesteps.to(DEVICE)
        for i in tqdm(range(sde.N)):
            t = timesteps[i]
            x, x_mean = one_step(x, t)
        samples.append(x_mean.detach().cpu().numpy())
    np.save(data_path + f'sample_{task_id}.npy', np.array(samples))
np.save(data_path + f'sample_{task_id}.npy', np.array(samples))
