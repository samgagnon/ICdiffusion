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

task_id = 1
cosmo_dir = 'fiducial/'

config = get_config('./config.json')
Nside = config.data.image_size
#DEVICE = config.device
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# Create directory structure
checkpoint_dir = os.path.join(config.model.workdir, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

sigma_time = get_sigma_time(config.model.sigma_min, config.model.sigma_max)
sample_time = get_sample_time(config.model.sampling_eps, config.model.T)

data_path = config.model.workdir + cosmo_dir


# Build pytorch dataloaders and apply data preprocessing
validation_dataset = GalaxyDataset(datadir='../diff_data/galaxies/', job_type='galsmear', train_or_val='validation')
validation_loader = DataLoader(validation_dataset, config.training.batch_size, shuffle=True, num_workers=1)

# Initialize score model
model = UNet3DModel(config)
#model = DataParallel(model)
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
checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
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
np.save(data_path + 'input_data{}.npy'.format(task_id), np.array(input_data.clone().detach().cpu().numpy()))
label_data = validation_dataset[0][1].to(DEVICE)
np.save(data_path + 'label_data{}.npy'.format(task_id), np.array(label_data.clone().detach().cpu().numpy()))
input_data = torch.tile(input_data, dims=(config.sampling.batch_size, 1, 1, 1, 1)) # shape is too large
shape = (config.sampling.batch_size, 1, Nside, Nside, Nside)

print(input_data.shape, label_data.shape)

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
    np.save(data_path + 'sample{}.npy'.format(task_id), np.array(samples))
np.save(data_path + 'sample{}.npy'.format(task_id), np.array(samples))
