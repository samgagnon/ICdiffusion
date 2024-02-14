import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel, DataParallel
from utils import get_sigma_time, get_sample_time, get_config
from model import UNet3DModel
from dataloader import GalaxyDataset
torch.backends.cudnn.benchmark = True
import os
import logging
from torch_ema import ExponentialMovingAverage

import argparse

parser = argparse.ArgumentParser(description='Train the model')
parser.add_argument('-j', default='galpure', type=str, help='job type')
parser.add_argument('--num_workers', default=1, type=int, help='number of workers for dataloader')
args = parser.parse_args()

config = get_config('./config.json')
Nside = config.data.image_size
# for multi-gpu training
# original read cuda:0
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Create directory structure
checkpoint_dir = os.path.join(config.model.workdir, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

sigma_time = get_sigma_time(config.model.sigma_min, config.model.sigma_max)
sample_time = get_sample_time(config.model.sampling_eps, config.model.T)

gfile_stream = open(os.path.join(config.model.workdir, 'stdout.txt'), 'w')
handler = logging.StreamHandler(gfile_stream)
formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
handler.setFormatter(formatter)
logger = logging.getLogger()
logger.addHandler(handler)
logger.setLevel('INFO')

def train_one_epoch():
    avg_loss = 0.
    counter = 0
    for i, data_list in enumerate(training_loader):
        input_data = data_list[0].to(DEVICE)
        label_data = data_list[1].to(DEVICE)
        
        B = label_data.size(dim=0)        
        
        # Sample random observation noise
        input_data += config.data.noise_sigma * torch.randn_like(input_data).to(DEVICE)

        # Sample random time steps
        time_steps = sample_time(shape=(B,)).to(DEVICE)
        sigmas = sigma_time(time_steps).to(DEVICE)
        sigmas = sigmas[:,None,None,None,None]


        # Generate noise perturbed input
        z = torch.randn_like(label_data,  device=DEVICE)
        inputs = torch.cat([label_data + sigmas * z, input_data], dim=1)

        optimizer.zero_grad()
        output = model(inputs, time_steps)

        # Optimize with score matching loss
        loss = torch.sum(torch.square(output + z)) /  B
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.grad_clip)
        optimizer.step()
        ema.update()
        avg_loss += loss.item()
        counter += 1
    return avg_loss/counter

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
init_epoch = 0

# Check for existing checkpoint
# don't worry about this for now
# checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
# if os.path.isfile(checkpoint_path):
#     loaded_state = torch.load(checkpoint_path, map_location=DEVICE)
#     optimizer.load_state_dict(loaded_state['optimizer'])
#     model.load_state_dict(loaded_state['model'], strict=False)
#     ema.load_state_dict(loaded_state['ema'])
#     init_epoch = int(loaded_state['epoch'])
#     logging.warning(f"Loaded checkpoint from {checkpoint_path}.")
# else:
#     logging.warning(f"No checkpoint found at {checkpoint_path}. Starting from scratch.")


# Build pytorch dataloaders and apply data preprocessing
scratch_ddir = '/leonardo_scratch/large/userexternal/sgagnonh/diff_data/diff_data/galaxies/'
training_dataset = GalaxyDataset(datadir=scratch_ddir, job_type=args.j, train_or_val='training', single_nf=0.4)
training_loader = DataLoader(training_dataset, config.training.batch_size, shuffle=True, num_workers=args.num_workers)
# validation_dataset = GalaxyDataset(datadir=scratch_ddir, job_type='galsmear', train_or_val='validation', single_ion_eff=True)
# validation_loader = DataLoader(validation_dataset, config.training.batch_size, shuffle=True, num_workers=1)

model.train(True)

logging.info('Starting training loop.')
for epoch in range(init_epoch, config.training.n_epochs + 1):
    
    # TODO pass arguments to train_one_epoch
    # rather than using global variables
    avg_loss = train_one_epoch()
    
    if epoch % 10 == 0:
        logging.info('epoch: {}, training loss: {}'.format(epoch+1, avg_loss))
        torch.save(
            dict(optimizer=optimizer.state_dict(), model=model.module.state_dict(), ema=ema.state_dict(), epoch=epoch),
            os.path.join(checkpoint_dir, f'checkpoint_{epoch}.pth')
            )
    torch.save(
        dict(optimizer=optimizer.state_dict(), model=model.module.state_dict(), ema=ema.state_dict(), epoch=epoch),
        os.path.join(checkpoint_dir, f'checkpoint.pth')
        )

logging.info('Training complete.')
gfile_stream.close()

