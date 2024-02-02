'''
A specialized dataloader for preprocessed galaxy and ionization field data.
Samuel Gagnon-Hartman, Scuola Normale Superiore 2024
'''

import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.parallel import DistributedDataParallel, DataParallel


class GalaxyDataset(TensorDataset):
    def __init__(self, datadir, job_type='galsmear', train_or_val='training', transform=None):
        super().__init__()
        assert job_type in ['mean_std', 'galpure', 'galsmear']
        assert train_or_val in ['training', 'validation']
        self.data = []
        self.labels = []
        if job_type == 'galsmear':
            img_dir = datadir + 'galsmear/muv_cubes/'
            label_dir = datadir + 'xHI_maps/'
            data_list = os.listdir(img_dir)
            label_list = os.listdir(label_dir)
            for file in data_list:
                if file.split('_')[0][:4] != 'seed':
                    continue
                seed = float(file.split('_')[0][4:])
                # bypass select data
                if train_or_val=='training' and seed>7:
                    continue
                elif train_or_val=='validation' and seed<8:
                    continue
                self.data += [np.load(img_dir+file)]
            for file in label_list:
                if file.split('_')[0][:4] != 'seed':
                    continue
                seed = float(file.split('_')[0][4:])
                # bypass select data
                if train_or_val=='training' and seed>7:
                    continue
                elif train_or_val=='validation' and seed<8:
                    continue
                # read labels from seeds 1-7
                self.labels += [np.load(label_dir+file)]
            self.data = torch.from_numpy(np.stack(self.data, axis=0))
            print(self.data.shape)
            self.labels = torch.from_numpy(np.stack(self.labels, axis=0))
            # unsqueeze channel dimension
            self.data = torch.unsqueeze(self.data, dim=1)
            print(self.data.shape)
            self.labels = torch.unsqueeze(self.labels, dim=1)
        # apply transform if necessary
        self.transform = transform

    def __getitem__(self, index):
        data = self.data[index]
        labels = self.labels[index]
        if self.transform is not None:
            data = self.transform(data)
        return data, labels

    def __len__(self):
        return len(self.data)
    
if __name__ == '__main__':
    # imports
    from model import UNet3DModel
    from utils import get_config, get_sample_time, get_sigma_time
    config = get_config('./config.json')
    sigma_time = get_sigma_time(config.model.sigma_min, config.model.sigma_max)
    sample_time = get_sample_time(config.model.sampling_eps, config.model.T)

    from torch.utils.data import DataLoader
    from torch_ema import ExponentialMovingAverage
    from time import time

    # device
    DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # argument parser
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()
    
    # create dataloaders
    t1 = time()
    training_dataset = GalaxyDataset(datadir='../diff_data/galaxies/', job_type='galsmear', train_or_val='training')
    validation_dataset = GalaxyDataset(datadir='../diff_data/galaxies/', job_type='galsmear', train_or_val='validation')
    training_loader = DataLoader(training_dataset, config.training.batch_size, shuffle=True, num_workers=args.num_workers)
    validation_loader = DataLoader(validation_dataset, config.training.batch_size, shuffle=True, num_workers=args.num_workers)
    t2 = time()
    print(f'{t2-t1} seconds to load data')

    # test dataloader
    for i, data_list in enumerate(training_loader):
        input_data = data_list[0].to(DEVICE)
        label_data = data_list[1].to(DEVICE)
        print(input_data.shape, label_data.shape)
        break
    t3 = time()
    print(f'{t3-t2} seconds to load first batch')

    # test network training
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
    t4 = time()
    print(f'{t4-t3} seconds to train first batch')