'''
A specialized dataloader for preprocessed galaxy and ionization field data.
This version loads data through active I/O
Samuel Gagnon-Hartman, Scuola Normale Superiore 2024
'''

import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.parallel import DataParallel


class GalaxyDataset(TensorDataset):
    def __init__(self, datadir, job_type='galsmear', train_or_val='training', single_nf=False, transform=None,\
                 give_filenames=False):
        super().__init__()
        assert job_type in ['mean_std', 'galpure', 'galsmear']
        assert train_or_val in ['training', 'validation']
        # if transformations are desired, set them here
        self.transform = transform
        self.give_filenames = give_filenames
        # initializes dataset locations
        self.muv_files = []
        self.lya_files = []
        self.count_files = []
        self.label_files = []
        muv_dir = datadir + job_type + '/muv_cubes/'
        lya_dir = datadir + job_type + '/lya_cubes/'
        count_dir = datadir + 'count/'
        label_dir = datadir + 'xHI_maps/'
        # load relevant file directories
        muv_list = os.listdir(muv_dir)
        lya_list = os.listdir(lya_dir)
        count_list = os.listdir(count_dir)
        label_list = os.listdir(label_dir)
        list_list = [muv_list, lya_list, count_list, label_list]
        dir_list = [muv_dir, lya_dir, count_dir, label_dir]
        data_list = [self.muv_files, self.lya_files, self.count_files, self.label_files]
        # apply desired cuts
        for i, l in enumerate(list_list):
            for file in l:
                if file.split('_')[0][:4] != 'seed':
                    continue
                seed = float(file.split('_')[0][4:])
                # bypass select data
                # focus only on a single ionization efficiency
                if single_nf is not None:
                    nf = float(file.split('_')[4][2:])
                    if nf < single_nf - 0.1 or nf > single_nf + 0.1:
                        continue
                if train_or_val=='training' and seed>7:
                    continue
                elif train_or_val=='validation' and seed<8:
                    continue
                # get relevant object and add relevant directory
                data_list[i] += [f'{dir_list[i]}{file}']

    def __getitem__(self, index):
        muv_file = self.muv_files[index]
        lya_file = self.lya_files[index]
        labels_file = self.label_files[index]
        muv = torch.from_numpy(np.stack(np.load(muv_file), axis=0))
        lya = torch.from_numpy(np.stack(np.load(lya_file), axis=0))
        labels = torch.from_numpy(np.stack(np.load(labels_file), axis=0))
        # unsqueeze channel dimension
        # muv = torch.unsqueeze(muv, dim=0)
        # lya = torch.unsqueeze(lya, dim=0)
        data = torch.cat((muv, lya), dim=0)
        labels = torch.unsqueeze(labels, dim=0)
        # convert datatype to float
        data = data.float()
        labels = labels.float()
        # apply transform if necessary
        if self.transform is not None:
            data = self.transform(data)
        if self.give_filenames:
            return data, labels, muv_file
        else:
            return data, labels

    def __len__(self):
        return len(self.muv_files)
    
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
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # argument parser
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--num_bins', type=int, default=1)
    args = parser.parse_args()
    
    # create dataloaders
    t1 = time()
    scratch_ddir = f'/leonardo_scratch/large/userexternal/sgagnonh/diff_data/galbin_{args.num_bins}/'
    training_dataset = GalaxyDataset(datadir=scratch_ddir, job_type='galsmear', train_or_val='training', single_nf=0.4)
    validation_dataset = GalaxyDataset(datadir=scratch_ddir, job_type='galsmear', train_or_val='validation', single_nf=0.4)
    training_loader = DataLoader(training_dataset, config.training.batch_size, shuffle=True, num_workers=args.num_workers)
    validation_loader = DataLoader(validation_dataset, config.training.batch_size, shuffle=True, num_workers=args.num_workers)
    t2 = time()
    print(f'{t2-t1} seconds to load data')

    # test dataloader
    print(len(training_dataset))
    for i, data_list in enumerate(training_loader):
        if i == 0:
            print(data_list[0].mean(), data_list[0].std())
            print(data_list[1].mean(), data_list[1].std())
        input_data = data_list[0].to(DEVICE)
        label_data = data_list[1].to(DEVICE)
        print(input_data.shape, label_data.shape)
        break
    t3 = time()
    print(f'{t3-t2} seconds to load first batch')

    # modify config for testing
    config.data.num_input_channels = int(args.num_bins * 2 + 1)

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