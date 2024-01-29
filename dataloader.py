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
                seed = float(file.split('_')[0][4:])
                # bypass select data
                if train_or_val=='training' and seed>7:
                    continue
                elif train_or_val=='validation' and seed<8:
                    continue
                self.data += [np.load(img_dir+file)]
            for file in label_list:
                seed = float(file.split('_')[0][4:])
                # bypass select data
                if train_or_val=='training' and seed>7:
                    continue
                elif train_or_val=='validation' and seed<8:
                    continue
                # read labels from seeds 1-7
                self.labels += [np.load(label_dir+file)]
            self.data = torch.from_numpy(np.concatenate(self.data, axis=0))
            self.labels = torch.from_numpy(np.concatenate(self.labels, axis=0))
            # unsqueeze channel dimension
            self.data = torch.unsqueeze(self.data, dim=1)
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
    training_dataset = GalaxyDataset(datadir='../diff_data/galaxies/', job_type='galsmear', train_or_val='training')
    validation_dataset = GalaxyDataset(datadir='../diff_data/galaxies/', job_type='galsmear', train_or_val='validation')
    print(len(training_dataset), len(validation_dataset))
    print(training_dataset[0].shape, validation_dataset[0].shape)
