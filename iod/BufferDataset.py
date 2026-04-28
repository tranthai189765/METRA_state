import torch
from torch.utils.data import Dataset, DataLoader
import time

class BufferDataset(Dataset):
    def __init__(self, data, len):
        self.data = data
        self.len = len

    def __getitem__(self, index):
        keys = ['obs' , 'next_obs', 'options', 'next_options', 'dones', 'actions', 'psi_g', 's_0']
        epoch_data = {}
        for i in range(len(keys)):
            key = keys[i]
            if key in ['obs', 'next_obs', 'sub_goal', 's_0'] and 'obs_pixel' in self.data.keys():
                key_ = key + '_pixel'
                epoch_data[key] = self.data[key_][index]
            elif key not in self.data.keys():
                continue
            else:
                epoch_data[key] = self.data[key][index]
        return epoch_data

    def __len__(self):
        return self.len
