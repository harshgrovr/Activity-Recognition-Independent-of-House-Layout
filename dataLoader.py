import torch
from torch.utils.data import Dataset
import h5py
import numpy as np


class datasetHDF5(Dataset):
    def __init__(self, h5_path):
        super(datasetHDF5, self).__init__()
        self.h5File = h5py.File(h5_path, 'r')

        shape = self.h5File['images'].shape
        shape = self.h5File['labels'].shape
        print('keys are: ',list(self.h5File.keys()), 'shape is: ', shape)

    def __getitem__(self, idx):
        input = self.h5File['images'][idx, :, :, :]
        label = self.h5File['labels'][idx]
        print(type(input), type(label))
        input = torch.as_tensor(np.array(input).astype('float'))
        label = torch.as_tensor(np.array(label).astype('long'))
        return input, label

    def __len__(self):
        self.length = len(self.h5File)
        return self.length




