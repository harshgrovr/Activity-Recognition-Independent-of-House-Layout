import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class datasetHDF5(Dataset):
    def __init__(self,objectChannel, curr_file_path, ):
        super(datasetHDF5, self).__init__()
        self.h5_path = curr_file_path
        self.h5File = h5py.File(self.h5_path, 'r')
        self.length = self.h5File['length'].value
        self.objectChannel = objectChannel

    def __getitem__(self, idx):
        input = self.h5File['images'][idx, :, :, :]
        label = self.h5File['labels'][idx]
        input = np.concatenate((input, self.objectChannel), axis=2)
        input = torch.as_tensor(np.array(input).astype('float'))
        label = torch.as_tensor(np.array(label).astype('long'))
        return input, label

    def __len__(self):
        # Equal to number of rows in current h5 file corresponding to current data
        return self.length





