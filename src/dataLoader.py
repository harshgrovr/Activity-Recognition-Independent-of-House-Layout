import cv2
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import pandas as pd
from config.config import config
class datasetHDF5(Dataset):
    def __init__(self,objectChannel,sensorChannel, curr_file_path, ):
        super(datasetHDF5, self).__init__()
        self.h5_path = curr_file_path
        with h5py.File(self.h5_path, 'r') as f:
            self.h5File = f
            self.length = self.h5File['length'].value - config['seq_dim'] + 1

        objectChannel = np.repeat(objectChannel[np.newaxis, ...], config['seq_dim'], axis=0)
        sensorChannel = np.repeat(sensorChannel[np.newaxis, ...], config['seq_dim'], axis=0)

        self.objectChannel= objectChannel
        self.sensorChannel = sensorChannel

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            self.h5File = f
            input = self.h5File['images'][idx : idx + config['seq_dim'], :, :, :]
            label = self.h5File['labels'][idx: idx + config['seq_dim']]
            input = np.concatenate((input, self.objectChannel), axis=3)
            input = np.concatenate((input, self.sensorChannel), axis=3)
            # input = torch.as_tensor(np.array(input).astype('float'))
            # label = torch.as_tensor(np.array(label).astype('long'))

            return input, label

    def __len__(self):
        # Equal to number of rows in current h5 file corresponding to current data
        return self.length


class datasetCSV(Dataset):
    def __init__(self, trainData, seq_dim):
        super(datasetCSV, self).__init__()
        self.csv_input = trainData
        self.seq_dim = seq_dim

    def __getitem__(self, idx):
        self.input = self.csv_input[idx][0]
        self.label = self.csv_input[idx][1]
        self.input = torch.as_tensor(np.array(self.input).astype('float'))
        self.label = torch.as_tensor(np.array(self.label).astype('long'))
        return self.input, self.label

    def __len__(self):
        return len(self.csv_input)



