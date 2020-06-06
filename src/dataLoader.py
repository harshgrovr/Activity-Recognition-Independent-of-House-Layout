import os

import cv2
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import pandas as pd
from config.config import config
import datetime
from datetime import datetime

class datasetHDF5(Dataset):
    def __init__(self, objectChannel,sensorChannel, h5Files, h5Directory, train_index, dataSequence):
        super(datasetHDF5, self).__init__()
        self.length = 0
        self.i = 0
        self.count = 0
        self.csv_input = dataSequence

        self.h5Files = h5Files
        self.train_index = train_index
        self.h5Directory = h5Directory

        # Load First file
        self.currentFileName = self.h5Files[self.train_index[self.count]][-1]
        self.currentFileName = self.currentFileName + '.h5'
        self.currentFilePath = os.path.join(self.h5Directory, self.currentFileName)

        # calculate total length of the dataloader
        self.h5FilelengthList = []
        for index in self.train_index:
            h5_path = os.path.join(self.h5Directory, self.h5Files[index][0]) + '.h5'
            with h5py.File(h5_path, 'r') as f:
                self.h5File = f
                length = self.h5File['length'].value - config['seq_dim'] + 1
                self.h5FilelengthList.append(length)
                self.length += length

        objectChannel = np.repeat(objectChannel[np.newaxis, ...], config['seq_dim'], axis=0)
        sensorChannel = np.repeat(sensorChannel[np.newaxis, ...], config['seq_dim'], axis=0)

        self.objectChannel= objectChannel
        self.sensorChannel = sensorChannel

    def __getitem__(self, idx):

        self.textData = self.csv_input[idx][0]

        self.textData = torch.as_tensor(np.array(self.textData).astype('float'))

        if self.count < len(self.train_index):
            if idx > (self.h5FilelengthList[self.count] - 1):
                self.count += 1
                self.currentFileName = self.h5Files[self.train_index[self.count]]
                self.currentFilePath = os.path.join(self.h5Directory, self.currentFileName.strftime('%d-%b-%Y') + '.h5')

        with h5py.File(self.currentFilePath, 'r') as f:
            self.h5File = f
            image = self.h5File['images'][idx : idx + config['seq_dim'], :, :, 0]
            label = self.h5File['labels'][idx: idx + config['seq_dim']]
            
            # input = np.concatenate((input, self.objectChannel), axis=3)
            # input = np.concatenate((input, self.sensorChannel), axis=3)

        return image, label, self.objectChannel, self.sensorChannel, self.textData

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



