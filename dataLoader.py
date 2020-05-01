import torch
from torch.utils.data import Dataset
import h5py
import pandas as pd
from makeHDF5 import Sensor
import numpy as np
import datetime
from datetime import datetime, timedelta
import os


class datasetHDF5(Dataset):
    def __init__(self, csvFileName, h5Directory):
        super(datasetHDF5, self).__init__()
        self.df = pd.read_csv(csvFileName)
        self.h5Directory = h5Directory
        self.startDate = self.df.iloc[0, 0]
        self.startDate = datetime.strptime(self.startDate, '%d-%b-%Y %H:%M:%S')
        self.h5_path = os.path.join(self.h5Directory, self.startDate.strftime('%d-%b-%Y') + '.h5')
        self.h5File = h5py.File(self.h5_path, 'r')
        self.counter = 0

    def __getitem__(self, idx):
        try:
            input = self.h5File['images'][self.counter, :, :, :]
            label = self.h5File['labels'][self.counter]
            input = torch.as_tensor(np.array(input).astype('float'))
            label = torch.as_tensor(np.array(label).astype('long'))
            self.counter += 1
            return input, label
        except:
            # Closing the pre opened file
            if isinstance(self.h5File, h5py.File):  # Just HDF5 files
                try:
                    self.h5File.close()
                except:
                    pass  # Was already closed

            self.counter = 1
            print('start date is: ', self.startDate)
            self.startDate = self.startDate + timedelta(days=1)
            self.h5_path = os.path.join(self.h5Directory, self.startDate.strftime('%d-%b-%Y') + '.h5')
            self.h5File = h5py.File(self.h5_path, 'r')
            input = self.h5File['images'][0, :, :, :]
            label = self.h5File['labels'][0]
            input = torch.as_tensor(np.array(input).astype('float'))
            label = torch.as_tensor(np.array(label).astype('long'))
            return input, label

    def __len__(self):
        # Equal to number of rows in CSV
        self.length = self.df.shape[0]





