import json
import os

import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
import h5py
import numpy as np
import pandas as pd
from config.config import config
import datetime
from datetime import datetime


class datasetHDF5(Dataset):
    def __init__(self, objectChannel, sensorChannel, h5Files, h5Directory, train_index, dataSequence):
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
                length = np.array(self.h5File['length'].value) - config['seq_dim'] + 1
                self.h5FilelengthList.append(length)
                self.length += length

        objectChannel = np.repeat(objectChannel[np.newaxis, ...], config['seq_dim'], axis=0)
        sensorChannel = np.repeat(sensorChannel[np.newaxis, ...], config['seq_dim'], axis=0)

        self.objectChannel = objectChannel
        self.sensorChannel = sensorChannel

    def __getitem__(self, idx):
        print(idx)
        self.textData = self.csv_input[idx][0]

        self.textData = torch.as_tensor(np.array(self.textData).astype('float'))

        if self.count < len(self.train_index):
            if idx >= sum(self.h5FilelengthList[:self.count + 1]):
                self.count += 1
                self.currentFileName = self.h5Files[self.train_index[self.count]][-1]
                self.currentFilePath = os.path.join(self.h5Directory, self.currentFileName + '.h5')
                # print(self.currentFilePath)
                # print(idx - sum(self.h5FilelengthList[:self.count]))
                # print(sum(self.h5FilelengthList[:self.count]))

        if self.count != 0:
            idx = idx - sum(self.h5FilelengthList[:self.count])
            # print(idx)

        with h5py.File(self.currentFilePath, 'r') as f:
            self.h5File = f
            image = np.array(self.h5File['images'][idx: idx + config['seq_dim'], :, :, 0])
            label = np.array(self.h5File['labels'][idx: idx + config['seq_dim']])

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


class datasetFolder(Dataset):
    def __init__(self, train_index, folders, file_name, objectChannel, sensorChannel, csv_input, transform=None):
        super(datasetFolder, self).__init__()
        self.folders_path = os.path.join(os.getcwd(), '../', 'data', file_name, 'images')
        self.csv_input = csv_input
        self.folders = folders
        self.transform = transform
        self.length = 0
        self.index = 0
        self.count = 0
        self.foldersLength = []
        self.filesinCurrentFolder = []
        self.train_index = train_index
        for index in self.train_index:
            folder = os.path.join(self.folders_path, self.folders[index])
            length = len([name for name in os.listdir(folder, ) if
                          name.endswith('.png') and os.path.isfile(os.path.join(folder, name))]) - config['seq_dim'] + 1
            self.foldersLength.append(length)

        self.totalDataLength = sum(self.foldersLength)
        self.currentfolderIndex = self.train_index[self.index]
        self.currentFolderPath = os.path.join(self.folders_path, self.folders[self.currentfolderIndex])
        self.filesinCurrentFolder = [name for name in os.listdir(self.currentFolderPath) if
                                     name.endswith('.png') and os.path.isfile(
                                         os.path.join(self.currentFolderPath, name))]

        format = "%d-%b-%Y %H:%M:%S"
        time_sorted_list = sorted(self.filesinCurrentFolder,
                                  key=lambda line: datetime.strptime(line.split(".png")[0], format))
        self.filesinCurrentFolder = [os.path.basename(i) for i in time_sorted_list]
        # print(self.filesinCurrentFolder)
        with open(os.path.join(self.currentFolderPath, 'labels.json')) as json_file:
            self.labelsData = json.load(json_file)

        self.objectChannel = cv2.resize(objectChannel, (config['resize_width'], config['resize_height']),
                                        interpolation=cv2.INTER_AREA)

        self.sensorChannel = cv2.resize(sensorChannel, (config['resize_width'], config['resize_height']),
                                        interpolation=cv2.INTER_AREA)

        self.objectChannel = np.expand_dims(self.objectChannel, axis=2)
        self.sensorChannel = np.expand_dims(self.sensorChannel, axis=2)

    def __getitem__(self, id):
        # self.textData = self.csv_input[id][0]
        # self.textData = torch.as_tensor(np.array(self.textData).astype('float'))
        self.labels = []
        self.images = []
        # iterate while id less than sum of current folder
        if id >= sum(self.foldersLength[:self.index + 1]):
            # print(sum(self.foldersLength[:self.index + 1]))
            self.index += 1
            self.currentfolderIndex = self.train_index[self.index]
            self.currentFolderPath = os.path.join(self.folders_path, self.folders[self.currentfolderIndex])
            self.filesinCurrentFolder = [name for name in os.listdir(self.currentFolderPath) if
                                         name.endswith('.png') and os.path.isfile(
                                             os.path.join(self.currentFolderPath, name))]

            with open(os.path.join(self.currentFolderPath, 'labels.json')) as json_file:
                self.labelsData = json.load(json_file)

            full_list = [os.path.join(self.currentFolderPath, i) for i in self.filesinCurrentFolder]
            format = "%d-%b-%Y %H:%M:%S"
            time_sorted_list = sorted(self.filesinCurrentFolder,
                                      key=lambda line: datetime.strptime(line.split(".png")[0], format))
            self.filesinCurrentFolder = [os.path.basename(i) for i in time_sorted_list]

        for i in range(config['seq_dim']):
            self.image = self.filesinCurrentFolder[(id + i) % len(self.filesinCurrentFolder)]
            self.label = self.labelsData[(id + i) % len(self.filesinCurrentFolder)][self.image.split('.png')[0]]
            self.image = cv2.imread(os.path.join(self.currentFolderPath, self.image), 0)
            self.image = np.expand_dims(self.image, axis=2)
            self.image = np.concatenate((self.image, self.objectChannel, self.sensorChannel), axis=2)
            self.labels.append(self.label)
            self.images.append(self.image)

        self.images = np.array(self.images)
        self.labels = np.array(self.labels)

        return self.images, self.labels

    def __len__(self):
        return self.totalDataLength


