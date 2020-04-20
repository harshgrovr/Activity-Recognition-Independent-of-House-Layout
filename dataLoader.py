import json
from datetime import datetime
from datetime import timedelta
import csv
import sys
import cv2
import pandas as pd
import numpy as np
import glob
import os
import collections
from itertools import islice
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from dataGeneration import generateObjectChannelsImage, generateObjectChannels, generateSensorChannelForTheMinute


class SensorImageDataset(Dataset):
    def __init__(self, csv_file_path, json_file_path, root_dir, transform = None):
        df = pd.read_csv(csv_file_path)
        self.csvFile = df.iloc[:1000, :]
        with open(json_file_path) as f:
            d = json.load(f)
        self.jsonFile = d
        self.transform = transform
        self.root_dir = root_dir
        self.width = 908
        self.height = 740
        self.channel = 1
        generateObjectChannelsImage(self.jsonFile, self.width, self.height, self.channel)
        self.objectChannel = generateObjectChannels(self.jsonFile, self.width, self.height, self.channel)
        self.ActivityIdList = [
            { "name": "idle", "id": 0 },
            { "name": "leaveHouse", "id": 1 },
            { "name": "useToilet", "id": 2 },
            { "name": "takeShower", "id": 3 },
            { "name": "brushTeeth", "id": 4 },
            { "name": "goToBed", "id": 5 },
            { "name": "getDressed", "id": 6 },
            { "name": "prepareBreakfast", "id": 7 },
            { "name": "prepareDinner", "id": 8 },
            { "name": "getSnack", "id": 9 },
            { "name": "getDrink", "id": 10 },
            { "name": "loadDishwasher", "id": 11 },
            { "name": "unloadDishwasher", "id": 12 },
            { "name": "storeGroceries", "id": 13 },
            { "name": "washDishes", "id": 14 },
            { "name": "answerPhone", "id": 15 },
            { "name": "eatDinner", "id": 16 },
            { "name": "eatBreakfast", "id": 17 }
        ]

    def __len__(self):
        return len(self.csvFile)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        self.image_name = os.path.join(self.root_dir,'AnnotatedImage' , self.csvFile.iloc[idx, 0])
        self.image = cv2.imread(self.image_name + '.png')
        self.sensorChannel = generateSensorChannelForTheMinute(self.jsonFile,self.csvFile.iloc[idx, 0], self.csvFile,self.width, self.height, self.channel)
        self.image = np.concatenate((self.image, self.objectChannel, self.sensorChannel), axis=2)

        if self.transform:
            self.image = self.transform(self.image)
        label = self.csvFile.iloc[idx, 2]
        label = [x for x in self.ActivityIdList if x["name"] == label]
        label = label[0]['id']
        self.image = torch.from_numpy(self.image).float()
        label = torch.from_numpy(np.array(label)).long()
        return self.image, label





