
import numpy as np
import h5py
import json
import cv2
import pandas as pd
import numpy as np
import os
import sys
import datetime
from datetime import datetime, timedelta
from dataGeneration import generateObjectChannels, generateSensorChannelForTheMinute
import torch

class Sensor():
  def __init__(self, csv_file_path, json_file_path, root_dir, transform=None):
    df = pd.read_csv(csv_file_path)
    self.csvFile = df.iloc[:2, :]
    with open(json_file_path) as f:
      d = json.load(f)
    self.jsonFile = d
    self.transform = transform
    self.root_dir = root_dir
    self.width = 908
    self.height = 740
    self.channel = 1
    self.objectChannel = generateObjectChannels(self.jsonFile, self.width, self.height, self.channel)
    self.ActivityIdList = [
      {"name": "idle", "id": 0},
      {"name": "leaveHouse", "id": 1},
      {"name": "useToilet", "id": 2},
      {"name": "takeShower", "id": 3},
      {"name": "brushTeeth", "id": 4},
      {"name": "goToBed", "id": 5},
      {"name": "getDressed", "id": 6},
      {"name": "prepareBreakfast", "id": 7},
      {"name": "prepareDinner", "id": 8},
      {"name": "getSnack", "id": 9},
      {"name": "getDrink", "id": 10},
      {"name": "loadDishwasher", "id": 11},
      {"name": "unloadDishwasher", "id": 12},
      {"name": "storeGroceries", "id": 13},
      {"name": "washDishes", "id": 14},
      {"name": "answerPhone", "id": 15},
      {"name": "eatDinner", "id": 16},
      {"name": "eatBreakfast", "id": 17}
    ]

  def getDate(self, start_datetime):
    if not isinstance(start_datetime, datetime):
      start_datetime = datetime.strptime(start_datetime, '%d-%b-%Y %H:%M:%S')
    start_datetime = start_datetime.date()
    return start_datetime

  def generateOffline(self):

    firstdate = self.getDate(self.csvFile.head(1).iloc[0, 0])
    lastDate = self.getDate(self.csvFile.tail(1).iloc[0,0])

    idx = -1
    while firstdate <= lastDate:
      images = np.zeros((1, 740, 908, 22))
      labels = np.array([], dtype=np.long)
      idx += 1
      while firstdate == self.getDate(self.csvFile.iloc[idx, 0]):
        self.image_name = os.path.join(self.root_dir, 'AnnotatedImage', self.csvFile.iloc[idx, 0])
        self.image = cv2.imread(self.image_name + '.png')

        self.sensorChannel = generateSensorChannelForTheMinute(self.jsonFile, self.csvFile.iloc[idx, 0], self.csvFile, self.width, self.height, self.channel)

        self.image = np.concatenate((self.image, self.objectChannel, self.sensorChannel), axis=2)
        self.image = np.expand_dims(self.image, axis= 0)
        if self.transform:
            self.image = self.transform(self.image)

        images = np.append(images, self.image, axis = 0)
        # get label
        label = self.csvFile.iloc[idx, 2]
        # Get label ID
        label = [x for x in self.ActivityIdList if x["name"] == label]
        label = label[0]['id']
        labels = np.append(labels, label)
        idx += 1
        if idx >= len(self.csvFile.index):
          break
      self.h5Name = os.path.join(self.root_dir, 'h5py', firstdate.strftime('%d-%b-%Y') + '.h5')
      archive = h5py.File(self.h5Name, 'w')
      archive.create_dataset('/images', data=images[1:,...], compression='gzip',compression_opts=6)
      archive.create_dataset('/labels',data=labels, compression='gzip', compression_opts=6)
      archive.close()
      firstdate += timedelta(days=1)


if __name__ == "__main__":
    if sys.argv[1] != None:
        fileName = sys.argv[1].split('.')[0]
        dataset = Sensor(csv_file_path=fileName + '.csv', json_file_path=fileName + '.json', root_dir=os.getcwd(), transform=None)
        dataset.generateOffline()
