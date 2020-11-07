import shutil

import h5py
import json
import cv2
import pandas as pd
import numpy as np
import os
import sys
import datetime
from datetime import datetime, timedelta
from src.dataGeneration import generateObjectChannels, generateSensorChannel
from config.config import config
class Folder():
  def __init__(self, csv_file_path, json_file_path, root_dir, transform=None):
    df = pd.read_csv(csv_file_path)
    df.drop_duplicates(subset=['start'], inplace=True)
    self.csvFile = df.iloc[:, :]
    with open(json_file_path) as f:
      d = json.load(f)
    self.jsonFile = d
    self.transform = transform
    self.root_dir = root_dir
    self.width = config['image_width']
    self.height = config['image_height']
    self.channel = 1
    self.objectChannel = generateObjectChannels(self.jsonFile, self.width, self.height, self.channel)
    cv2.imwrite(os.path.join(self.root_dir, 'images', 'objectChannel.png'), self.objectChannel)
    self.sensorChannel = generateSensorChannel(self.jsonFile)
    cv2.imwrite(os.path.join(self.root_dir, 'images', 'sensorChannel.png'), self.sensorChannel)
    self.ActivityIdList = config['ActivityIdList']

  def getDate(self, start_datetime):
    if not isinstance(start_datetime, datetime):
      start_datetime = datetime.strptime(start_datetime, '%d-%b-%Y %H:%M:%S')
    start_datetime = start_datetime.date()
    return start_datetime

  def getIDFromClassName(self, train_label):
    ActivityIdList = config['ActivityIdList']
    train_label = [x for x in ActivityIdList if x["name"] == train_label]
    return train_label[0]['id']

  def generateOffline(self):
    firstdate = self.getDate(self.csvFile.head(1).iloc[0, 0])
    lastDate = self.getDate(self.csvFile.tail(1).iloc[0,0])
    idx = 0
    flag = 0

    # loop for each entry of csv, first to last date

    while firstdate <= lastDate:
      labels = np.array([], dtype=np.long)
      print(firstdate)
      index = 0
      self.folderName = os.path.join(self.root_dir, 'images', firstdate.strftime('%d-%b-%Y'))

      if not os.path.exists(self.folderName):
        os.mkdir(self.folderName)


      # Make a single image for each minute
      while firstdate == self.getDate(self.csvFile.iloc[idx, 0]):
        self.image_name = os.path.join(self.root_dir, 'AnnotatedImage', self.csvFile.iloc[idx, 0])

        # activtyID = self.getIDFromClassName(self.csvFile.iloc[idx, 2])

        if os.path.exists(self.image_name+'.png'):

          shutil.copyfile(self.image_name+'.png', os.path.join(self.folderName, self.csvFile.iloc[idx, 0]) + '.png')

          # self.image = cv2.imread(self.image_name + '.png', 0)
          #
          # cv2.imwrite(os.path.join(self.folderName, self.csvFile.iloc[idx, 0]) + '.png', self.image)


        # get label
          label = self.csvFile.iloc[idx, 2]
        # Get label ID
          label = [x for x in self.ActivityIdList if x["name"] == label]
          self.label = {}
          self.label[self.csvFile.iloc[idx, 0]] = label[0]['id']

          labels = np.append(labels, self.label)
        else:
          print(self.image_name + '.png')
        idx += 1
        index += 1

        if idx >= len(self.csvFile.index):
          break

      with open(os.path.join(self.folderName,  'labels.json'), 'w') as file:
        file.write(json.dumps(labels.tolist()))

      # Incrementing first date till it reaches to last date
      firstdate += timedelta(days=1)


#
# if __name__ == "__main__":
#     if sys.argv[1] != None:
#         fileName = sys.argv[1].split('.')[0]
#         dataset = Folder(csv_file_path=fileName + '.csv', json_file_path=fileName + '.json', root_dir=os.getcwd(), transform=None)
#         dataset.generateOffline()
