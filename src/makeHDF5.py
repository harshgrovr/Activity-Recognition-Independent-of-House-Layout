
import h5py
import json
import cv2
import pandas as pd
import numpy as np
import os
import sys
import datetime
from datetime import datetime, timedelta
from config.config import config


def generateSensorChannel(jsonFile, width=config['image_width'], height=config['image_height'], channel=1):
  # Generate sensors Channel. A single channel for each unique sensor
  outputChannel = []
  sensorChannelDict = {}
  d = jsonFile
  sensorChannel = np.zeros((height, width, channel), dtype=int)
  for sensor in d['sensorLocation']:
    x1, y1, x2, y2 = sensor['location']
    sensorChannel[y1:y2, x1:x2, :] = 2

  sensorChannel = sensorChannel.astype('uint8')
  sensorChannel = cv2.resize(sensorChannel, (config['resize_width'], config['resize_height']),
                             interpolation=cv2.INTER_AREA)
  sensorChannel = sensorChannel.reshape((config['resize_width'], config['resize_height'], 1))

  cv2.imwrite("sensorChannel.jpg", sensorChannel * 255)

  return sensorChannel


def generateObjectChannels(jsonFile, width=config['image_width'], height=config['image_height'], channel=1):
  # Generate Objects Channel. A single channel for each unique object
  outputChannel = []
  objectChannelDict = {}
  d = jsonFile
  objectChannel = np.zeros((height, width, channel), dtype=int)
  for object in d['baseImage']:
    if object['name'] == 'angleWall':
      continue
    x1, y1, x2, y2 = object['location']
    objectChannel[y1:y2, x1:x2, :] = object['id']

  objectChannel = objectChannel.astype('uint8')
  objectChannel = cv2.resize(objectChannel, (config['resize_width'], config['resize_height']),
                             interpolation=cv2.INTER_AREA)
  objectChannel = objectChannel.reshape((config['resize_width'], config['resize_height'], 1))

  # cv2.imwrite("objectChannel.jpg", objectChannel * 255)

  return objectChannel

class Sensor():
  def __init__(self, csv_file_path, json_file_path, root_dir, transform=None):
    df = pd.read_csv(csv_file_path)
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
    self.sensorChannel = generateSensorChannel(self.jsonFile)
    self.ActivityIdList = config['ActivityIdList']

  def getDate(self, start_datetime):
    if not isinstance(start_datetime, datetime):
      start_datetime = datetime.strptime(start_datetime, '%d-%b-%Y %H:%M:%S')
    start_datetime = start_datetime.date()
    return start_datetime

  def generateOffline(self):
    firstdate = self.getDate(self.csvFile.head(1).iloc[0, 0])
    lastDate = self.getDate(self.csvFile.tail(1).iloc[0,0])
    idx = 0
    flag = 0

    # loop for each entry of csv, first to last date
    while firstdate <= lastDate:
      print(firstdate)
      zeros = np.zeros((1, config['resize_height'], config['resize_width'], 1))
      labels = np.array([], dtype=np.long)
      index = 0
      prev_sensor_values = ''

      self.h5Name = os.path.join(self.root_dir, 'h5py', firstdate.strftime('%d-%b-%Y') + '.h5')
      archive = h5py.File(self.h5Name, 'a')
      archive.create_dataset('/images', data=np.empty((1, config['resize_height'], config['resize_width'], 1)), compression="gzip", chunks=True, maxshape=(None, None, None, None))

      # Make a single file for each day
      while firstdate == self.getDate(self.csvFile.iloc[idx, 0]):
        print(idx)
        new_sensor_values = ''
        for val in self.csvFile.iloc[idx, 4:]:
          new_sensor_values = ''.join((new_sensor_values, str(val)))

        if prev_sensor_values != new_sensor_values:
          self.image_name = os.path.join(self.root_dir, 'AnnotatedImage', self.csvFile.iloc[idx, 0])
          self.image = cv2.imread(self.image_name + '.png', 0)
          # self.sensorChannel = generateSensorChannelForTheMinute(self.jsonFile, self.csvFile.iloc[idx, 0], self.csvFile, self.width, self.height, self.channel)

          self.image = self.image

          self.image = np.expand_dims(self.image, axis= 0)
          # get label
          label = self.csvFile.iloc[idx, 2]
          # Get label ID
          label = [x for x in self.ActivityIdList if x["name"] == label]
          self.label = label[0]['id']

          prev_sensor_values = new_sensor_values

        if idx != 0:
          archive['images'].resize((archive["images"].shape[0] + zeros.shape[0]), axis=0)

        archive['images'][-zeros.shape[0]:] = np.expand_dims(self.image, axis=3)

        labels = np.append(labels, self.label)
        idx += 1
        index += 1


        if idx >= len(self.csvFile.index):
          break


      archive.create_dataset('/labels',data=labels)
      archive.create_dataset('/length', data=index)
      if flag == 0:
        archive.create_dataset('/object', data=self.objectChannel)
        archive.create_dataset('/sensor', data=self.sensorChannel)
        flag = 1
      archive.close()

      # Incrementing first date till it reaches to last date
      firstdate += timedelta(days=1)

#
# if __name__ == "__main__":
#     if sys.argv[1] != None:
#         fileName = sys.argv[1].split('.')[0]
#         dataset = Sensor(csv_file_path=fileName + '.csv', json_file_path=fileName + '.json', root_dir=os.getcwd(), transform=None)
#         dataset.generateOffline()
