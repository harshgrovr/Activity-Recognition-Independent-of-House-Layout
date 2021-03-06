import json
from datetime import datetime
from datetime import timedelta
import csv
import sys



import pandas as pd
import numpy as np
import os
from itertools import islice
import glob
from skimage import draw

from src.lstmTraining import getStartAndEndIndex, getUniqueStartIndex
from src.makeFolderStructure import *
# from src.makeHDF5 import *
from config.config import config
# from src.makeFolderStructure import Folder
from src.makeHDF5 import Sensor


def JSON_to_CSV(file):
    with open(file) as f:
        d = json.load(f)

    file = file.split('.')[-2].split('/')[-1]

    activities_label = d['activities']
    sensor_label = d['sensors']
    activities = d['activityData']
    sensor_readings = d['sensorData']

    def get_activity_name(id):
        for i,d in enumerate(activities_label):
            if id == d['id']:
                return d['name']

    def get_sensor_name(id):
        for i,d in enumerate(sensor_label):
            if id == d['id']:
                return d['name']

    def get_time_diff(start, end):
        FMT = '%d-%b-%Y %H:%M:%S'
        if not isinstance(start, datetime):
            start = datetime.strptime(start, FMT)

        if not isinstance(end, datetime):
            end = datetime.strptime(end, FMT)

        tdelta = end - start
        minutes = (tdelta.seconds)/60
        return minutes

    def generateheaderforcsv():
        sensor_list = []
        d = {}
        temp_list = []
        fieldnames = ['start', 'end', 'activity', 'time_of_the_day']

        for sensor in sensor_label:
            d = {}
            d['name'] = sensor['name']
            d['id'] = sensor['id']
            temp_list.append(d)

        newlist = sorted(temp_list, key=lambda k: k['id'])

        for sensor in newlist:
            name = sensor['name'] + '_' + str(sensor['id'])
            sensor_list.append(name)
            fieldnames.append(name)

        return fieldnames, sensor_list

    fieldnames, sensor_list = generateheaderforcsv()

    def normalize_start_and_end_value(start, end):
        FMT = '%d-%b-%Y %H:%M:%S'
        if not isinstance(start, datetime):
            start = datetime.strptime(start, "%d-%b-%Y %H:%M:%S")
        start = start.replace(second=0)

        if not isinstance(end, datetime):
            end = datetime.strptime(end, "%d-%b-%Y %H:%M:%S")

        end = end.replace(second=59)
        return start, end

    def get_sensor_id_for_the_Minute(minutes, sensorIndex = 0):
        sensor_list = []
        sensorIndexs = []
        if  not (isinstance(minutes, datetime)):
            minutes = datetime.strptime(minutes, "%d-%b-%Y %H:%M:%S")
        for sensor in islice(sensor_readings, sensorIndex, None):
            if not (isinstance(sensor['start'], datetime)):
                sensor['start'] =  datetime.strptime(sensor['start'], "%d-%b-%Y %H:%M:%S")
            if  not (isinstance(sensor['end'], datetime)):
                sensor['end'] =  datetime.strptime(sensor['end'], "%d-%b-%Y %H:%M:%S")
            sensor['start'] , sensor['end'] = normalize_start_and_end_value(sensor['start'], sensor['end'])

            if sensor['start'] <= minutes <= sensor['end']:
                d = {}
                d['id'] = sensor['id']
                d['value'] = sensor['value']
                sensor_list.append(d)
                sensorIndexs.append(sensorIndex)


            # return as soon as you find a start date which is in the future, and yet to occur.
            if sensor['start'] > minutes:
                break

            # Keep the index and start from this index, rather always starting from 0
            sensorIndex += 1

        if len(sensorIndexs) != 0:
            return sensor_list, min(sensorIndexs)
        else:
            return sensor_list, sensorIndex

    def get_activity_id_for_the_Minute(minutes, activityIndex, d={}):
        activity_list = []
        activityIndexs = []
        if  not (isinstance(minutes, datetime)):
            minutes = datetime.strptime(minutes, "%d-%b-%Y %H:%M:%S")

        # select activity that covers most of that minute unless it is supressing some other other activity,
        # in that case select least one.
        for activity in islice(activities, activityIndex, None):
            if not (isinstance(activity['start'], datetime)):
                activity['start'] =  datetime.strptime(activity['start'], "%d-%b-%Y %H:%M:%S")

            if not (isinstance(activity['end'], datetime)):
                activity['end'] =  datetime.strptime(activity['end'], "%d-%b-%Y %H:%M:%S")

            start, end = normalize_start_and_end_value(activity['start'] ,activity['end'])

            # activity['start'] = start
            # activity['end'] = end

            if start <= minutes <= end:
                d = {}
                d['start'] = activity['start']
                d['end'] = activity['end']
                d['id'] = activity['id']
                activity_list.append(d)
                activityIndexs.append(activityIndex)

            # return as soon as you find a start date which is in the future, and yet to occur.
            if start > minutes:
                break

            # Keep the index and start from this index, rather always starting from 0
            activityIndex += 1


        if(len(activity_list) == 1):
            return activity_list[0]['id'], min(activityIndexs)

        if (len(activity_list) >= 2):
            # select the one with most time for that minute
            time = []
            max_index = 0

            for j in range(len(activity_list)):
                t = get_time_diff(activity_list[j]['start'], activity_list[j]['end'])
                time.append(t)
                min_index = time.index(min(time))

            return activity_list[min_index]['id'], min(activityIndexs)

        return 404, activityIndex

    def change_datetime_format(date, to = '%d-%b-%Y %H:%M:%S', fromm = '%Y-%m-%d %H:%M:%S'):
        if not isinstance(date, datetime):
            date = datetime.strptime(date, fromm)
        date =  date.strftime(to)

        return date

    def minutediff(activity_start, activity_end):
        daysDiff = (activity_end - activity_start).days
        daysDiff = daysDiff * 24 * 60
        minuteDiff = ((activity_end - activity_start).seconds)/60
        return minuteDiff + daysDiff

    def time_of_the_day(time):
        if not isinstance(time, datetime):
            time = datetime.strptime(time, '%d-%b-%Y %H:%M:%S')

        seconds = timedelta(hours=time.hour,minutes=time.minute,seconds=time.second).total_seconds()
        return (seconds)/ (24*60*60)

    # Pick a start, end pair from activity loop and corresponding to that pick up sensor id from above dict for that
    # range
    csv_dictionary = {}
    csv_file_name = file+'.csv'
    with open(csv_file_name, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        activity_start = activities[0]['start']
        activity_end = (activities[len(activities) - 1]['end'])

        activity_start , activity_end = normalize_start_and_end_value(activity_start ,activity_end)

        if not isinstance(activity_start, datetime):
            activity_start = datetime.strptime( activity_start, "%d-%b-%Y %H:%M:%S")
        if not isinstance(activity_end, datetime):
            activity_end = datetime.strptime( activity_end, "%d-%b-%Y %H:%M:%S")


        minutes = minutediff(activity_start, activity_end)
        minute = activity_start
        sensorIndex = 0
        activityIndex = 0
        for i in range(int(minutes+1)):
            print(i+1, 'line written to csv ')
            # create an entry in csv file
            activity_id, activityIndex = get_activity_id_for_the_Minute(minute, activityIndex)
            activity_name = get_activity_name(activity_id)
            # write that sensor_value_duplicate to csv
            csv_dictionary = {}
            csv_dictionary['start'] = change_datetime_format(minute)
            csv_dictionary['end'] = change_datetime_format(minute + timedelta(seconds=59))
            csv_dictionary['activity'] = activity_name
            csv_dictionary['time_of_the_day'] = time_of_the_day(minute)


            for name in sensor_list:
                csv_dictionary[name] = 0

            activatedSensorDict, sensorIndex = get_sensor_id_for_the_Minute(minute, sensorIndex)

            for sensor_dict in activatedSensorDict:
                name = get_sensor_name(sensor_dict['id']) + '_' + str(sensor_dict['id'])
                csv_dictionary[name] = sensor_dict['value']

            writer.writerow(csv_dictionary)
            minute = minute + timedelta(seconds=60)
        print('DONE !')

def sensorLocation(jsonFile, activatedSensorNames):
    sensorLocationList = []
    for activateSensorName in activatedSensorNames:
        # get location for this activatedSensors
        for sensorLocationDict in jsonFile['sensor_location']:
            if sensorLocationDict['name'] == activateSensorName:
                d = {}
                d['location'] = sensorLocationDict['location']
                d['angle'] = sensorLocationDict['angle']
                d['name'] = sensorLocationDict['name']
                sensorLocationList.append(d)

    return sensorLocationList
def image_resize(h=740, w=908, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None

    # if both the width and height are None, then return the
    # original image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    # resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return dim

def image_size_in_16_by_16_image(curr_width, curr_height,curr_width_1, curr_height_1, prev_height, prev_width, final_width, final_height, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None

    # calculate the ratio of the width and construct the
    # dimensions
    rw = final_width / float(prev_width)
    # calculate the ratio of the width and construct the
    # dimensions
    rh = final_height / float(prev_height)


    print(round(rw * curr_width) - 1, round(rh * curr_height) - 1, round(rw * curr_width_1) - 1, round(rh * curr_height_1) - 1)

    # resize the image
    # resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image

def annotateImage(filename, input_dir, minutesToGenrate = 100):
    font = cv2.FONT_HERSHEY_SIMPLEX
    topLeftCornerOfText = (500-294, 186)
    fontScale = 0.7
    fontColor = (0)
    lineType = 2
    jsonFile= os.path.join(input_dir, file_name+'.json')
    w,h = image_resize(width=120)
    w, h = 24, 16
    with open(jsonFile) as f:
        jsonFile = json.load(f)
    csv_name = os.path.join(input_dir, file_name+'.csv')
    df = pd.read_csv(csv_name)
    df = df[0:minutesToGenrate]
    annoatedImageDir = os.path.join(input_dir, 'AnnotatedImage')
    if not os.path.exists(annoatedImageDir):
        os.makedirs(annoatedImageDir)

    print('total files to generate: ', minutesToGenrate)
    uniqueIndex = getUniqueStartIndex(df)
    df_temp = pd.read_csv(csv_name)
    end = -1
    i = -1
    count = 0
    sensorValue = []
    previousActivatedSensorValue = {}
    activatedSensorNames = df.columns[df.iloc[0, :] == 1]
    for name in activatedSensorNames:
        previousActivatedSensorValue[name] = -1

    for index, row in df.iterrows():
        # skip idle activities
        # print(df.iloc[index, df.columns.get_loc('activity')])
        if(df.iloc[index, df.columns.get_loc('activity')] == 'Idle'):
            k=1
        if end < index:
            i += 1
            count = 0
            start, end = getStartAndEndIndex(df, uniqueIndex[i])
            # x = np.linspace(end - start, 0, end - start + 1)
            # z = 1 / (1 + np.exp(-x))
            # sensorValue = 22 * (1-z)
            # sensorValue = sensorValue[::-1]



        print('generating image file:',index)
        temp_image = cv2.imread(os.path.join(input_dir, filename+'.png'), 0)
        temp_image = np.array(temp_image)
        temp_image = cv2.resize(temp_image, (w, h),
                                interpolation=cv2.INTER_AREA)

        activatedSensorNames = df.columns[df.iloc[index, :] == 1].tolist()
        temp_activatedSensorNames = activatedSensorNames.copy()

        activatedSensorNames = temp_activatedSensorNames.copy()
        circleDict = sensorLocation(jsonFile, activatedSensorNames)


        pre_w = 663
        pre_h =446
        rw =1
        rh =1
        # rw = w/pre_w
        # rh =  h/pre_h


        if circleDict != None:
            for dict in circleDict:
                if len(dict['location']) == 4:
                    x1, y1, x2, y2 = dict['location']
                    x1 =int((x1+x2)/2)
                    y1 = int((y1 + y2)/2)
                elif len(dict['location']) == 3:
                    x1, y1, _ = dict['location']
                else:
                    x1, y1 = dict['location']
                    if dict['name'][0] == 'D' or dict['name'][0] == 'd':
                        sensorValue = 110
                        temp_image[y1, x1] = sensorValue
                    else:
                        sensorValue = 60
                        temp_image[y1, x1] = sensorValue



                # cv2.circle(temp_image, (round(x1),round(y1)), 6, sensorValue[count], -1, lineType=cv2.LINE_AA)

                # cv2.circle(temp_image, (round(x1), round(y1)), 6, sensorValue, -1, lineType=cv2.LINE_AA)

        # nonActivatedSensorNames = df.columns[df.iloc[index, :] == 0].tolist()
        # circleDict = sensorLocation(jsonFile, nonActivatedSensorNames)
        #
        # if circleDict != None:
        #     for dict in circleDict:
        #         if len(dict['location']) == 4:
        #             x1, y1, x2, y2 = dict['location']
        #             x1 =int((x1+x2)/2)
        #             y1 = int((y1 + y2)/2)
        #         elif len(dict['location']) == 3:
        #             x1, y1, _ = dict['location']
        #         else:
        #             x1, y1 = dict['location']
        #
        #         x1, y1 = round(rw * x1),round(rh * y1)
        #         cv2.circle(temp_image, (round(x1),round(y1)), 6, sensorValue[count], 1, lineType=cv2.LINE_AA)


        # cv2.putText(temp_image, df.iloc[index, df.columns.get_loc('activity')],
        #             topLeftCornerOfText,
        #             font,
        #             fontScale,
        #             fontColor,
        #             lineType)


        cv2.imwrite(os.path.join(annoatedImageDir, str(df_temp['start'][index]) + '.png'), temp_image)

        count += 1
        if index == minutesToGenrate:
            break


def makeVideo(annoatedImageDir, fps = 10):
    img_array = []
    format = "%d-%b-%Y %H:%M:%S"
    currentFolderPath = annoatedImageDir
    filesinCurrentFolder = [name for name in os.listdir(currentFolderPath) if
                                 name.endswith('.png') and os.path.isfile(
                                     os.path.join(currentFolderPath, name))]

    time_sorted_list = sorted(filesinCurrentFolder,
                              key=lambda line: datetime.strptime(line.split(".png")[0], format))

    size = 1
    for filename in time_sorted_list:
        img = cv2.imread(os.path.join(annoatedImageDir, filename))
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)


    out = cv2.VideoWriter(os.path.join(annoatedImageDir, '../', 'project.avi'), cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

def generateSensorLocationTemplateBasedUponJSON(file):
    with open(file) as f:
        jsonFile = json.load(f)
    d = {"sensorLocation": []}
    for sensorDict in jsonFile['sensors']:
        dict = {}
        dict['name'] = sensorDict['name']
        dict['location'] = []
        dict['id'] = sensorDict['id']
        dict['angle'] = 0
        d['sensorLocation'].append(dict)
    print(json.dumps(d))


def generateImagewithAllAnnoations(input_dir, file_name):
    font = cv2.FONT_HERSHEY_SIMPLEX
    image = cv2.imread(os.path.join(input_dir, file_name+'.png'))
    with open(os.path.join(input_dir, file_name+'.json')) as f:
        jsonFile = json.load(f)
    for dict in jsonFile['sensorLocation']:
        location = dict['location']
        center_x = (location[0] + location[2])/2
        center_y = (location[1] + location[3])/2
        width_x = abs(location[0] - location[2])
        width_y = abs(location[1] - location[3])
        angle = dict['angle']
        if len(location) == 4:
            box = cv2.boxPoints(((center_x,center_y), (width_x, width_y), angle))
            box = np.int0(box)
            if dict['name'] == 'Bed':
                color = (0)
            else:
                color = (0)
            cv2.drawContours(image, [box], 0, color, 2)
            if len(dict['name']) > 20:
                name = dict['name'][:20] + '...'
            else:
                name = dict['name']
            cv2.putText(image, name, (int(center_x), int(center_y)), font, 0.4, (255, 0, 0), 1)
        cv2.imwrite(os.path.join(input_dir, 'AnnotatedImage.png'), image)


def sortDictionary(file, key):
    with open(file) as f:
        d = json.load(f)

    d[key] = sorted(d[key], key=lambda item: datetime.strptime(item['start'], '%d-%b-%Y %H:%M:%S'))
    with open(file, 'w') as outfile:
        json.dump(d, outfile)

def create_blank(width, height, rgb_color=(255, 255, 255)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)
    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color
    return image




def generateBaseImage(input_dir, file_name, width1 = 0, height1 = 0):
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Create new blank image
    white = (255, 255, 255)

    image = create_blank(width1, height1, white)

    labels = {"kitchen": (334,155), "Dining": (265,295) , "Living": (315,465), "Front Door": (522,580), "BedRoom_1": (748,498), "closet": (838,515), "Bathroom_2": (833,411), "BedRoom": (850,301), "Office": (766,153), "Garage": (653,106), "Bathroom": (568,159)}
    # 233, 139
    # for key in list(labels.keys()):
    #     x, y = labels[key]
    #     x-=264
    #     y-=95
    #     cv2.putText(image, key, (int(x), int(y)), font, 0.5, (255, 0, 0), 1)



    # with open(os.path.join(input_dir, file_name+'.json')) as f:
    #     jsonFile = json.load(f)
    # for dict in jsonFile['baseImage']:
    #     location = dict['location']
    #
    #     # Remove this  block for other datasets #
    #
    #     # Remove this  block for other datasets #
    #
    #     center_x = (location[0] + location[2]) / 2
    #     center_y = (location[1] + location[3]) / 2
    #     width_x = abs(location[0] - location[2])
    #     width_y = abs(location[1] - location[3])
    #     angle = dict['angle']
    #     if len(location) == 4:
    #         if dict['name'] == 'angleWall':
    #             box = cv2.boxPoints(((center_x, center_y), (dict['height'], dict['width']), angle))
    #             box = np.int0(box)
    #             color = (0, 0, 255)
    #             cv2.drawContours(image, [box], 0, color, 2)
    #         else:
    #             box = cv2.boxPoints(((center_x, center_y), (width_x, width_y), angle))
    #             box = np.int0(box)
    #             color = (0, 0, 255)
    #             cv2.drawContours(image, [box], 0, color, 2)
    #             # cv2.putText(image, dict['name'], (int(center_x), int(center_y)), font, 0.4, (255, 0, 0), 1)
    #             # if len(dict['name']) > 20:
    #             #     name = dict['name'][:20] + '...'
    #             # else:
    #             #     name = dict['name']
    #             # cv2.putText(image, name, (int(center_x), int(center_y)), font, 0.4, (255, 0, 0), 1)
    # # w, h = image_resize(width=120)
    # # image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # image[image== 76] = 0
    print(np.unique(image))
    filename =  file_name +'.png'

    cv2.imwrite(os.path.join(input_dir, filename), image)

def generateSensorChannel(jsonFile, width = config['image_width'], height = config['image_height'],channel = 1):
    w, h = image_resize(width=24)
    w, h = 24, 16
    # Generate sensors Channel. A single channel for each unique sensor
    outputChannel = []
    sensorChannelDict = {}
    d = jsonFile
    sensorChannel = 255 * np.ones((h, w, channel), dtype=int)
    for sensor in d['sensor_location']:
        pre_w = config['image_width']
        pre_h = config['image_height']
        rw = w / pre_w
        rh = h / pre_h
        x1, y1 = sensor['location']
        if sensor['name'][0] == 'D' or sensor['name'][0] == 'd':
            sensorValue = 110
            sensorChannel[y1, x1, :]  = sensorValue
        else:
            sensorValue = 60
            sensorChannel[y1, x1, :]  = sensorValue

    # sensorChannel = sensorChannel.astype('uint8')
    # sensorChannel = cv2.resize(sensorChannel, (config['resize_width'], config['resize_height']),
    #                            interpolation=cv2.INTER_AREA)
    # sensorChannel = sensorChannel.reshape((config['resize_width'], config['resize_height'], 1))

    cv2.imwrite("sensorChannel.jpg", sensorChannel)

    return sensorChannel

def generateObjectChannels(jsonFile, width = config['image_width'], height = config['image_height'],channel = 1):
    w, h = image_resize(width=24)
    w, h = 24, 16
    # Generate Objects Channel. A single channel for each unique object
    outputChannel = []
    objectChannelDict = {}
    d = jsonFile
    objectChannel =255 * np.ones((h, w, channel), dtype=int)
    for object in d['rooms_location']:
        if object['name'] == 'angleWall':
            continue

        pre_w = config['image_width']
        pre_h = config['image_height']
        rw = w / pre_w
        rh = h / pre_h

        x1, y1, x2, y2 = object['location']
        objectChannel[y1:y2, x1:x2, :] = object['id']
        print(x1, y1, x2, y2, ' ---> ', object['id'])


    # objectChannel = objectChannel.astype('uint8')
    # objectChannel = cv2.resize(objectChannel, (config['resize_width'], config['resize_height']), interpolation=cv2.INTER_AREA)
    # objectChannel = np.expand_dims(objectChannel, axis= 2)

    cv2.imwrite("objectChannel.jpg", objectChannel)

    return objectChannel


def generateSensorChannelForTheMinute(jsonFile, minute='24-Jul-2009 16:46:00', csvFile = '', width = config['image_width'], height = config['image_height'], channel = 1):
    # Generate Objects Channel. A single channel for each unique object
    df = csvFile[csvFile['start'] == minute]
    activatedSensorNames = df.columns[df.iloc[0, :] == 1]
    rectangleDict = sensorLocation(jsonFile, activatedSensorNames)
    sensorChannel = np.zeros((height, width, channel), dtype=int)
    for dicts in rectangleDict:
        x1,y1,x2,y2 = dicts['location']
        sensorChannel[y1:y2, x1:x2, :] = 1
    sensorChannel = sensorChannel.astype('uint8')
    sensorChannel = cv2.resize(sensorChannel, (config['resize_width'], config['resize_height']), interpolation=cv2.INTER_AREA)
    sensorChannel = sensorChannel.reshape((config['resize_width'], config['resize_height'], 1))

    return sensorChannel


def generateSeqData(input_dir, minutesToGenrate = 10000):
    currentFolderPath = os.path.join(input_dir, 'AnnotatedImage')

    currentFolderPath_1 = os.path.join(input_dir, 'Sequence_AnnotatedImage')


    filesinCurrentFolder = [name for name in os.listdir(currentFolderPath) if
                            name.endswith('.png') and os.path.isfile(
                                os.path.join(currentFolderPath, name))]


    filesinCurrentFolder = filesinCurrentFolder [0:minutesToGenrate]

    format = "%d-%b-%Y %H:%M:%S"

    time_sorted_list = sorted(filesinCurrentFolder,
                              key=lambda line: datetime.strptime(line.split(".png")[0], format))

    filesinCurrentFolder = [os.path.basename(i) for i in time_sorted_list]


    image_array = []
    for id in range(len(filesinCurrentFolder) - config['seq_dim'] + 1):
        print('file is: ', filesinCurrentFolder[id])
        temp_array = []
        image = filesinCurrentFolder[id: id + config['seq_dim']]
        for imgName in image:
            img = cv2.imread(os.path.join(currentFolderPath, imgName), 0)
            img = np.expand_dims(img, axis=0)
            temp_array.append(img)

        temp_array = np.concatenate((temp_array), axis=0)
        temp_array = np.min(temp_array, axis=0)
        cv2.imwrite(os.path.join(currentFolderPath_1, filesinCurrentFolder[id]), temp_array)



if __name__ == "__main__":
    if sys.argv[1] != None:
        file_name = sys.argv[1].split('.json')[0]
        input_dir = os.path.join(os.getcwd(), '../', 'data',  file_name)
        # input_dir = os.path.join(os.getcwd(), '../../GitRepo/video-classification', 'data', file_name)

        csv_file_path = os.path.join(input_dir, file_name + '.csv')
        json_file_path = os.path.join(input_dir, file_name + '.json')
        #
        # csv_length = pd.read_csv(csv_file_path).shape[0]
        # print(csv_length)
        # sort the two Sensor and Activity data of JSON and replace the old JSON file
        # sortDictionary(sys.argv[1], 'sensorData')
        # sortDictionary(sys.argv[1], 'activityData')


        # Convert JSON to CSV
        JSON_to_CSV(json_file_path)

        # # Generate Base Image
        # generateBaseImage(input_dir, file_name, width1=24, height1= 16)

        # # Generate an Image named Annoation.png , showing all the sensors and objects
        # generateImagewithAllAnnoations(input_dir, file_name)

        # csv_length = 8000

        # with open(json_file_path) as jsonFile:
        #     f = json.load(jsonFile)
        #
        # generateObjectChannels(f)
        # generateSensorChannel(f)

        # image_size_in_16_by_16_image(450, 134, 661, 263, prev_height=446, prev_width=663, final_width=24, final_height=16)


        # # Make a folder and save all the annotated Image per minute bases24-Jul-2009 20:21:00.png

        # annotateImage(file_name,input_dir, minutesToGenrate = 100000)


        # # Generate a video on above generated Image
        # makeVideo(os.path.join(input_dir, 'AnnotatedImage'), fps=10)


        # Generate H5 file for the images per day basic
        # dataset = Sensor(csv_file_path, json_file_path, root_dir=input_dir,
        #                  transform=None)

        # # Generate Seq Data
        # generateSeqData(input_dir, minutesToGenrate = 143132)

        # input("Press Enter to continue...")
        #
        #
        # dataset = Folder(csv_file_path,json_file_path , root_dir=input_dir,
        #                  transform=None)
        #
        # dataset.generateOffline()
