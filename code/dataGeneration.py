import json
from datetime import datetime
from datetime import timedelta
import csv
import sys
import cv2
import pandas as pd
import numpy as np
import os
from itertools import islice
import glob


def JSON_to_CSV(file):
    with open(file) as f:
        d = json.load(f)

    file = file.split('.')[0]

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
        for sensorLocationDict in jsonFile['sensorLocation']:
            if sensorLocationDict['name'] + '_' + str(sensorLocationDict['id']) == activateSensorName:
                d = {}
                d['location'] = sensorLocationDict['location']
                d['angle'] = sensorLocationDict['angle']
                sensorLocationList.append(d)
    return sensorLocationList

def annotateImage(file, imagePath= '', minutesToGenrate = 100):
    with open(file) as f:
        jsonFile = json.load(f)
    file = file.split('.')[0]
    csv_name = file+ '.csv'
    df = pd.read_csv(csv_name)
    annoatedImageDir =''

    if not os.path.exists(os.path.join(os.getcwd(), 'AnnotatedImage')):
        os.makedirs(os.path.join(os.getcwd(), 'AnnotatedImage'))
        annoatedImageDir = os.path.join(os.getcwd(), 'AnnotatedImage')
    else:
        annoatedImageDir = os.path.join(os.getcwd(), 'AnnotatedImage')
        print('total files to generate: ', minutesToGenrate)
    for index, row in df.iterrows():
        print('generating image file:',index)
        image = cv2.imread(imagePath)
        activatedSensorNames = df.columns[df.iloc[index, :] == 1]
        rectangleDict = sensorLocation(jsonFile, activatedSensorNames)
        if rectangleDict != None:
            for dict in rectangleDict:
                if len(dict['location']) == 4:
                    location = dict['location']
                    center_x = (location[0] + location[2]) / 2
                    center_y = (location[1] + location[3]) / 2
                    width_x = abs(location[0] - location[2])
                    width_y = abs(location[1] - location[3])
                    angle = dict['angle']
                    box = cv2.boxPoints(((center_x, center_y), (width_x, width_y), angle))
                    box = np.int0(box)
                    cv2.drawContours(image, [box], 0, (0, 255, 255), 2)
                    # cv2.rectangle(image, (location[0], location[1]), (location[2], location[3]), (0, 255, 255), -1)
                    cv2.imwrite(os.path.join(annoatedImageDir, row['start']+'.png'), image)
        if index == minutesToGenrate:
            break


def makeVideo(annoatedImageDir, fps = 5):
    img_array = []
    for filename in glob.glob(annoatedImageDir+'/*.png'):
        img = cv2.imread(filename)
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


def generateImagewithAllAnnoations(file, imagePath= ''):
    font = cv2.FONT_HERSHEY_SIMPLEX
    image = cv2.imread(imagePath)
    with open(file) as f:
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
                color = (255, 0, 0)
            else:
                color = (0, 255, 255)
            cv2.drawContours(image, [box], 0, color, 2)
            if len(dict['name']) > 20:
                name = dict['name'][:20] + '...'
            else:
                name = dict['name']
            cv2.putText(image, name, (int(center_x), int(center_y)), font, 0.4, (255, 0, 0), 1)
            cv2.imwrite(os.path.join(os.getcwd(), 'AnnoattedImage.png'), image)


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


def generateBaseImage(file, imagePath= '', width1 = 0, height1 = 0):
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Create new blank image
    white = (255, 255, 255)
    image = create_blank(width1, height1, white)

    with open(file) as f:
        jsonFile = json.load(f)
    for dict in jsonFile['baseImage']:
        location = dict['location']
        center_x = (location[0] + location[2]) / 2
        center_y = (location[1] + location[3]) / 2
        width_x = abs(location[0] - location[2])
        width_y = abs(location[1] - location[3])
        angle = dict['angle']
        if len(location) == 4:
            if dict['name'] == 'angleWall':
                box = cv2.boxPoints(((center_x, center_y), (dict['height'], dict['width']), angle))
                box = np.int0(box)
                color = (0, 0, 255)
                cv2.drawContours(image, [box], 0, color, 2)
            else:
                box = cv2.boxPoints(((center_x, center_y), (width_x, width_y), angle))
                box = np.int0(box)
                color = (0, 0, 255)
                cv2.drawContours(image, [box], 0, color, 2)
                # cv2.putText(image, dict['name'], (int(center_x), int(center_y)), font, 0.4, (255, 0, 0), 1)
                # if len(dict['name']) > 20:
                #     name = dict['name'][:20] + '...'
                # else:
                #     name = dict['name']
                # cv2.putText(image, name, (int(center_x), int(center_y)), font, 0.4, (255, 0, 0), 1)
    cv2.imwrite(os.path.join(os.getcwd(), 'houseB.png'), image)

def generateObjectChannels(jsonFile, width = 908, height = 740,channel = 1):
    # Generate Objects Channel. A single channel for each unique object
    outputChannel = []
    objectChannelDict = {}
    d= jsonFile
    for object in d['baseImage']:
        objectChannel = np.zeros((height, width, channel), dtype=int)
        if object['name'] == 'angleWall':
            continue
        x1, y1, x2, y2 = object['location']
        if object['name'] in objectChannelDict.keys():
            objectChannelDict[object['name']][x1:y2, y1:y2, :] = 1
        else:
            objectChannel[y1:y2, x1:x2, :] = 1
            objectChannelDict[object['name']] = objectChannel
    for k in objectChannelDict:
        outputChannel.append(objectChannelDict[k])
    outputChannel = np.concatenate(outputChannel, axis=2)
    return outputChannel


def generateObjectChannelsImage(jsonFile, width = 908, height = 740,channel = 1):
    # Generate Objects Channel. A single channel for each unique object
    outputChannel = []
    objectChannelDict = {}
    d= jsonFile
    objectChannel = np.zeros((height, width, channel), dtype=int)
    for object in d['baseImage']:
        # if object['name'] == 'angleWall':
        #     continue
        x1, y1, x2, y2 = object['location']
        if object['name'] in objectChannelDict.keys():
            objectChannelDict[object['name']][y1:y2, x1:x2, :] = 1
        else:
            objectChannel[y1:y2, x1:x2, :] = 1
            cv2.imwrite("color_img.jpg", objectChannel * 255)
    cv2.imwrite("color_img.jpg", objectChannel * 255)

def generateSensorChannelForTheMinute(jsonFile, minute='24-Jul-2009 16:46:00', csvFile = '', width = 908, height = 740, channel = 1):
    # Generate Objects Channel. A single channel for each unique object
    df = csvFile[csvFile['start'] == minute]
    activatedSensorNames = df.columns[df.iloc[0, :] == 1]
    rectangleDict = sensorLocation(jsonFile, activatedSensorNames)
    sensorChannel = np.zeros((height, width, channel), dtype=int)
    for dicts in rectangleDict:
        x1,y1,x2,y2 = dicts['location']
        sensorChannel[y1:y2, x1:x2, :] = 1
    return sensorChannel


if __name__ == "__main__":
    if sys.argv[1] != None:
        file_name = sys.argv[1].split('.')[0]

        # sort the two Sensor and Activity data of JSON and replace the old JSON file
        sortDictionary(sys.argv[1], 'sensorData')
        sortDictionary(sys.argv[1], 'activityData')

        # Convert JSON to CSV
        # JSON_to_CSV(sys.argv[1])

        # # Generate Dictionary Template for Annotations
        # generateBaseImage(sys.argv[1], file_name + '.png', width1=908, height1= 740)

        # # Generate an Image named Annoation.png , showing all the sensors and objects
        # generateImagewithAllAnnoations(sys.argv[1], file_name + '.png')

        # # Make a folder and save all the annotated Image per minute bases
        annotateImage(sys.argv[1], file_name + '.png', minutesToGenrate = 2000)

        # # Generate a video on above generated Image
        # makeVideo(os.path.join(os.getcwd(), 'AnnotatedImage'), fps=10)





