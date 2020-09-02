import shutil

import h5py
import os
import torch
from sklearn.metrics import f1_score
from torch.autograd import Variable
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from imgaug import augmenters as iaa


from torchvision import transforms

from src.dataLoader import datasetHDF5, datasetFolder
import sys

from sklearn.model_selection import LeaveOneOut
import pandas as pd
import datetime
from datetime import datetime
import cv2

from src.lstmTraining import getIDFromClassName, getClassnameFromID
from src.lstmTrainingWithLOOCV import getUniqueStartIndex
from src.network import CNNLSTM, LSTMModel, Network, EncoderCNN
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


import config
from config.config import config

def save_checkpoint(state, is_best, filename='checkpoint_cnn_lstm.pth.tar'):
    saved_model_path = os.path.join("../saved_model/cnn_lstm/model_best.pth.tar")
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, saved_model_path)



def create_inout_sequences(input_data, tw ):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data.iloc[i:i+tw, input_data.columns.isin(['time_of_the_day'])]
        train_seq = train_seq.values
        train_label = input_data.iloc[i:i+tw, input_data.columns.isin(['activity'])]
        train_label = train_label.values
        (values, counts) = np.unique(train_label, return_counts=True)
        ind = np.argmax(counts)
        train_seq = train_seq[ind]
        train_label = values[ind]
        inout_seq.append([train_seq, train_label])
    return inout_seq

def getWeightedSampler(Data):
    target = np.array([x for x in Data['activity']])

    weights = {}
    for t in np.unique(target):
        weights[t] = 1/len(np.where(target == str(t))[0])

    samples_weights = np.array([weights[t] for t in target])
    sampler = WeightedRandomSampler(weights=samples_weights,num_samples=len(samples_weights), replacement=True)
    return sampler

def evaluate(epoch, model, optimizer, loader, device, writer,split, test=True):
    nb_classes = config['output_dim']
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, True)

    correct = 0
    total = 0
    batches = 0
    f1 = 0
    total_acc = 0
    total_f1 = 0
    correctList = []
    predictedList = []

    if test:
        text = 'test'
    else:
        text = 'train'

    with torch.no_grad():
        for i, (image, label) in enumerate(loader):
            image = image.float().to(device)
            label = label.to(device)
            optimizer.zero_grad()

            batch_size, timesteps, H, W, C = image.size()
            # Change Image shape
            # Change Image shape
            image = image.view(batch_size * timesteps, H, W, C)
            image = image.permute(0, 3, 1, 2)  # from NHWC to NCHW
            output = model(image)

            label = label.view(-1)
            # output = output.view(-1, output.size(2))

            _, predicted = torch.max(output.data, 1)
            total += label.size(0)

            if torch.cuda.is_available():
                correct += (predicted.cpu() == label.cpu()).sum()
            else:
                correct += (predicted == label).sum()
            for t, p in zip(label.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

            correctList.extend(label.cpu())
            predictedList.extend(predicted.cpu())

        f1 = f1_score(correctList, predictedList, average='macro')

        per_class_acc = confusion_matrix.diag() / confusion_matrix.sum(1)
        per_class_acc = per_class_acc.cpu().numpy()
        per_class_acc[np.isnan(per_class_acc)] = 0
        d = {}
        for i in range(len(per_class_acc)):
            d[getClassnameFromID(i)] = per_class_acc[i]
        # print(correct.cpu().item(), total)

        accuracy = 100 * correct.cpu().item() / total

        if test:
            print('\n\n Epoch: {}.  \n\n Test Accuracy: {},  \n\n   Test f1 {}. \n\n  Test pca: {}. \n\n '.format(
                    epoch, accuracy, f1, d))
        else:
            print('\n\n Epoch: {}.  \n\n Train Accuracy: {},  \n\n   Train f1 {}. \n\n  Train pca: {}. \n\n '.format(
                    epoch, accuracy, f1, d))



        writer.add_scalars(str(split) + ' ' + text+' Mean_class_Accuracy', d, epoch + 1)
        writer.add_scalar(str(split) + ' ' + text+' Accuracy', accuracy, epoch + 1)
        writer.add_scalar(str(split) + ' ' + text+' Accuracy', f1, epoch + 1)


def lossAndOtherFunctionality(device, df):
    print(df['activity'].unique())

    df = df[0:70000]
    # Get class Frequency as a dictionary
    classFrequencyDict = df['activity'].value_counts().to_dict()
    temp_dict = {}

    # Initialize the dict and set frequ value 0 intially because weight tensor in Loss requires all the classes values
    for dict in ActivityIdList:
        temp_dict[dict['id']] = 0

    # make of classLabel and frequency
    for className, frequency in classFrequencyDict.items():
        classLabel = getIDFromClassName(className)
        temp_dict[classLabel] = frequency

    # Sort it according to the class labels
    classFrequenciesList = np.array([value for key, value in sorted(temp_dict.items())])
    classFrequenciesList = 1 / classFrequenciesList * 1000
    classFrequenciesList[classFrequenciesList == np.inf] = 0
    class_weights = torch.tensor(classFrequenciesList).float().to(device)
    print(class_weights)
    criterion = nn.CrossEntropyLoss()

    images_path = os.path.join(os.getcwd(), '../', 'data', file_name, 'images')

    folder_list = [f for f in os.listdir(images_path) if os.path.isdir(os.path.join(images_path, f))]
    full_list = [os.path.join(images_path, i) for i in folder_list]
    time_sorted_list = sorted(full_list, key=os.path.getmtime)
    sorted_folder_list = [os.path.basename(i) for i in time_sorted_list]

    uniqueIndex = getUniqueStartIndex(df)
    uniqueIndex = np.append(uniqueIndex,[len(df)] , axis=0)


    # Apply Leave one out on all the h5 files(h5files list)
    loo = LeaveOneOut()
    objectsChannel = cv2.imread(os.path.join(os.getcwd(), '../', 'data', file_name, 'images', 'objectChannel.png'), 0)
    sensorChannel = cv2.imread(os.path.join(os.getcwd(), '../', 'data', file_name, 'images', 'sensorChannel.png'), 0)

    return criterion, uniqueIndex, loo, objectsChannel, sensorChannel, sorted_folder_list, class_weights


def getLoaders(df, uniqueIndex, test_index, train_index, sorted_folder_list, objectsChannel, sensorChannel, class_weights):

    start, end = uniqueIndex[test_index][0], uniqueIndex[test_index + 1][0]
    # start, end = 0, 5307
    end = end -1
    if start != 0:
        dfFrames = [df[:start - 1], df[end + 1:]]
        trainDataFrame = pd.concat(dfFrames)
    else:
        trainDataFrame = df[end + 1:]

    transform = transforms.Compose([
        iaa.Sequential([
            iaa.flip.Fliplr(p=0.2),
            iaa.flip.Flipud(p=0.2),
            iaa.GaussianBlur(sigma=(0.0, 0.1)),
        ]).augment_image
    ])

    # Train dataLoader
    trainDataseq = create_inout_sequences(trainDataFrame[0:100], config['seq_dim'])
    trainDataset = datasetFolder(train_index, sorted_folder_list, file_name, objectsChannel, sensorChannel,
                                 trainDataseq, transform)
    trainLoader = DataLoader(trainDataset, batch_size=config['batch_size'], shuffle=False,
                             num_workers=config['num_workers'], drop_last=True, pin_memory=True,
                             sampler = getWeightedSampler(trainDataFrame))

    testDataFrame = df[start: end]
    testDataseq = create_inout_sequences(testDataFrame[0:100], config['seq_dim'])
    testDataset = datasetFolder(test_index, sorted_folder_list, file_name, objectsChannel, sensorChannel, testDataseq)
    testLoader = DataLoader(testDataset, batch_size=config['batch_size'], shuffle=False,
                            num_workers=config['num_workers'], drop_last=True, pin_memory=True,
                            sampler=getWeightedSampler(testDataFrame))

    return trainLoader, testLoader


def train(file_name, input_dir, csv_file_path, json_file_path, totalFiles):
    writer = SummaryWriter(os.path.join('../logs', file_name, 'cnn_lstm'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_num_iteration_for_LOOCV = 0
    avg_per_class_accuracy = 0
    avg_acc = 0
    avg_f1 = 0


    start_epoch = 0
    df = pd.read_csv(csv_file_path)
    df = df[0:totalFiles]
    criterion, uniqueIndex, loo, objectsChannel, sensorChannel, sorted_folder_list, class_weights = lossAndOtherFunctionality(device, df)

    split = 0

    for train_index, test_index in loo.split(sorted_folder_list):
        # Defining Model, Optimizer and Loss
        model = EncoderCNN().to(device)

        # Parallelize model to multiple GPUs
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)

        print('cuda available: ', torch.cuda.is_available())
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['decay'])

        trainLoader, testLoader = getLoaders(df, uniqueIndex, test_index, train_index, sorted_folder_list,
                                             objectsChannel, sensorChannel, class_weights)

        split += 1
        print('******** SPLIT ************: ', split)
        for epoch in range(config['num_epochs']):
            print('epoch', epoch)
            running_loss = 0
            nb_classes = config['output_dim']


            # # Check dataset
            # for i, (image, label, name) in enumerate(trainLoader):
            #     if df[df['start'] == name[0]]['activity'].item() != getClassnameFromID(label):
            #         print('issue in train dataset', name)
            #
            # for i, (image, label, name) in enumerate(testLoader):
            #     if df[df['start'] == name[0]]['activity'].item() != getClassnameFromID(label):
            #         print('issue in test dataset', name)
            # # Check dataset

            for i, (image, label) in enumerate(trainLoader):
                image = image.float().to(device)
                label = label.to(device)
                optimizer.zero_grad()

                batch_size, timesteps, H, W, C = image.size()
                # Change Image shape
                image = image.view(batch_size * timesteps, H, W, C)
                image = image.permute(0, 3, 1, 2)  # from NHWC to NCHW
                output = model(image)
                label = label.view(-1)

                # output = output.view(-1, output.size(2))

                loss = criterion(output, label)
                loss.backward()  # Backward pass
                optimizer.step()  # Now we can do an optimizer step
                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)

            if epoch % 5 == 0:
                print('\n \n loss is ', running_loss)
                print('Test')
                evaluate(epoch, model, optimizer, testLoader, device, writer,split, test=True)

                print('Train')
                evaluate(epoch, model, optimizer, trainLoader, device, writer,split,  test=False)

        total_num_iteration_for_LOOCV += 1
        # print('\n\n ******** AVERAGE ************** \n\n')
        # print('avg_per_class_accuracy: {}. avg_acc: {}. avg_f1: {}'.format(avg_per_class_accuracy/total_num_iteration_for_LOOCV, avg_acc/total_num_iteration_for_LOOCV, avg_f1/total_num_iteration_for_LOOCV))

if __name__ == "__main__":
    if sys.argv[1] != None:
        file_name = sys.argv[1].split('.json')[0]
        input_dir = os.path.join(os.getcwd(), '../', 'data', file_name)
        csv_file_path = os.path.join(input_dir, file_name + '.csv')
        json_file_path = os.path.join(input_dir, file_name + '.json')
        # csv_length = pd.read_csv(csv_file_path).shape[0]
        ActivityIdList = config['ActivityIdList']
        # file_name = sys.argv[1].split('.')[0]
        # total files
        currentFolderPath= os.path.join(input_dir, 'Sequence_AnnotatedImage')
        filesinCurrentFolder = [name for name in os.listdir(currentFolderPath) if
                                name.endswith('.png') and os.path.isfile(
                                    os.path.join(currentFolderPath, name))]
        totalFiles = len(filesinCurrentFolder)

        train(file_name, input_dir, csv_file_path, json_file_path, totalFiles)

