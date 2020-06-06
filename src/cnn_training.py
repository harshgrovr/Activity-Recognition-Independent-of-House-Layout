import shutil

import h5py
import os
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.dataLoader import datasetHDF5
import sys
from sklearn.model_selection import LeaveOneOut
import pandas as pd
import datetime
from datetime import datetime
import cv2

from src.lstmTraining import getIDFromClassName, getClassnameFromID
from src.lstmTrainingWithLOOCV import getUniqueStartIndex, getStartAndEndIndex, create_inout_sequences
from src.network import CNNModel, CNNLSTM, LSTMModel
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
        # (values, counts) = np.unique(train_label, return_counts=True)
        # ind = np.argmax(counts)
        # train_label = values[ind]
        for i in range(len(train_label)):
            train_label[i] = getIDFromClassName(train_label[i])
        inout_seq.append((train_seq, train_label))
    return inout_seq

def train(file_name, input_dir, csv_file_path, json_file_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    total_num_iteration_for_LOOCV = 0
    total_acc_for_LOOCV = 0

    # Defining Model, Optimizer and Loss
    model = LSTMModel().to(device)
    avg_accuracy, avg_per_class_accuracy, avg_f1 = 0
    print('cuda available: ', torch.cuda.is_available())

    df = pd.read_csv(csv_file_path)
    print(df['activity'].unique())
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
    classFrequenciesList = 1 / classFrequenciesList
    classFrequenciesList[classFrequenciesList == np.inf] = 0
    class_weights = torch.tensor(classFrequenciesList).float().to(device)
    print(class_weights)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['decay'])

    path = "../saved_model/cnn_lstm/model_best.pth.tar"
    start_epoch = 0
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch += checkpoint['epoch']
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(path, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(path))



    # Get names of all h5 files
    h5Files = [f.split('.h5')[:-1] for f in os.listdir(os.path.join(os.getcwd(), '../','data', file_name, 'h5py')) if f.endswith('.h5')]

    # Just first h5 file stores object and sensor channels. Get Object channel and sensorchannel from the first file
    h5Directory = os.path.join(os.getcwd(), '../', 'data', file_name, 'h5py')
    firstdate = df.iloc[0, 0]
    if not isinstance(firstdate, datetime):
        firstdate = datetime.strptime(firstdate, '%d-%b-%Y %H:%M:%S')
    firstdate = firstdate.strftime('%d-%b-%Y') + '.h5'

    # Get sensor value stored in first h5 file
    with h5py.File(os.path.join(h5Directory, firstdate), 'r') as f:
        objectsChannel = f['object'].value
        sensorChannel = f['sensor'].value


    # Prepare text data
    # Generate unique index for each day in csv
    uniqueIndex = getUniqueStartIndex(df)


    # Apply Leave one out on all the h5 files(h5files list)
    loo = LeaveOneOut()
    for train_index, test_index in loo.split(h5Files):
        print('split: ', total_num_iteration_for_LOOCV)
        total_num_iteration_for_LOOCV += 1

        start, end = getStartAndEndIndex(df, uniqueIndex[test_index])

        if start != 0:
            dfFrames = [df[:start - 1], df[end + 1:]]
            trainDataFrame = pd.concat(dfFrames)
        else:
            trainDataFrame = df[end + 1:]

        # Train dataLoader
        trainDataseq = create_inout_sequences(trainDataFrame[0:100], config['seq_dim'])
        trainDataset = datasetHDF5(objectsChannel, sensorChannel, h5Files, h5Directory, train_index, trainDataseq)
        trainLoader = DataLoader(trainDataset, batch_size=config['batch_size'], shuffle=False,
                                 num_workers=config['num_workers'], drop_last=True)
        training(config['num_epochs'], trainLoader, optimizer, model, criterion, start_epoch, trainLoader)

        # Test dataLoader
        if start - config['seq_dim'] > 0:
            testDataFrame = df[start - config['seq_dim']: end]
            testDataSeq = create_inout_sequences(testDataFrame[0:100], config['seq_dim'])
            testDataset = datasetHDF5(objectsChannel, sensorChannel, h5Files, h5Directory, test_index, testDataSeq)
            testLoader = DataLoader(testDataset, batch_size=config['batch_size'], shuffle=False,
                                    num_workers=config['num_workers'], drop_last=True)
            print('testing it')
            accuracy, per_class_accuracy, f1 = evaluate(testLoader, model)
            avg_accuracy += accuracy
            avg_per_class_accuracy += per_class_accuracy
            avg_f1 += f1
            print('in meanwhile testing: ')
            print(avg_accuracy, avg_per_class_accuracy, avg_f1)

    print("Avg. Accuracy is {}".format(avg_accuracy/total_num_iteration_for_LOOCV))
    print("Avg. per_class_accuracy is {}".format(avg_per_class_accuracy/total_num_iteration_for_LOOCV))
    print("Avg. F1 is {}".format(avg_f1/total_num_iteration_for_LOOCV))

def image_from_tensor(image):
    image = image[0].numpy()
    image = image[0]
    print(np.unique(image[:, :, [0,1,2]]))
    cv2.imshow('img', image[:, :, [0,1,2]])
    cv2.waitKey(0)
    cv2.imshow('img', 255 * image[:, :, 3])
    cv2.waitKey(0)
    cv2.imshow('img', 255 * image[:, :, 4])
    cv2.waitKey(0)


def training(num_epochs, testLoader, optimizer, model, criterion, start_epoch, trainLoader, accumulation_steps = config['accumulation_steps']):

    writer = SummaryWriter(os.path.join('../logs', file_name, 'cnn_lstm'))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for epoch in range(num_epochs):
        print('epoch', epoch + start_epoch)
        running_loss = 0

        hn,cn = model.init_hidden(config['batch_size'])
        for i, (image, label, objectChannel, sensorChannel, textData) in enumerate(trainLoader):
            hn.detach_()
            cn.detach_()
            batch_size, timesteps, H, W = image.size()

            # Change Image shape
            image = image.view(batch_size * timesteps, H,W,1)
            image = image.permute(0, 3, 1, 2)  # from NHWC to NCHW

            # Change object channel shape
            objectChannel = objectChannel.view(batch_size * timesteps, H, W, 1)
            objectChannel = objectChannel.permute(0, 3, 1, 2)  # from NHWC to NCHW

            # Change object channel shape
            sensorChannel = sensorChannel.view(batch_size * timesteps, H, W, 1)
            sensorChannel = sensorChannel.permute(0, 3, 1, 2)  # from NHWC to NCHW

            if i % 10 == 9:
                print(i)

            image = image.float().to(device)
            label = label.to(device)
            objectChannel = objectChannel.float().to(device)
            sensorChannel = sensorChannel.float().to(device)
            textData = textData.float().to(device)

            # Forward pass to get output/ logits
            output, (hn,cn) = model(image, label,objectChannel, sensorChannel,textData, hn,cn)

            # Calculate Loss: softmax --> cross entropy loss
            label = label.view(-1)
            output = output.view(-1, output.size(2))
            loss = criterion(output, label)

            # print(loss)
            running_loss += loss
            loss = loss / accumulation_steps  # Normalize our loss (if averaged)
            loss.backward()  # Backward pass
            if i % accumulation_steps == accumulation_steps - 1:  # Wait for several backward steps
                optimizer.step()  # Now we can do an optimizer step
                optimizer.zero_grad()  # Reset gradients tensors
        print('epoch: {} Loss: {}'.format(epoch, running_loss))

        if epoch % 20 == 19:
            accuracy, per_class_accuracy,f1 = evaluate(testLoader, model)
            print('train accuracy', accuracy)
            print('per class accuracy', per_class_accuracy)
            print('F1 score', f1)


            accuracy, per_class_accuracy, f1 = evaluate(trainLoader, model)
            print('test accuracy', accuracy)
            print('per class accuracy', per_class_accuracy)
            print('F1 score', f1)
            # Logging mean class accuracy
            d = {}
            for i in range(len(per_class_accuracy)):
                d[getClassnameFromID(i)] = per_class_accuracy[i]

            writer.add_scalars('Mean_class_Accuracy', d, epoch + 1)

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, True)

        # Loggin loss
        writer.add_scalar('Loss', running_loss, epoch + 1)
        print('%d loss: %.3f' %
              (epoch + 1,  running_loss))

# Calculate Accuracy
def evaluate(testLoader, model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    nb_classes = config['output_dim']
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    correct = 0
    total = 0
    f1 = 0
    batches = 0

    # Iterate through test dataset
    with torch.no_grad():
        hn, cn = model.init_hidden(config['batch_size'])
        for i, (image, labels, objectChannel, sensorChannel, textData) in enumerate(testLoader):
            hn.detach_()
            cn.detach_()
            batch_size, timesteps, H, W = image.size()

            # Change Image shape
            image = image.view(batch_size * timesteps, H, W, 1)
            image = image.permute(0, 3, 1, 2)  # from NHWC to NCHW

            # Change object channel shape
            objectChannel = objectChannel.view(batch_size * timesteps, H, W, 1)
            objectChannel = objectChannel.permute(0, 3, 1, 2)  # from NHWC to NCHW

            # Change object channel shape
            sensorChannel = sensorChannel.view(batch_size * timesteps, H, W, 1)
            sensorChannel = sensorChannel.permute(0, 3, 1, 2)  # from NHWC to NCHW

            if i % 10 == 9:
                print(i)

            image = image.float().to(device)
            labels = labels.to(device)
            objectChannel = objectChannel.float().to(device)
            sensorChannel = sensorChannel.float().to(device)
            textData = textData.float().to(device)

            # Forward pass to get output/ logits
            output, (hn, cn) = model(image,labels, objectChannel, sensorChannel, textData, hn, cn)

            labels = labels.view(-1)
            output = output.view(-1, output.size(2))

            # Get predictions from the maximum value
            _, predicted = torch.max(output.data, 1)
            # Total correct predictions
            correct += (predicted.to(device) == labels.to(device)).sum()
            f1 += f1_score(labels.cpu(), predicted.cpu(), average='macro')
            batches = i
            print('batch: ',batches)
            # Total number of labels
            total += labels.size(0)

            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    f1 = f1/batches
    print('f1 score: ',f1)
    print('per class accuracy')
    per_class_acc = confusion_matrix.diag() / confusion_matrix.sum(1)
    per_class_acc = per_class_acc.cpu().numpy()
    per_class_acc[np.isnan(per_class_acc)] = -1
    d = {}
    for i in range(len(per_class_acc)):
        if per_class_acc[i] != -1:
            d[getClassnameFromID(i)] = per_class_acc[i]

    print(d)

    # pd.isnull(np.array([np.nan, -1], dtype=float))

    # df_cm = pd.DataFrame(confusion_matrix, index=[getClassnameFromID(i) for i in range(confusion_matrix.shape[0])],
    #                      columns=[getClassnameFromID(i) for i in range(confusion_matrix.shape[0])], dtype=float)
    # plt.figure(figsize=(20, 20))
    # sn.heatmap(df_cm, annot=True)
    # plt.show()

    accuracy = 100 * correct / total

    # Print Accuracy
    print('Accuracy: {}'.format(accuracy))
    return accuracy, per_class_acc, f1

if __name__ == "__main__":
    if sys.argv[1] != None:
        file_name = sys.argv[1].split('.json')[0]
        input_dir = os.path.join(os.getcwd(), '../', 'data', file_name)
        csv_file_path = os.path.join(input_dir, file_name + '.csv')
        json_file_path = os.path.join(input_dir, file_name + '.json')
        csv_length = pd.read_csv(csv_file_path).shape[0]
        ActivityIdList = config['ActivityIdList']
        # file_name = sys.argv[1].split('.')[0]
        train(file_name, input_dir, csv_file_path, json_file_path)




