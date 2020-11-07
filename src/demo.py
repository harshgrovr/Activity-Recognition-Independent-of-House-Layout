import os
import shutil

import torch
import random
from matplotlib.lines import Line2D
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn
import sys
from sklearn.model_selection import LeaveOneOut
import datetime
from datetime import datetime
from torch.utils.data.sampler import Sampler, WeightedRandomSampler
from src.network import LSTM

from torch.utils.tensorboard import SummaryWriter

from src.network import HARmodel, CNNModel,LSTMModel
import numpy as np
import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
from src.dataLoader import datasetCSV
import torch.utils.tensorboard
from config.config import config

# give an index(date) start and end index
def getStartAndEndIndex(df, test_index):
    # this line converts the string object in Timestamp object
    date = df['start'].iloc[test_index]
    index = df.index[df['start'] == date].tolist()
    # get start and end of this date
    return index[0], index[-1]

# Give all unique dates index
def getUniqueStartIndex(df):
    # this line converts the string object in Timestamp object
    if isinstance(df['start'][0], str):
        df['start'] = [datetime.strptime(d, '%d-%b-%Y %H:%M:%S') for d in df["start"]]
    # extracting date from timestamp
    if isinstance(df['start'][0], datetime):
        df['start'] = [datetime.date(d) for d in df['start']]
    s = df['start']
    return s[s.diff().dt.days != 0].index.values

def getIDFromClassName(train_label):
    ActivityIdList = config['ActivityIdList']
    train_label = [x for x in ActivityIdList if x["name"] == train_label]
    return train_label[0]['id']

def getClassnameFromID(train_label):
    ActivityIdList = config['ActivityIdList']
    train_label = [x for x in ActivityIdList if x["id"] == int(train_label)]
    return train_label[0]['name']

# Create sequence of input and output depending upon the window size
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data.iloc[i:i+tw, ~input_data.columns.isin(['activity', 'start', 'end'])]
        train_seq = train_seq.values
        train_label = input_data.iloc[i, input_data.columns.isin(['activity'])]
        train_label = train_label.values

        # (values, counts) = np.unique(train_label, return_counts=True)
        # ind = np.argmax(counts)
        # train_label = values[ind]
        train_label = getIDFromClassName(train_label[0])
        inout_seq.append((train_seq, train_label))
    return inout_seq

def splitDatasetIntoTrainAndTest(df):
    testDataFrame = pd.DataFrame([])
    trainDataFrame = pd.DataFrame([])
    trainDataFrame = df[0 : int(df.shape[0] * config['split_ratio'])]
    testDataFrame = df[int(df.shape[0] * config['split_ratio']) : df.shape[0]]
    return trainDataFrame, testDataFrame

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def train(file_name, ActivityIdList):

    csv_file = file_name + '.csv'
    trainDataFrame = pd.read_csv('/home/harsh/Downloads/Thesis/dummyTestforPycharm/data/houseA/one_hot.csv')[132:]

    # testDataFrame = trainDataFrame
    # testDataFrame = pd.read_csv('/home/harsh/Downloads/Thesis/dummyTestforPycharm/data/houseA/one_hot.csv')[:231]

    testDataFrame = pd.read_csv('/home/harsh/Downloads/Thesis/dummyTestforPycharm/data/houseA/eval_csv.csv')[:2805]

    df = trainDataFrame
    # Get class Frequency as a dictionary
    classFrequencyDict = trainDataFrame['activity'].value_counts().to_dict()
    temp_dict ={}


    # Initialize the dict and set frequ value 0 intially because weight tensor in Loss requires all the classes values
    for dict in ActivityIdList:
        temp_dict[dict['id']] = 0

    # make of classLabel and frequency
    for className, frequency in classFrequencyDict.items():
        classLabel = getIDFromClassName(className)
        temp_dict[classLabel] = frequency

    # Sort it according to the class labels
    classFrequenciesList = np.array([value for key, value in sorted(temp_dict.items())])
    classFrequenciesList = np.max(classFrequenciesList)/classFrequenciesList
    classFrequenciesList[classFrequenciesList == np.inf] = 0

    class_weights = classFrequenciesList
    class_weights_dict = {}

    for i, item in enumerate(class_weights):
        class_weights_dict[getClassnameFromID(i)] = item

    print(class_weights_dict)

    if torch.cuda.is_available():
        class_weights = torch.tensor(classFrequenciesList).float().cuda()
    else:
        class_weights = torch.tensor(classFrequenciesList).float()

    model = LSTM(config['input_dim'], config['hidden_dim'])

    if torch.cuda.is_available():
        model.cuda()
    print('cuda available: ', torch.cuda.is_available())


    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=config['learning_rate'], weight_decay=config['decay'])

    # Split the data into test and train
    # trainDataFrame, testDataFrame  = splitDatasetIntoTrainAndTest(df)

    # Generate Test DataLoader
    testData = create_inout_sequences(testDataFrame, config['seq_dim'])
    testLoader = DataLoader(testData, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'],
                            )
    training(config['num_epochs'], trainDataFrame, optimizer, model, criterion, config['seq_dim'], config['input_dim'], config['batch_size'], df, testLoader)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    saved_model_path = os.path.join('../saved_model/model_best.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, saved_model_path)

# Train the Network
def training(num_epochs, trainDataFrame,  optimizer, model, criterion, seq_dim, input_dim, batch_size, df, testLoader):
    minutesToRunFor = []
    randomDaySelected = []

    # for each random run select the random point and minute to run for
    # do this for each random point
    # for number in getUniqueStartIndex(trainDataFrame):
    #     start, end = getStartAndEndIndex(trainDataFrame, number)
    #     number = random.sample(range(start, end - config['seq_dim']), 1)[0]
    #     randomDaySelected.append(number)
    #     minutesToRunFor.append(end - number)

    writer = SummaryWriter('../logs')
    trainData = create_inout_sequences(trainDataFrame, config['seq_dim'])
    trainDataset = datasetCSV(trainData, config['seq_dim'])
    trainLoader = DataLoader(trainDataset, batch_size=config['batch_size'], shuffle=False,
                             num_workers=config['num_workers'],)

    for epoch in range(num_epochs):
        running_loss = 0

        for i, (input, label) in enumerate(trainLoader):
            input = input.view(-1, seq_dim, input_dim)
            if torch.cuda.is_available():
                input = input.float().cuda()
                label = label.cuda()
            else:
                input = input.float()
                label = label

            # Forward pass to get output/logits
            output, (hn, cn) = model(input)

            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(output, label)#weig pram
            running_loss += loss
            loss.backward()  # Backward pass

            optimizer.step()  # Now we can do an optimizer step
            optimizer.zero_grad()  # Reset gradients tensors
        # print(running_loss)
        running_loss = 0
        if epoch % 10 == 9:

            print('\n epoch', epoch)
            train_loss, train_acc, train_f1_score, train_per_class_accuracy = eval_net(
                 model, trainLoader, criterion, config)

            print('train set - average loss: {:.4f}, accuracy: {:.0f}%  train_f1_score: {:.4f} \n '
                  .format(train_loss, 100. * train_acc, train_f1_score))

            print('train per_class accuracy', train_per_class_accuracy)

            valid_loss, valid_acc, val_f1_score, val_per_class_accuracy = eval_net(
                model, testLoader, criterion,config, text='val')

            print('\n valid set - average loss: {:.4f}, accuracy: {:.0f}% val_f1_score {:.4f}:  \n'
                  .format(valid_loss, 100. * valid_acc, val_f1_score))

            print('Val per_class accuracy', val_per_class_accuracy)

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, True)

# Evaluate the network
def eval_net(model, dataloader, criterion, config, text = 'train'):
    model.eval()
    total = 0
    total_loss = 0
    total_correct = 0
    f1 = 0
    all_labels = []
    all_predicted = []
    nb_classes = config['output_dim']
    confusion_matrix = torch.zeros(nb_classes, nb_classes)

    for i, (input, labels) in enumerate(dataloader):
        input = input.view(-1, config['seq_dim'], config['input_dim'])
        if torch.cuda.is_available():
            input = input.float().cuda()
            labels = labels.cuda()
        else:
            input = input.float()
            labels = labels

        # Forward pass to get output/logits
        output, (hn, cn) = model(input)

        total += len(labels)
        _, predicted = torch.max(output.data, 1)

        total_correct += (predicted == labels.data).sum().item()

        loss = criterion(output, labels)
        # crossentropy(reduce=True) for default
        total_loss += loss.item() * len(labels)

        all_labels.extend(labels.cpu())
        all_predicted.extend(predicted.cpu())

        for t, p in zip(labels.view(-1), predicted.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1


    # np.save('./' + text + '_confusion_matrix.npy', confusion_matrix)

    per_class_acc = confusion_matrix.diag() / confusion_matrix.sum(1)
    per_class_acc = per_class_acc.cpu().numpy()
    per_class_acc[np.isnan(per_class_acc)] = -1
    per_class_acc_dict = {}
    for i, entry in enumerate(per_class_acc):
        if entry != -1:
            per_class_acc_dict[getClassnameFromID(i)] = entry

    f1 = f1_score(all_labels, all_predicted, average='macro')

    loss, acc = total_loss / total, total_correct / total

    return loss, acc, f1, per_class_acc_dict

if __name__ == "__main__":
    if sys.argv[1] != None:
        ActivityIdList = config['ActivityIdList']
        file_name = '../data/houseA/houseA'
        # file_name = sys.argv[1].split('.')[0]
        train(file_name, ActivityIdList)
