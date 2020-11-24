### Code for just one house, train/test/val

# indexes for test and val are given randomly.

import shutil

import torch
import random
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader
import torch.nn as nn
import sys
import torchvision.datasets as dset
from sklearn.model_selection import LeaveOneOut
import datetime
from datetime import datetime
from torch.utils.data.sampler import Sampler, WeightedRandomSampler
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter


from src.network import LSTM
import numpy as np
# import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from src.dataLoader import datasetCSV
import torch.utils.tensorboard
from config.config import config
import os
# give an index(date) start and end index
from src.pytorchtools import EarlyStopping


def getStartAndEndIndex(df, test_index):
    # this line converts the string object in Timestamp object
    date = df['start'].iloc[test_index].item()
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

# Give all index of new activity starting
def getActivitiesStartIndex(df):
    s = df['activity']
    activityStartIndexes = np.array(s[s.ne(s.shift(-1)) == True].index.values)
    activityStartIndexes = np.insert(activityStartIndexes, 0, 4, axis= 0)
    activityStartDict = {}
    for index in activityStartIndexes[:-1]:
        activityName = s[index+1]
        activityStartDict.setdefault(activityName, []).append(index + 1)
    return activityStartDict


def getIDFromClassName(train_label):
    ActivityIdList = config['ActivityIdList']
    # print(train_label)
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
        if i % 10000 == 0:
            print('creating sequence')
        train_seq = input_data.iloc[i:i+tw, ~input_data.columns.isin(['activity', 'start', 'end'])]
        train_seq = train_seq.values
        train_label = input_data.iloc[i:i+tw, input_data.columns.isin(['activity'])]
        train_label = train_label.values
        (values, counts) = np.unique(train_label, return_counts=True)
        ind = np.argmax(counts)
        train_label = getIDFromClassName(values[ind])
        # for i in range(len(train_label)):
        #     train_label[i] = getIDFromClassName(train_label[i])
        inout_seq.append((train_seq, train_label))
    return inout_seq

def splitDatasetIntoTrainAndTest(df):
    testDataFrame = pd.DataFrame([])
    trainDataFrame = pd.DataFrame([])
    trainDataFrame = df[0 : int(df.shape[0] * config['split_ratio'])]
    testDataFrame = df[int(df.shape[0] * config['split_ratio']) : df.shape[0]]
    return trainDataFrame, testDataFrame

def save_checkpoint(state, is_best, filename='checkpoint_lstm.pth.tar'):
    saved_model_path = os.path.join("../saved_model/lstm/model_best.pth.tar")
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, saved_model_path)

def getWeightedSampler(trainData):
    target = [trainData[x][1] for x in range(len(trainData))]
    class_sample_count = []
    for t in np.unique(target):
        class_sample_count.append([t, len(np.where(target == t)[0])])

    weights = np.zeros(np.max(target) + 1)
    for val in class_sample_count:
        weights[val[0]] = 1/val[1]

    samples_weights = np.array([weights[t] for t in target])

    sampler = WeightedRandomSampler(weights=samples_weights,num_samples=len(samples_weights))
    return sampler

def getWeightedLoss(trainDataFrame):
    # Get class Frequency as a dictionary

    classFrequencyDict = trainDataFrame['activity'].value_counts().to_dict()
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
    classFrequenciesList = (1 / classFrequenciesList)
    classFrequenciesList[classFrequenciesList == np.inf] = 0

    if torch.cuda.is_available():
        class_weights = torch.tensor(classFrequenciesList).float().cuda()
    else:
        class_weights = torch.tensor(classFrequenciesList).float()


    criterion = nn.CrossEntropyLoss(weight=class_weights)
    return criterion

def train(csv_file, ActivityIdList, ob_csv_file_path = None, decompressed_csv_path = None):
    total_num_iteration_for_LOOCV = 0
    total_acc_for_LOOCV = []
    total_f1_for_LOOCV = []
    total_per_class_accuracy = []
    total_confusion_matrix = []

    score = 0
    accuracy = 0
    confusion_matrix = np.zeros((config['output_dim'], config['output_dim']))
    df = None
    # read csv Files
    house_name, all_test_loss, all_test_acc, all_test_f1_score, all_test_per_class_accuracy, all_test_confusion_matrix = [], [], [], [], [], []


    house_name_list = ['ordonezB', 'houseB', 'houseC', 'houseA', 'ordonezA']
    decompressed_csv = pd.read_csv(decompressed_csv_path)
    compressed_csv_file = pd.read_csv(ob_csv_file_path)

    uniqueIndex = getUniqueStartIndex(decompressed_csv)

    # Mapped Activity as per the config/generalizing the activities not present in all csvs'

    # compressed_csv_file = compressed_csv_file.iloc[:,:-1]

    loo = LeaveOneOut()

    for train_index, test_index in loo.split(uniqueIndex):
        print('split: ', total_num_iteration_for_LOOCV)
        total_num_iteration_for_LOOCV += 1
        df = decompressed_csv
        # Get start and end of test dataset
        start, end = getStartAndEndIndex(decompressed_csv, uniqueIndex[test_index])
        # make dataframe for train, skip everything b/w test start and test end. rest everything is train.
        dfFrames = [df[:start], df[end:]]
        trainDataFrame = pd.concat(dfFrames)

        # Divide train, test and val dataframe
        valDataFrame = trainDataFrame[:int(len(trainDataFrame) * config['split_ratio'])]
        trainDataFrame = trainDataFrame[int(len(trainDataFrame) * config['split_ratio']):]
        testDataFrame = df[start:end]

        # # Get test Decompressed CSV index
        # test_start_Decompressed, test_end_Decompressed = 967, 1036
        #
        # trainDfFramesIndex = []
        # valDfFrames = []
        # # Make val dataframe excluding the house
        #
        # valDfFrames.append(compressed_csv_file[132: 281])
        #
        # testDFFrameIndex = np.arange(test_start_Decompressed, test_end_Decompressed)
        #
        # trainDfFramesIndex = list(set(np.arange(len(compressed_csv_file))) - set(trainDfFramesIndex) - set(testDFFrameIndex))
        #
        # # Train, Val, Test Data frame for current split
        # trainDataFrame = compressed_csv_file.iloc[trainDfFramesIndex]
        # valDataFrame = pd.concat(valDfFrames)
        # testDataFrame = decompressed_csv.iloc[test_start_Decompressed: test_end_Decompressed]

        print('Total Epochs :', config['num_epochs'])

        model = LSTM(config['input_dim'], config['hidden_dim'])

        if torch.cuda.is_available():
            model.cuda()
        print('cuda available: ', torch.cuda.is_available())

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['decay'])

        # path = "../saved_model/lstm/model_best.pth"
        start_epoch = 0

        # # # Load Saved Model if exists
        # if os.path.isfile(path):
        #     print("=> loading checkpoint '{}'".format(path))
        #     checkpoint = torch.load(path, map_location=torch.device('cpu'))
        #     model.load_state_dict(checkpoint['state_dict'])
        #     optimizer.load_state_dict(checkpoint['optimizer'])
        #     print("=> loaded checkpoint '{}'"
        #           .format(path))
        # else:
        #     print("=> no checkpoint found at '{}'".format(path))

        # get Weighted Loss
        criterion = getWeightedLoss(trainDataFrame)

        # generate train, val and test sequence list based upon above dataframe.
        trainDataseq = create_inout_sequences(trainDataFrame, config['seq_dim'])
        valDataSeq = create_inout_sequences(valDataFrame, config['seq_dim'])
        testDataSeq = create_inout_sequences(testDataFrame, config['seq_dim'])


        trainDataset = datasetCSV(trainDataseq, config['seq_dim'])
        valDataset = datasetCSV(valDataSeq, config['seq_dim'])
        testDataset = datasetCSV(testDataSeq, config['seq_dim'])

        # Make Loaders for each
        trainLoader = DataLoader(trainDataset, batch_size=config['batch_size'], shuffle=False,
                                 num_workers=config['num_workers'],
                                 drop_last=True)
        valLoader = DataLoader(valDataset, batch_size=config['batch_size'], shuffle=False,
                                num_workers=config['num_workers'],
                                drop_last=True)

        testLoader = DataLoader(testDataset, batch_size=config['batch_size'], shuffle=False,
                                num_workers=config['num_workers'],
                                drop_last=True)

        early_stopping = EarlyStopping(patience=15, verbose=True)

        training(config['num_epochs'], trainLoader, optimizer, model, criterion, config['seq_dim'],
                 config['input_dim'], config['batch_size'],
                 df, valLoader, start_epoch, file_name, early_stopping)

        test_loss, test_acc, test_f1_score, test_per_class_accuracy, test_confusion_matrix = eval_net(
            model, testLoader, criterion, config, text='test')


        total_acc_for_LOOCV.append(test_acc)
        total_f1_for_LOOCV.append(test_f1_score)
        total_per_class_accuracy.append(test_per_class_accuracy)
        total_confusion_matrix.append(test_confusion_matrix)


    print('test_acc:\t', np.mean(total_acc_for_LOOCV), '\t test f1 score', np.mean(total_f1_for_LOOCV),
          '\t test_per_class_accuracy: \n', dict(pd.DataFrame(total_per_class_accuracy).mean()))

    np.save('ordonezA_confusion_matrix', dict(pd.DataFrame(total_confusion_matrix).mean()))


def log_mean_class_accuracy(writer, per_class_accuracy, epoch, test_index, datasettype):
    # Logging mean class accuracy
    d = {}
    for i in range(len(per_class_accuracy)):
        d[i] = per_class_accuracy[i]

    writer.add_scalars(str(test_index) + str(datasettype) + '_Mean_class_Accuracy', d, epoch + 1)


# Train the Network
def training(num_epochs, trainLoader,  optimizer, model, criterion, seq_dim, input_dim, batch_size, df, valLoader, start_epoch, file_name, early_stopping):
    valid_acc, val_per_class_accuracy, valid_loss, val_f1_score, val_confusion_matrix = [],[],[],[],[]
    writer = SummaryWriter(os.path.join('../../../logs', file_name, 'masterleaveOneHouseOut'))
    model.train()
    total_num_iteration_for_LOOCV = 0

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

            output, (hn,cn) = model((input))
            loss = criterion(output, label)#weig pram
            running_loss += loss
            loss.backward()  # Backward pass
            optimizer.step()  # Now we can do an optimizer stepx`
            optimizer.zero_grad()  # Reset gradients tensors

        if epoch % 10 == 9:
            train_loss, train_acc, train_f1_score, train_per_class_accuracy, train_confusion_matrix = eval_net(
                model, trainLoader, criterion, config)

            valid_loss, valid_acc, val_f1_score, val_per_class_accuracy, val_confusion_matrix= eval_net(
                model, valLoader, criterion, config, text='test')

            print('\n epoch', epoch)

            print('train set - average loss: {:.4f}, accuracy: {:.0f}%  train_f1_score: {:.4f} \n '
                  .format(train_loss, 100. * train_acc, train_f1_score))

            # print('train per_class accuracy', train_per_class_accuracy)

            print('\n valid set - average loss: {:.4f}, accuracy: {:.0f}% val_f1_score {:.4f}:  \n'
                  .format(valid_loss, 100. * valid_acc, val_f1_score))

            # print('Val per_class accuracy', val_per_class_accuracy)

            print(
                '\n\n --------------------------------------------------------------------------------------------\n\n')



        # early_stopping needs the validation F1 score to check if it has increased,
        # and if it has, it will make a checkpoint of the current model

            early_stopping(val_f1_score, model, str(0))

            if early_stopping.early_stop:
                print("Early stopping")
                break
    return valid_acc, val_per_class_accuracy, valid_loss, val_f1_score, val_confusion_matrix

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

    return loss, acc, f1, per_class_acc_dict, confusion_matrix

def getParticularsOfEachHouse(csv_file_path):

    df = pd.read_csv(csv_file_path)
    uniqueActivities = np.unique(df['activity'].values)
    output_dim = len(uniqueActivities)
    input_dim = len(df.iloc[0,3:])
    activity_dict_list = []
    for i, activity in enumerate(uniqueActivities):
        d = {}
        d['id'] = i
        d['name'] = activity
        activity_dict_list.append(d)


    return input_dim, output_dim, activity_dict_list

if __name__ == "__main__":
    if sys.argv[1] != None:
        house_name_list = ['ordonezB', 'houseB', 'houseC', 'houseA', 'ordonezA']
        for house in house_name_list:
            file_name = sys.argv[1].split('.json')[0]
            input_dir = os.path.join(os.getcwd(), '../', 'data', file_name)
            csv_file_path = os.path.join(input_dir, file_name + '.csv')

            # file_path = os.path.join(input_dir, file_name + '.json')
            # csv_length = pd.read_csv(csv_file_path).shape[0]
            house_list = []

            ob_csv_file_path = os.path.join(os.getcwd(), '../', 'data', house, 'ob_' + house + '.csv')
            decompressed_csv_path = os.path.join(os.getcwd(), '../', 'data', house, 'ob_' + house + '.csv')

            config['input_dim'], config['output_dim'], config['ActivityIdList'] = getParticularsOfEachHouse(ob_csv_file_path)

            ActivityIdList = config['ActivityIdList']
            train(csv_file_path, ActivityIdList, ob_csv_file_path, decompressed_csv_path)



