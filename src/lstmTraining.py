import shutil

import torch
import random
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader
import torch.nn as nn
import sys
from sklearn.model_selection import LeaveOneOut
import datetime
from datetime import datetime
from torch.utils.data.sampler import Sampler, WeightedRandomSampler

from torch.utils.tensorboard import SummaryWriter

from src.network import LSTMModel
import numpy as np
import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from src.dataLoader import datasetCSV
import torch.utils.tensorboard
from config.config import config
import os
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
    train_label = [x for x in ActivityIdList if x["name"] == train_label]
    return train_label[0]['id']

def getClassnameFromID(train_label):
    ActivityIdList = config['ActivityIdList']
    train_label = [x for x in ActivityIdList if x["id"] == int(train_label)]
    return train_label[0]['name']

# Create sequence of input and output depending upon the window size
def create_inout_sequences(input_data, tw ):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data.iloc[i:i+tw, ~input_data.columns.isin(['activity', 'start', 'end'])]
        train_seq = train_seq.values
        train_label = input_data.iloc[i:i+tw, input_data.columns.isin(['activity'])]
        train_label = train_label.values

        (values, counts) = np.unique(train_label, return_counts=True)
        ind = np.argmax(counts)
        train_label = values[ind]
        train_label = getIDFromClassName(train_label)
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

def train(csv_file, ActivityIdList):

    df = pd.read_csv(csv_file)
    # Get class Frequency as a dictionary
    classFrequencyDict = df['activity'].value_counts().to_dict()
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
    classFrequenciesList = 1/classFrequenciesList
    classFrequenciesList[classFrequenciesList == np.inf] = 0
    if torch.cuda.is_available():
        class_weights = torch.tensor(classFrequenciesList).float().cuda()
    else:
        class_weights = torch.tensor(classFrequenciesList).float()

    model = LSTMModel(config['input_dim'], config['hidden_dim'], config['layer_dim'],config['output_dim'], config['seq_dim'])
    if torch.cuda.is_available():
        model.cuda()

    print('cuda available: ', torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss(weight = class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['decay'])

    path = "../saved_model/lstm/model_best.pth.tar"
    start_epoch = 0
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch += checkpoint['epoch']
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(path, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(path))


    # Split the data into test and train
    trainDataFrame, testDataFrame  = splitDatasetIntoTrainAndTest(df)

    # Generate Test DataLoader
    testData = create_inout_sequences(testDataFrame, config['seq_dim'])
    testLoader = DataLoader(testData, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'],
                            drop_last=True)

    training(config['num_epochs'], trainDataFrame, optimizer, model, criterion, config['seq_dim'], config['input_dim'], config['batch_size'], df, testLoader, start_epoch, file_name)

    evaluate(testLoader, model, config['seq_dim'], config['input_dim'],
             batch_size=config['batch_size'])

# Train the Network
def training(num_epochs, trainDataFrame,  optimizer, model, criterion, seq_dim, input_dim, batch_size, df, testLoader, start_epoch, file_name):

    # for each random run select the random point and minute to run for
    # do this for each random point

    writer = SummaryWriter(os.path.join('../logs', file_name, 'lstm'))
    # Get Start Index(Subset) for each of the activity
    activitiesStartDict = getActivitiesStartIndex(trainDataFrame)
    running_loss = 0
    for epoch in range(num_epochs):
        print('epoch', epoch + start_epoch)
        # Get Start Index(Subset) for each of the activity and the minutes to run for
        for index in range(config['no_of_subset']):
            trainData = []
            randomSelectedSubsets = []
            minutesToRunFor = []
            for i, key in enumerate(activitiesStartDict.keys()):
                activityIndexList = activitiesStartDict[key]
                randomSelectedSubsets.append(random.choice(activityIndexList) - config['subset_overlap_length'])
                minutesToRunFor.append(random.choice(range(20, 30)))

            sorted_index = sorted(randomSelectedSubsets)

            for index,(time_to_start_from) in enumerate(sorted_index):

                results = \
                create_inout_sequences(trainDataFrame[time_to_start_from: time_to_start_from + minutesToRunFor[i]],
                                       config['seq_dim'])
                trainData.extend(results)

            # Make Train DataLoader
            trainDataset = datasetCSV(trainData, config['seq_dim'])
            trainLoader = DataLoader(trainDataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'],
                                     drop_last=True)

            hn, cn = model.init_hidden(batch_size)
            for i, (input, label) in enumerate(trainLoader):
                hn.detach_()
                cn.detach_()
                input = input.view(-1, seq_dim, input_dim)

                if torch.cuda.is_available():
                    input = input.float().cuda()
                    label = label.cuda()
                else:
                    input = input.float()
                    label = label

                # Forward pass to get output/logits
                output, (hn, cn) = model((input, (hn, cn)))

                # l1_regularization = torch.tensor(0)
                # for param in model.parameters():
                #     l1_regularization += torch.norm(param, 1) ** 2
                # loss = loss + config['decay'] * l1_regularization
                # Calculate Loss: softmax --> cross entropy loss
                loss = criterion(output, label)#weig pram
                running_loss += loss
                loss.backward()  # Backward pass
                # if (i) % config["accumulation_steps"] -1 == 0:  # Wait for several backward steps
                optimizer.step()  # Now we can do an optimizer step
                optimizer.zero_grad()  # Reset gradients tensors

        if epoch % 10 == 0:
            accuracy, per_class_accuracy = evaluate(testLoader, model, config['seq_dim'], config['input_dim'],
                 batch_size=config['batch_size'])

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

            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                print(value.grad.data.cpu().numpy())
                writer.add_histogram(tag + '/grad', value.grad.data.cpu().numpy(), epoch + 1)

            # plot weights historgram
            for key in model.lstm.state_dict():
                writer.add_histogram(key, model.lstm.state_dict()[key].data.cpu().numpy(), epoch + 1)
            for key in model.fc.state_dict():
                writer.add_histogram(key, model.fc.state_dict()[key].data.cpu().numpy(), epoch + 1)

        # Loggin loss
        writer.add_scalar('Loss', running_loss, epoch + 1)
        print('%d loss: %.3f' %
              (epoch + 1,  running_loss))
        running_loss = 0


# Evaluate the network
def evaluate(testLoader, model, seq_dim, input_dim, batch_size):
    # Initialize the prediction and label lists(tensors)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    nb_classes = config['output_dim']
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    correct = 0
    total = 0
    # Iterate through test dataset
    with torch.no_grad():
        for input, labels in testLoader:
            hn, cn = model.init_hidden(batch_size)
            input = input.view(-1, seq_dim, input_dim)
            # Load images to a Torch Variable
            if torch.cuda.is_available():
                input = input.float().cuda()
                labels = labels.cuda()
            else:
                input = input.float()

            # Forward pass only to get logits/output
            output, (hn, cn) = model((input, (hn, cn)))

            # Get predictions from the maximum value
            _, predicted = torch.max(output, 1)
            # Total number of labels
            total += labels.size(0)
            # Total correct predictions
            if torch.cuda.is_available():
                print(predicted.cpu(), labels.cpu())
                correct += (predicted.cpu() == labels.cpu()).sum()
            else:
                print(predicted, labels)
                correct += (predicted == labels).sum()
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    print('per class accuracy')
    per_class_acc = confusion_matrix.diag() / confusion_matrix.sum(1)
    per_class_acc = per_class_acc.cpu().numpy()
    per_class_acc[np.isnan(per_class_acc)] = -1
    print(confusion_matrix.diag())
    print(confusion_matrix.sum(1))
    d ={}
    for i, entry in enumerate(per_class_acc):
        if entry != -1:
            d[getClassnameFromID(i)] = entry
    print(d)
    pd.isnull(np.array([np.nan, -1], dtype=float))

    df_cm = pd.DataFrame(confusion_matrix, index=[getClassnameFromID(i) for i in range(confusion_matrix.shape[0])],
                         columns=[getClassnameFromID(i) for i in range(confusion_matrix.shape[0])], dtype=float)
    # plt.figure(figsize=(20, 20))
    # sn.heatmap(df_cm, annot=True)
    # plt.show()

    accuracy = 100 * correct / total

    # Print Accuracy
    print('Accuracy: {}'.format(accuracy))
    return accuracy, per_class_acc

if __name__ == "__main__":
    if sys.argv[1] != None:
        file_name = sys.argv[1].split('.json')[0]
        input_dir = os.path.join(os.getcwd(), '../', 'data', file_name)
        csv_file_path = os.path.join(input_dir, file_name + '.csv')
        json_file_path = os.path.join(input_dir, file_name + '.json')
        csv_length = pd.read_csv(csv_file_path).shape[0]
        ActivityIdList = config['ActivityIdList']
        train(csv_file_path, ActivityIdList)



