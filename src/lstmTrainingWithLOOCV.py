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
        # (values, counts) = np.unique(train_label, return_counts=True)
        # ind = np.argmax(counts)
        # train_label = values[ind]
        for i in range(len(train_label)):
            train_label[i] = getIDFromClassName(train_label[i])
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
    classFrequenciesList = 1 / classFrequenciesList
    classFrequenciesList[classFrequenciesList == np.inf] = 0

    if torch.cuda.is_available():
        class_weights = torch.tensor(classFrequenciesList).float().cuda()
    else:
        class_weights = torch.tensor(classFrequenciesList).float()


    criterion = nn.CrossEntropyLoss(weight=class_weights)
    return criterion

def train(csv_file, ActivityIdList):
    total_num_iteration_for_LOOCV = 0
    total_acc_for_LOOCV = 0
    score = 0
    accuracy = 0
    confusion_matrix = np.zeros((config['output_dim'], config['output_dim']))
    # read csv File
    df = pd.read_csv(csv_file)

    df = df[0:198147]

    # df = df[0:10000]

    # Generate unique index for each day in csv
    uniqueIndex = getUniqueStartIndex(df)

    # leave one out
    loo = LeaveOneOut()
    print('Total splits: ', len(uniqueIndex) - 1)
    print('Total Epochs per split:', config['num_epochs'])
    total_num_iteration_for_LOOCV = len(uniqueIndex) - 1
    # Generate split over uniqueIndex and train and evaluate over it
    for train_index, test_index in loo.split(uniqueIndex):
        if test_index == 0:
            continue
        model = LSTM(config['input_dim'], config['hidden_dim'])
        if torch.cuda.is_available():
            model.cuda()
        print('cuda available: ', torch.cuda.is_available())

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['decay'])

        path = "../saved_model/lstm/model_best.pth.tar"
        start_epoch = 0

        # # # Load Saved Model if exists
        # if os.path.isfile(path):
        #     print("=> loading checkpoint '{}'".format(path))
        #     checkpoint = torch.load(path, map_location=torch.device('cpu'))
        #     model.load_state_dict(checkpoint['state_dict'])
        #     optimizer.load_state_dict(checkpoint['optimizer'])
        #     start_epoch += checkpoint['epoch']
        #     print("=> loaded checkpoint '{}' (epoch {})"
        #           .format(path, checkpoint['epoch']))
        # else:
        #     print("=> no checkpoint found at '{}'".format(path))

        # Get start and end of test dataset

        start, end = getStartAndEndIndex(df, uniqueIndex[test_index])
        # make dataframe for train, skip everything b/w test start and test end. rest everything is train.

        if start != 0:
            dfFrames = [df[:start - 1], df[end + 1:]]
            trainDataFrame = pd.concat(dfFrames)
        else:
            trainDataFrame = df[end + 1:]


        # get Weighted Loss
        criterion = getWeightedLoss(trainDataFrame)

        # generate train sequence list based upon above dataframe.
        trainDataseq = create_inout_sequences(trainDataFrame, config['seq_dim'])
        testLoader = []
        # Make Test DataLoader
        flag = 0

        if start - config['seq_dim'] > 0:
            flag = 1
            print('start and end is: ',start, end)
            testDataFrame = df[start - config['seq_dim']: end]
            testDataSeq = create_inout_sequences(testDataFrame, config['seq_dim'])
            testDataset = datasetCSV(testDataSeq, config['seq_dim'])
            testLoader = DataLoader(testDataset, batch_size=config['batch_size'], shuffle=False,
                                     num_workers=config['num_workers'],
                                     drop_last=True)
            print('length of test dataframe:  ', len(testDataFrame))
            print('length of train dataframe: ', len(trainDataFrame))

        # Make Train DataLoader
        trainDataset = datasetCSV(trainDataseq, config['seq_dim'])
        trainLoader = DataLoader(trainDataset, batch_size=config['batch_size'], shuffle=False,
                                 num_workers=config['num_workers'],
                                 drop_last=True)



        training(config['num_epochs'], trainLoader, optimizer, model, criterion, config['seq_dim'],
                 config['input_dim'], config['batch_size'],
                 df, testLoader, start_epoch, file_name, flag)
        per_class_accuracies = np.zeros(config['output_dim'])
        # Generate Test DataLoader
        if start - config['seq_dim'] > 0:
            print('Testing')
            acc, per_class_accuracy, loss, f1, matrix = evaluate(testLoader, model, config['seq_dim'], config['input_dim'],
                 config['batch_size'], criterion)
            print('testing accuracy', acc)
            # print('testing loss', loss)
            print('F1 score', f1)
            print('per class acuracy : ', per_class_accuracy)
            # print('confusion matrix: ',confusion_matrix)
            score += f1
            confusion_matrix += np.array(matrix)
            accuracy += acc
            per_class_accuracies += np.array(per_class_accuracy)

            print('avg score: ', (score / total_num_iteration_for_LOOCV))
            print('per class accuracy : ', np.divide(per_class_accuracies, np.array(total_num_iteration_for_LOOCV)))
            print('confusion matrix: ', np.divide(confusion_matrix, np.array(total_num_iteration_for_LOOCV)))
            print('avg accuracy: ', (accuracy / (total_num_iteration_for_LOOCV)))
        break
    total_num_iteration_for_LOOCV =1


    confusion_matrix *= np.array(total_num_iteration_for_LOOCV)
    confusion_matrix /= total_num_iteration_for_LOOCV
    confusion_matrix = confusion_matrix.astype(int)
    # print('avg score: ', (score / total_num_iteration_for_LOOCV))
    # print('per class accuracy : ', np.divide(per_class_accuracies, np.array(total_num_iteration_for_LOOCV)))
    # print('confusion matrix: ', np.divide(confusion_matrix , np.array(total_num_iteration_for_LOOCV)))
    # print('avg accuracy: ', (accuracy / (total_num_iteration_for_LOOCV)))

    pd.isnull(np.array([np.nan, -1], dtype=float))

    df_cm = pd.DataFrame(confusion_matrix, index=[getClassnameFromID(i) for i in range(confusion_matrix.shape[0])],
                         columns=[getClassnameFromID(i) for i in range(confusion_matrix.shape[0])], dtype=float)
    # plt.figure(figsize=(20, 20))
    # sn.heatmap(df_cm, annot=True)
    # plt.show()


def log_mean_class_accuracy(writer, per_class_accuracy, epoch, datasettype):
    # Logging mean class accuracy
    d = {}
    for i in range(len(per_class_accuracy)):
        d[getClassnameFromID(i)] = per_class_accuracy[i]

    writer.add_scalars(datasettype + 'Mean_class_Accuracy', d, epoch + 1)


# Train the Network
def training(num_epochs, trainLoader,  optimizer, model, criterion, seq_dim, input_dim, batch_size, df, testLoader, start_epoch, file_name, flag):

    # for each random run select the random point and minute to run for
    # do this for each random point

    writer = SummaryWriter(os.path.join('../logs', file_name, 'lstm'))

    for epoch in range(num_epochs):

        running_loss = 0
        print('epoch', epoch + start_epoch)
        # Get Start Index(Subset) for each of the activity and the minutes to run for
        for i, (input, label) in enumerate(trainLoader):

            input = input.view(-1, seq_dim, input_dim)

            if torch.cuda.is_available():
                input = input.float().cuda()
                label = label.cuda()
            else:
                input = input.float()
                label = label

            output, (hn,cn) = model((input))
            output = output.view(-1, output.size(2))
            label = label.view(-1, label.size(2)).squeeze()



            # l1_regularization = torch.tensor(0)
            # for param in model.parameters():
            #     l1_regularization += torch.norm(param, 1) ** 2
            # loss = loss + config['decay'] * l1_regularization
            # Calculate Loss: softmax --> cross entropy loss

            loss = criterion(output, label)#weig pram

            loss *= config['seq_dim']
            running_loss += loss
            loss.backward()  # Backward pass
            optimizer.step()  # Now we can do an optimizer stepx`
            optimizer.zero_grad()  # Reset gradients tensors



        if epoch % 10 == 0:
            accuracy, per_class_accuracy, trainLoss, f1, _ = evaluate(trainLoader, model, config['seq_dim'], config['input_dim'], config['batch_size'], criterion, train=True )
        #     # print('train loss is:',trainLoss)
            writer.add_scalar('train' + 'Accuracy', accuracy, epoch + start_epoch + 1)
            log_mean_class_accuracy(writer, per_class_accuracy, epoch + 1, datasettype='train')

            # Loggin trainloss
            writer.add_scalar('train Loss', trainLoss, epoch + start_epoch+ 1)
            writer.add_scalar('train f1', f1, epoch + start_epoch + 1)

            if flag != 0:
                accuracy, per_class_accuracy, testLoss, f1,_ = evaluate(testLoader, model, config['seq_dim'], config['input_dim'],
                 config['batch_size'], criterion)

                print('Test loss is:', testLoss)

                writer.add_scalar('test' + 'Accuracy', accuracy, epoch + start_epoch+ 1)
                log_mean_class_accuracy(writer, per_class_accuracy, epoch + start_epoch + 1, datasettype='test')
                # Loggin test loss
                writer.add_scalar('test Loss', testLoss, epoch + start_epoch+ 1)

                writer.add_scalar('test f1', f1, epoch + start_epoch + 1)

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, True)

            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                # print(value.grad.data.cpu().numpy())
                writer.add_histogram(tag + '/grad', value.grad.data.cpu().numpy(), epoch + start_epoch+ 1)


        print('%d loss: %.3f' %
              (epoch + 1,  running_loss))
        running_loss = 0


# Evaluate the network
def evaluate(testLoader, model, seq_dim, input_dim, batch_size, criterion, train=False):
    text = "test"
    if train:
        text = "train"
    # Initialize the prediction and label lists(tensors)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    running_loss = 0
    nb_classes = config['output_dim']
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    correct = 0
    total = 1
    batches = 0
    # Iterate through test dataset
    with torch.no_grad():
        f1 = 0
        for i, (input, labels) in enumerate(testLoader):
            input = input.view(-1, seq_dim, input_dim)
            batches = i
            # Load images to a Torch Variable
            if torch.cuda.is_available():
                input = input.float().cuda()
                labels = labels.cuda()
            else:
                input = input.float()

            # Forward pass only to get logits/output
            output, (hn, cn) = model(input)

            output = output.view(-1, output.size(2))
            labels = labels.view(-1, labels.size(2)).squeeze()

            loss = criterion(output, labels)  # weig pram
            running_loss += loss
            # Get predictions from the maximum value
            _, predicted = torch.max(output, 1)
            f1 += f1_score(labels.cpu(), predicted.cpu(),average='macro')
            # Total number of labels
            total += labels.size(0)
            # Total correct predictions
            if torch.cuda.is_available():
                # print(predicted.cpu(), labels.cpu())
                correct += (predicted.cpu() == labels.cpu()).sum()
                # print(correct)
            else:
                # print(predicted, labels)
                correct += (predicted == labels).sum()
                # print(correct)
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1


    np.save('./' + text + '_confusion_matrix.npy', confusion_matrix)

    # print('F1 SCORE',f1/batches)
    f1 = f1/batches
    # print('per class accuracy')
    per_class_acc = confusion_matrix.diag() / confusion_matrix.sum(1)
    per_class_acc = per_class_acc.cpu().numpy()
    per_class_acc[np.isnan(per_class_acc)] = -1
    # print(confusion_matrix.diag())
    # print(confusion_matrix.sum(1))
    d ={}
    for i, entry in enumerate(per_class_acc):
        if entry != -1:
            d[getClassnameFromID(i)] = entry
    # print(d)
    pd.isnull(np.array([np.nan, -1], dtype=float))

    df_cm = pd.DataFrame(confusion_matrix, index=[getClassnameFromID(i) for i in range(confusion_matrix.shape[0])],
                         columns=[getClassnameFromID(i) for i in range(confusion_matrix.shape[0])], dtype=float)
    # plt.figure(figsize=(20, 20))
    # sn.heatmap(df_cm, annot=True)
    # plt.show()

    accuracy = 100 * correct / total

    print('\naccuracy is {} \n per_class_acc is: {} \n running_loss is {} \n f1 is {} \n '.format(accuracy, per_class_acc, running_loss, f1))
    # Print Accuracy
    # print('Accuracy: {}'.format(accuracy))
    return accuracy, per_class_acc, running_loss, f1, confusion_matrix

if __name__ == "__main__":
    if sys.argv[1] != None:
        file_name = sys.argv[1].split('.json')[0]
        input_dir = os.path.join(os.getcwd(), '../', 'data', file_name)
        csv_file_path = os.path.join(input_dir, file_name + '.csv')
        s_file_path = os.path.join(input_dir, file_name + '.json')
        # csv_length = pd.read_csv(csv_file_path).shape[0]
        ActivityIdList = config['ActivityIdList']
        train(csv_file_path, ActivityIdList)



