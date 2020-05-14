import torch
import random
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader
import torch.nn as nn
import sys
from sklearn.model_selection import LeaveOneOut
import datetime
from datetime import datetime

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

# Create sequence of input and output depending upon the window size
def create_inout_sequences(input_data, tw ):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data.iloc[i:i+tw, ~input_data.columns.isin(['activity', 'start', 'end'])]
        train_seq = train_seq.values
        train_label = input_data.iloc[i+tw:i+tw+1, input_data.columns.isin(['activity'])]
        train_label = train_label.values
        train_label = getIDFromClassName(train_label)
        inout_seq.append((train_seq, train_label))
    return inout_seq

def splitDatasetIntoTrainAndTest(df):
    testDataFrame = pd.DataFrame([])
    trainDataFrame = pd.DataFrame([])
    uniqueIndex = getUniqueStartIndex(df)
    for index in range(len(uniqueIndex)):
        # Get starting and ending index of a day
        start, end = getStartAndEndIndex(df, uniqueIndex[index])
        split = int(config['split_ratio'] * (end - start))
        trainDataFrame = trainDataFrame.append(df[start : start + split], ignore_index= True)
        testDataFrame = testDataFrame.append(df[start + split : end], ignore_index= True)

    return trainDataFrame, testDataFrame

def train(file_name, ActivityIdList):

    csv_file = file_name + '.csv'
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
    classFrequenciesList = classFrequenciesList/np.sum(classFrequenciesList)

    if torch.cuda.is_available():
        class_weights = torch.tensor(classFrequenciesList).float().cuda()
    else:
        class_weights = torch.tensor(classFrequenciesList).float()

    model = LSTMModel(config['input_dim'], config['hidden_dim'], config['layer_dim'],config['output_dim'], config['seq_dim'])
    if torch.cuda.is_available():
        model.cuda()
    print('cuda available: ', torch.cuda.is_available())

    criterion = nn.CrossEntropyLoss(weight = class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['decay'])

    # Split the data into test and train
    trainDataFrame, testDataFrame  = splitDatasetIntoTrainAndTest(df)

    training(config['num_epochs'],trainDataFrame, optimizer, model, criterion, config['seq_dim'], config['input_dim'], config['batch_size'], df)

    # Generate Test DataLoader
    testData = create_inout_sequences(testDataFrame, config['seq_dim'])
    testLoader = DataLoader(testData, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], drop_last=True)
    evaluate(testLoader, model, config['seq_dim'], config['input_dim'], len(ActivityIdList), batch_size = config['batch_size'])


# Train the Network
def training(num_epochs, trainDataFrame,  optimizer, model, criterion, seq_dim, input_dim, batch_size, df,accumulation_steps=5):
    minutesToRunFor = []
    randomDaySelected = []

    # for each random run select the random point and minute to run for
    # do this for each random point
    for number in getUniqueStartIndex(df):
        start, end = getStartAndEndIndex(df, number)
        number = random.sample(range(start, end - config['seq_dim']), 1)[0]
        randomDaySelected.append(number)
        minutesToRunFor.append(end - number)

    writer = SummaryWriter('../logs')

    for epoch in range(num_epochs):
        for j in range(len(randomDaySelected)):
            # generate train sequence list based upon above dataframe.
            time_to_start_from = randomDaySelected[j]
            minutes_to_run = minutesToRunFor[j]
            trainData = create_inout_sequences(trainDataFrame[time_to_start_from: time_to_start_from + minutes_to_run], config['seq_dim'])
            # Make Train DataLoader
            trainDataset = datasetCSV(trainData, config['seq_dim'])
            trainLoader = DataLoader(trainDataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'],
                                     drop_last=True)
            hn, cn = model.init_hidden(batch_size)
            running_loss = 0
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

                # Calculate Loss: softmax --> cross entropy loss
                loss = criterion(output, label)#weig pram
                running_loss += loss
                loss = loss / accumulation_steps  # Normalize our loss (if averaged)
                loss.backward()  # Backward pass
                if (i) % accumulation_steps == 0:  # Wait for several backward steps
                    optimizer.step()  # Now we can do an optimizer step
                    optimizer.zero_grad()  # Reset gradients tensors

                if i % 10 == 0:  # print every 10 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 10))
                    running_loss = 0.0

            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                writer.add_histogram(tag + '/grad', value.grad.data.cpu().numpy(), epoch + 1)
# Evaluate the network
def evaluate(testLoader, model, seq_dim, input_dim, nb_classes, batch_size):
    # Initialize the prediction and label lists(tensors)
    predlist = torch.zeros(0, dtype=torch.long, device='cpu')
    lbllist = torch.zeros(0, dtype=torch.long, device='cpu')

    correct = 0
    total = 0
    # Iterate through test dataset
    hn, cn = model.init_hidden(batch_size)
    with torch.no_grad():
        for input, labels in testLoader:
            hn.detach_()
            cn.detach_()
            input = input.view(-1, seq_dim, input_dim)
            # Load images to a Torch Variable
            if torch.cuda.is_available():
                input = input.float().cuda()
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
            # Append batch prediction results
            predlist = torch.cat([predlist, predicted.view(-1).cpu()])
            lbllist = torch.cat([lbllist, labels.view(-1).cpu()])

    # Confusion matrix
    # conf_mat = confusion_matrix(lbllist.numpy(), predlist.numpy())
    # print(conf_mat)

    # df_cm = pd.DataFrame(conf_mat, index=[i for i in range(conf_mat.shape[0])],
    #                      columns=[i for i in range(conf_mat.shape[0])])
    # plt.figure(figsize=(10, 7))
    # sn.heatmap(df_cm, annot=True)
    # plt.show()

    accuracy = 100 * correct / total

    # Print Accuracy
    print('Accuracy: {}'.format(accuracy))
    return accuracy
if __name__ == "__main__":
    if sys.argv[1] != None:
        ActivityIdList = config['ActivityIdList']
        file_name = '../houseB'
        # file_name = sys.argv[1].split('.')[0]
        train(file_name, ActivityIdList)




