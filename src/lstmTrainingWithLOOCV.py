import torch
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
    date = df['start'].iloc[test_index].item()
    index = df.index[df['start'] == date].tolist()
    # get start and end of this date
    return index[0], index[-1]

# Give all unique dates index
def getUniqueStartIndex(df):
    # this line converts the string object in Timestamp object
    df['start'] = [datetime.strptime(d, '%d-%b-%Y %H:%M:%S') for d in df["start"]]
    # extracting date from timestamp
    df['start'] = [datetime.date(d) for d in df['start']]
    s = df['start']
    return s[s.diff().dt.days != 0].index.values

def getIDFromClassName(train_label):
    ActivityIdList = [

        {"name": "idle", "id": 0},
        {"name": "leaveHouse", "id": 1},
        {"name": "useToilet", "id": 2},
        {"name": "takeShower", "id": 3},
        {"name": "brushTeeth", "id": 4},
        {"name": "goToBed", "id": 5},
        {"name": "getDressed", "id": 6},
        {"name": "prepareBreakfast", "id": 7},
        {"name": "prepareDinner", "id": 8},
        {"name": "getSnack", "id": 9},
        {"name": "getDrink", "id": 10},
        {"name": "loadDishwasher", "id": 11},
        {"name": "unloadDishwasher", "id": 12},
        {"name": "storeGroceries", "id": 13},
        {"name": "washDishes", "id": 14},
        {"name": "answerPhone", "id": 15},
        {"name": "eatDinner", "id": 16},
        {"name": "eatBreakfast", "id": 17}
    ]
    train_label = [x for x in ActivityIdList if x["name"] == train_label]
    return train_label[0]['id']

# Create sequence of input and output depending upon the window size
def create_inout_sequences(input_data, label, tw ):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data.iloc[i:i+tw, ~input_data.columns.isin(['activity', 'start', 'end'])]
        train_seq = train_seq.values
        train_label = label.iloc[i+tw:i+tw+1]
        train_label = train_label.values
        train_label = getIDFromClassName(train_label)
        inout_seq.append((train_seq, train_label))
    return inout_seq

def train(file_name, ActivityIdList):
    total_num_iteration_for_LOOCV = 0
    total_acc_for_LOOCV = 0

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
    uniqueIndex = getUniqueStartIndex(df)
    loo = LeaveOneOut()

    print('Total splits: ', len(uniqueIndex) - 1)
    print('Total Epochs per split:', config['num_epochs'])
    for train_index, test_index in loo.split(uniqueIndex):
        model = LSTMModel(config['input_dim'], config['hidden_dim'], config['layer_dim'],config['output_dim'], config['seq_dim'])
        if torch.cuda.is_available():
            model.cuda()
        print('cuda available: ', torch.cuda.is_available())
        if torch.cuda.is_available():
            class_weights = torch.tensor(classFrequenciesList).float().cuda()
        else:
            class_weights = torch.tensor(classFrequenciesList).float()

        criterion = nn.CrossEntropyLoss(weight = class_weights)

        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['decay'])

        # Get start and end of test dataset
        start, end = getStartAndEndIndex(df, uniqueIndex[test_index])

        # df = df.astype('float32')

        # make dataframe for train, skip everything b/w test start and test end. rest everything is train.
        if start!=0:
            dfFrames = [df[:start - 1], df[end + 1:]]
            df1Frames = [df['activity'][:start - 1], df['activity'][end + 1:]]
            dfFrames = pd.concat(dfFrames)
            df1Frames = pd.concat(df1Frames)
        else:
            dfFrames = df[end + 1:]
            df1Frames = df['activity'][end + 1:]
        # generate train sequence list based upon above dataframe.
        trainData = create_inout_sequences(dfFrames, df1Frames, config['seq_dim'])

        # Make Train DataLoader
        trainDataset = datasetCSV(trainData, config['seq_dim'])
        trainLoader = DataLoader(trainDataset, batch_size=config['batch_size'], shuffle=False, num_workers=8, drop_last=True)
        training(config['num_epochs'], trainLoader, optimizer, model, criterion, config['seq_dim'], config['input_dim'], config['batch_size'])

        # Generate Test DataLoader
        if start - config['seq_dim'] > 0:
            test_inputs = df[start - config['seq_dim']: end]
            test_labels = df['activity'][start - config['seq_dim']: end]
            testData = create_inout_sequences(test_inputs, test_labels, config['seq_dim'])
            testLoader = DataLoader(testData, batch_size=config['batch_size'], shuffle=False, num_workers=8, drop_last=True)
            total_acc_for_LOOCV += evaluate(testLoader, model, config['seq_dim'], config['input_dim'], len(ActivityIdList), batch_size=config['batch_size'])
    print('avg accuracy i: ,', (total_acc_for_LOOCV/(total_num_iteration_for_LOOCV - 1)), '%')

# Train the Network
def training(num_epochs, trainLoader,  optimizer, model, criterion, seq_dim, input_dim, batch_size,accumulation_steps=5):
    writer = SummaryWriter('../logs')
    for epoch in range(num_epochs):
        hn, cn = model.init_hidden(batch_size)
        flag = True
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

            if flag:  # beginning of sequence of data you want cell states from or when no longer need past cell state and starting fresh again
                flag = False
                hn, cn = model.init_hidden(batch_size)


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
                # flag = True


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
        ActivityIdList = [

            {"name": "idle", "id": 0},
            {"name": "leaveHouse", "id": 1},
            {"name": "useToilet", "id": 2},
            {"name": "takeShower", "id": 3},
            {"name": "brushTeeth", "id": 4},
            {"name": "goToBed", "id": 5},
            {"name": "getDressed", "id": 6},
            {"name": "prepareBreakfast", "id": 7},
            {"name": "prepareDinner", "id": 8},
            {"name": "getSnack", "id": 9},
            {"name": "getDrink", "id": 10},
            {"name": "loadDishwasher", "id": 11},
            {"name": "unloadDishwasher", "id": 12},
            {"name": "storeGroceries", "id": 13},
            {"name": "washDishes", "id": 14},
            {"name": "answerPhone", "id": 15},
            {"name": "eatDinner", "id": 16},
            {"name": "eatBreakfast", "id": 17}
        ]
        file_name = '../houseB'
        # file_name = sys.argv[1].split('.')[0]
        train(file_name, ActivityIdList)




