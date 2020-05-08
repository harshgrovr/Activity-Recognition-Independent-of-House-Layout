import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import sys
from sklearn.model_selection import LeaveOneOut
import pandas as pd
import datetime
from datetime import datetime
from network import LSTMModel
from dataLoader import datasetCSV

def getStartAndEndIndex(df, test_index):
    # this line converts the string object in Timestamp object
    date = df['start'].iloc[test_index].item()
    index = df.index[df['start'] == date].tolist()
    # get start and end of this date
    return index[0], index[-1]

def getUniqueStartIndex(df):
    # this line converts the string object in Timestamp object
    df['start'] = [datetime.strptime(d, '%d-%b-%Y %H:%M:%S') for d in df["start"]]

    # extracting date from timestamp
    df['start'] = [datetime.date(d) for d in df['start']]
    s = df['start']
    return s[s.diff().dt.days != 0].index.values

def init_weights(m):
    torch.nn.init.xavier_uniform(m.weight)
    torch.nn.init.xavier_uniform(m.weight)
    m.bias.data.fill_(0.01)



def train(file_name):

    learning_rate = 1e-2
    num_epochs = 200
    total_num_iteration_for_LOOCV = 0
    total_acc_for_LOOCV = 0
    decay = 1e-6

    # Defining Model, Optimizer and Loss
    input_dim = 22
    hidden_dim = 100
    layer_dim = 2
    output_dim = 18
    seq_dim = 1

    csv_file = file_name + '.csv'
    df = pd.read_csv(csv_file)

    uniqueIndex = getUniqueStartIndex(df)

    loo = LeaveOneOut()
    print('Total splits: ', len(uniqueIndex)-1)
    print('Total Epochs per split:', num_epochs)

    for train_index, test_index in loo.split(uniqueIndex):
        model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)

        if torch.cuda.is_available():
            model.cuda()
        print('cuda available: ', torch.cuda.is_available())

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)

        # get start and end value to ignore
        start, end = getStartAndEndIndex(df, uniqueIndex[test_index])

        #Define Train and Test Loaders
        trainDataset = datasetCSV(df, 0)
        trainLoader = DataLoader(trainDataset, batch_size=128, shuffle=True, num_workers=8)
        testDataset = datasetCSV(df[start:end], start)
        testLoader = DataLoader(testDataset, batch_size=128, shuffle=False, num_workers=8)

        print('split: ', total_num_iteration_for_LOOCV)
        total_num_iteration_for_LOOCV += 1
        # Train
        # continue on those start and end value
        training(num_epochs, trainLoader,  optimizer, model, criterion, start, end, seq_dim, input_dim)

        print("Testing")
        total_acc_for_LOOCV += evaluate(testLoader, model, seq_dim, input_dim)

    print("Avg. Accuracy is {}".format(total_acc_for_LOOCV/total_num_iteration_for_LOOCV))

def training(num_epochs, trainLoader,  optimizer, model, criterion,start, end, seq_dim, input_dim, accumulation_steps=128):
    for epoch in range(num_epochs):
        for i, (image, label) in enumerate(trainLoader):
            if start <= i <= end:
                continue

            image = image.view(-1, seq_dim, input_dim)
            if torch.cuda.is_available():
                image = image.float().cuda()
                label = label.cuda()
            else:
                image = image.float()
                label = label


            # Forward pass to get output/logits
            output = model(image)

            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(output, label)
            loss = loss / accumulation_steps  # Normalize our loss (if averaged)
            loss.backward()  # Backward pass
            if (i) % accumulation_steps == 0:  # Wait for several backward steps
                print('epoch: {} Iteration: {} Loss: {}'.format(epoch, i, loss.item()))
                optimizer.step()  # Now we can do an optimizer step
                model.zero_grad()  # Reset gradients tensors


# Calculate Accuracy
def evaluate(testLoader, model, seq_dim, input_dim):

    correct = 0
    total = 0
    # Iterate through test dataset
    with torch.no_grad():
        for input, labels in testLoader:
            input = input.view(-1, seq_dim, input_dim)
            # Load images to a Torch Variable
            if torch.cuda.is_available():
                input = input.float().cuda()
            else:
                input = input.float()

            # Forward pass only to get logits/output
            outputs = model(input)

            # Get predictions from the maximum value
            _, predicted = torch.max(outputs.data, 1)
            # Total number of labels
            total += labels.size(0)

            # Total correct predictions
            if torch.cuda.is_available():
                correct += (predicted.cpu() == labels.cpu()).sum()
            else:
                correct += (predicted == labels).sum()

    accuracy = 100 * correct / total

    # Print Accuracy
    print('Accuracy: {}'.format(accuracy))
    return accuracy


if __name__ == "__main__":
    if sys.argv[1] != None:
        file_name = sys.argv[1].split('.')[0]
        train(file_name)




