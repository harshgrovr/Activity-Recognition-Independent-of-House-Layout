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
from pytorch.data_preprocessing import load

# Create sequence of input and output depending upon the window size
def create_inout_sequences(input_data, label, tw):
    inout_seq = []
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

    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data.iloc[i:i+tw, ~input_data.columns.isin(['activity', 'start', 'end'])]
        train_seq = train_seq.values
        train_label = label.iloc[i+tw:i+tw+1]
        train_label = train_label.values
        train_label = [x for x in ActivityIdList if x["name"] == train_label]
        train_label = train_label[0]['id']
        inout_seq.append((train_seq, train_label))
    return inout_seq

def train():
    learning_rate = 1e-3
    num_epochs = 20
    decay = 1e-6

    # Defining Model, Optimizer and Loss
    input_dim = 128
    hidden_dim = 200
    layer_dim = 2
    output_dim = 6
    seq_dim = 9
    total_num_iteration_for_LOOCV = 0
    total_acc_for_LOOCV = 0


    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
    if torch.cuda.is_available():
        model.cuda()
    print('cuda available: ', torch.cuda.is_available())
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)


    trainLoader, testLoader = load()
    training(num_epochs, trainLoader, optimizer, model, criterion, seq_dim, input_dim)

    evaluate(testLoader, model, seq_dim, input_dim)

# Train the Network
def training(num_epochs, trainLoader,  optimizer, model, criterion, seq_dim, input_dim, accumulation_steps=5):
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
            output = model(input)

            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(output, label)
            running_loss += loss
            loss = loss / accumulation_steps  # Normalize our loss (if averaged)
            loss.backward()  # Backward pass
            if (i) % accumulation_steps == 0:  # Wait for several backward steps
                optimizer.step()  # Now we can do an optimizer step
                optimizer.zero_grad()  # Reset gradients tensors

            if i % 10 == 9:  # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0


# Evaluate the network
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
            _, predicted = torch.max(outputs, 1)
            # Total number of labels
            total += labels.size(0)
            # Total correct predictions
            if torch.cuda.is_available():
                print(predicted.cpu(), labels.cpu())
                correct += (predicted.cpu() == labels.cpu()).sum()
            else:
                correct += (predicted == labels).sum()

    accuracy = 100 * correct / total

    # Print Accuracy
    print('Accuracy: {}'.format(accuracy))
    return accuracy
if __name__ == "__main__":
        train()




