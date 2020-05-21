import h5py
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable
from src.dataLoader import datasetHDF5
import sys
from sklearn.model_selection import LeaveOneOut
import gc
import pandas as pd
import datetime
from datetime import datetime
from src.network import CNNModel, Net
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import config
from config.config import config

def train(file_name, input_dir, csv_file_path, json_file_path):

    num_epochs = 20
    total_num_iteration_for_LOOCV = 0
    total_acc_for_LOOCV = 0

    # Defining Model, Optimizer and Loss
    model = Net()
    if torch.cuda.is_available():
        model.cuda()
    print('cuda available: ', torch.cuda.is_available())

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['decay'])

    # Get names of all h5 files
    h5Files = [f.split('.h5')[:-1] for f in os.listdir(os.path.join(os.getcwd(), '../','data', file_name, 'h5py')) if f.endswith('.h5')]

    df = pd.read_csv(csv_file_path)


    loo = LeaveOneOut()
    # Just first h5 file stores object channel. Get Object channel from the first file
    h5Directory = os.path.join(os.getcwd(), '../', 'data', file_name, 'h5py')
    firstdate = df.iloc[0, 0]
    if not isinstance(firstdate, datetime):
      firstdate = datetime.strptime(firstdate, '%d-%b-%Y %H:%M:%S')
    firstdate = firstdate.strftime('%d-%b-%Y') + '.h5'
    with h5py.File(os.path.join(h5Directory, firstdate), 'r') as f:
     objectsChannel = f['object'].value
    # Apply Leave one out on all the h5 files(h5files list)
    for train_index, test_index in loo.split(h5Files):
        print('split: ', total_num_iteration_for_LOOCV)
        total_num_iteration_for_LOOCV += 1
        # Train
        for file_index in train_index:
            date = datetime.strptime(h5Files[file_index][0], '%d-%b-%Y')
            file_path = os.path.join(h5Directory, date.strftime('%d-%b-%Y') + '.h5')

            dataset = datasetHDF5(objectsChannel, curr_file_path = file_path)
            trainLoader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=8)

            training(num_epochs, trainLoader,  optimizer, model, criterion)

        # Test
        for file_index in test_index:
            print(file_index)
            closeH5Files()
            date = datetime.strptime(h5Files[file_index][0], '%d-%b-%Y')
            file_path = os.path.join(h5Directory, date.strftime('%d-%b-%Y') + '.h5')

            dataset = datasetHDF5(objectsChannel, curr_file_path=file_path)
            testLoader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=1)
            total_acc_for_LOOCV += evaluate(testLoader, model)

    print("Avg. Accuracy is {}".format(total_acc_for_LOOCV/total_num_iteration_for_LOOCV))

def training(num_epochs, trainLoader,  optimizer, model, criterion, loss =0, accumulation_steps = 32):
    for epoch in range(num_epochs):
        for i, (image, label) in enumerate(trainLoader):
            print(i)
            image = image.permute(0,3, 1, 2)  # from NHWC to NCHW

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
                optimizer.step()  # Now we can do an optimizer step
                optimizer.zero_grad()  # Reset gradients tensors
                print('epoch: {} Iteration: {} Loss: {}'.format(epoch, i, loss.item()))
            # uncomment if you wanna see gradients in each iterations as a graph
            # plot_grad_flow(model.named_parameters())

# Calculate Accuracy
def evaluate(testLoader, model):

    correct = 0
    total = 0
    # Iterate through test dataset
    with torch.no_grad():
        for images, labels in testLoader:
            images = images.permute(0, 3, 1, 2)  # from NHWC to NCHW
            # Load images to a Torch Variable
            if torch.cuda.is_available():
                images = images.float().cuda()
            else:
                images = images.float()

            # Forward pass only to get logits/output
            outputs = model(images)

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

def closeH5Files():
    for obj in gc.get_objects():  # Browse through ALL objects
        if isinstance(obj, h5py.File):  # Just HDF5 files
            try:
                obj.close()
            except:
                pass  # Was already closed

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




