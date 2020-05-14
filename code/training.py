import h5py
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable
from dataLoader import datasetHDF5
import sys
from sklearn.model_selection import LeaveOneOut
import gc
import pandas as pd
import datetime
from datetime import datetime
from network import CNNModel
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def train(file_name):

    learning_rate = 1e-3
    num_epochs = 20
    total_num_iteration_for_LOOCV = 0
    total_acc_for_LOOCV = 0
    decay = 1e-6


    # Defining Model, Optimizer and Loss
    model = CNNModel()
    if torch.cuda.is_available():
        model.cuda()
    print('cuda available: ', torch.cuda.is_available())

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=decay)

    # Get names of all h5 files
    h5Files = [f.split('.h5')[:-1] for f in os.listdir(os.path.join(os.getcwd(), 'h5py')) if f.endswith('.h5')]
    csv_file = file_name + '.csv'

    df = pd.read_csv(csv_file)


    loo = LeaveOneOut()
    h5Directory = os.path.join(os.getcwd(), 'h5py')
    firstdate = df.iloc[0, 0]
    if not isinstance(firstdate, datetime):
      firstdate = datetime.strptime(firstdate, '%d-%b-%Y %H:%M:%S')
    firstdate = firstdate.strftime('%d-%b-%Y') + '.h5'
    objectsChannel = h5py.File(os.path.join(h5Directory, firstdate), 'r')['object'].value
    # closeH5Files()
    # Apply Leave one out on all the files in h5files
    for train_index, test_index in loo.split(h5Files):
        print('split: ', total_num_iteration_for_LOOCV)
        total_num_iteration_for_LOOCV += 1
        # Train
        for file_index in train_index:
            # closeH5Files()
            date = datetime.strptime(h5Files[file_index][0], '%d-%b-%Y')
            file_path = os.path.join(h5Directory, date.strftime('%d-%b-%Y') + '.h5')

            dataset = datasetHDF5(objectsChannel, curr_file_path = file_path)
            trainLoader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=1)

            training(num_epochs, trainLoader,  optimizer, model, criterion)

        # Test
        for file_index in test_index:
            print(file_index)
            # closeH5Files()
            date = datetime.strptime(h5Files[file_index][0], '%d-%b-%Y')
            file_path = os.path.join(h5Directory, date.strftime('%d-%b-%Y') + '.h5')

            dataset = datasetHDF5(objectsChannel, curr_file_path=file_path)
            testLoader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=1)
            total_acc_for_LOOCV += evaluate(testLoader, model)

    print("Avg. Accuracy is {}".format(total_acc_for_LOOCV/total_num_iteration_for_LOOCV))

def training(num_epochs, trainLoader,  optimizer, model, criterion, loss =0, accumulation_steps = 32):
    for epoch in range(num_epochs):
        for i, (image, label) in enumerate(trainLoader):

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




def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

    plt.show()

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
        file_name = sys.argv[1].split('.')[0]
        train(file_name)




