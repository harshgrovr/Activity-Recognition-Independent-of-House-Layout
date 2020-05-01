import h5py
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from dataLoader import datasetHDF5
from network import CNNModel
import sys
from sklearn.model_selection import LeaveOneOut


def train(fileName):
    iter = 0
    test_split = .2
    csv_file = fileName + '.csv'
    model = CNNModel()
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    h5Files = [f.split('.h5')[:-1] for f in os.listdir(os.path.join(os.getcwd(), 'h5py')) if f.endswith('.h5')]
    loo = LeaveOneOut()
    for train_index, test_index in loo.split(h5Files):
        dataset = datasetHDF5(csvFileName = csv_file, h5Directory = os.path.join(os.getcwd(), 'h5py'))
        trainLoader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
        num_epochs = 2
        training(num_epochs, trainLoader, dataset, h5Files, optimizer, model, criterion, test_index)


def training(num_epochs, trainLoader, dataset, h5Files, optimizer, model, criterion, test_index):
    for epoch in range(num_epochs):
        for i, (image, label) in enumerate(trainLoader):
            if dataset.startDate.strftime('%d-%b-%Y') == h5Files[test_index[0]][0]:
                continue
            print(dataset.startDate.strftime('%d-%b-%Y'), h5Files[test_index[0]][0])
            image = image.permute(0, 3, 1, 2)  # from NHWC to NCHW
            image = Variable(image.float())
            label = Variable(label)

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get output/logits
            output = model(image)

            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(output, label)

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()

            # Print Loss
            # if epoch % 5 == 0:
            #     print('Iteration: {}. Loss: {}'.format(iter, loss.data[0]))
    testLoader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    evaluate(testLoader, model, test_index)

# Calculate Accuracy
def evaluate(testLoader, model, test_index):

    correct = 0
    total = 0
    # Iterate through test dataset
    for images, labels in testLoader:
        images = images.permute(0, 3, 1, 2)  # from NHWC to NCHW
        # Load images to a Torch Variable
        images = Variable(images.float())

        # Forward pass only to get logits/output
        outputs = model(images)

        # Get predictions from the maximum value
        _, predicted = torch.max(outputs.data, 1)

        # Total number of labels
        total += labels.size(0)

        # Total correct predictions
        correct += (predicted == labels).sum()

    accuracy = 100 * correct / total

    # Print Loss
    print('Accuracy: {}'.format(accuracy))


if __name__ == "__main__":
    if sys.argv[1] != None:
        file_name = sys.argv[1].split('.')[0]
        train(file_name)




