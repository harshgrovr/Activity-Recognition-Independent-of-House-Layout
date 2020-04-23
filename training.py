
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable
from dataLoader import datasetHDF5
from network import CNNModel
import sys
import os

def train(fileName):

    model = CNNModel()
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    iter = 0
    # test_split = .2
    # dataset_size = 1000
    # indices = list(range(dataset_size))
    # split = int(np.floor(test_split * dataset_size))
    # train_indices, test_indices = indices[split:], indices[:split]
    # Creating PT data samplers and loaders:
    # train_sampler = SubsetRandomSampler(train_indices)
    # test_sampler = SubsetRandomSampler(test_indices)
    # #
    # trainLoader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, sampler=train_sampler)
    # testLoader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, sampler=test_sampler)
    train_file = os.path.join(os.getcwd(), 'h5py', '24-Jul-2009.h5')
    trainLoader = DataLoader(datasetHDF5(train_file), batch_size=1, shuffle=False)

    num_epochs = 1000
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(trainLoader):
            # Load images as Variable
            images = images.permute(0, 3, 1, 2)  # from NHWC to NCHW
            inputs = Variable(images)
            labels = Variable(labels)

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            # Forward pass to get output/logits
            outputs = model(inputs)

            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, labels)

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()

            iter += 1

            if iter % 2 == 0:
                # Calculate Accuracy
                correct = 0
                total = 0
                # Iterate through test dataset
                for images, labels in trainLoader:
                    images = images.permute(0, 3, 1, 2)  # from NHWC to NCHW
                    # Load images to a Torch Variable
                    images = Variable(images)

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
                print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.data[0], accuracy))


if __name__ == "__main__":
    if sys.argv[1] != None:
        file_name = sys.argv[1].split('.')[0]
        train(file_name)




