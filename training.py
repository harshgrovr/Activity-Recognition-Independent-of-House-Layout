import h5py
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable
from dataLoader import datasetHDF5
from network import CNNModel
import sys
from sklearn.model_selection import LeaveOneOut
import datetime
import gc

def train():
    learning_rate = 0.01
    num_epochs = 2
    total_num_iteration_for_LOOCV = 0
    total_acc_for_LOOCV = 0

    # Defining Model, Optimizer and Loss
    model = CNNModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Get names of all h5 files
    h5Files = [f.split('.h5')[:-1] for f in os.listdir(os.path.join(os.getcwd(), 'h5py')) if f.endswith('.h5')]

    loo = LeaveOneOut()
    h5Directory = os.path.join(os.getcwd(), 'h5py')

    # Apply Leave one out on all the files in h5files
    for train_index, test_index in loo.split(h5Files):
        total_num_iteration_for_LOOCV += 1
        # Train
        for file_index in train_index:
            closeH5Files()

            date = datetime.datetime.strptime(h5Files[file_index][0], '%d-%b-%Y')
            file_path = os.path.join(h5Directory, date.strftime('%d-%b-%Y') + '.h5')

            dataset = datasetHDF5(curr_file_path = file_path)
            trainLoader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

            training(num_epochs, trainLoader,  optimizer, model, criterion)

        # Test
        for file_index in test_index:
            closeH5Files()
            date = datetime.datetime.strptime(h5Files[file_index][0], '%d-%b-%Y')
            file_path = os.path.join(h5Directory, date.strftime('%d-%b-%Y') + '.h5')

            dataset = datasetHDF5(curr_file_path=file_path)
            testLoader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
            total_acc_for_LOOCV += evaluate(testLoader, model)

    print("Avg. Accuracy is {}".format(total_acc_for_LOOCV/total_num_iteration_for_LOOCV))

def training(num_epochs, trainLoader,  optimizer, model, criterion):
    for epoch in range(num_epochs):
        for i, (image, label) in enumerate(trainLoader):
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

            # Print Epoch and Loss
            if epoch % 5 == 0:
                print('epoch: {} Loss: {}'.format(epoch, loss))


# Calculate Accuracy
def evaluate(testLoader, model):

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
        train()




