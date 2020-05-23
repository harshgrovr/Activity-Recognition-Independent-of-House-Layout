import shutil

import h5py
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.dataLoader import datasetHDF5
import sys
from sklearn.model_selection import LeaveOneOut
import pandas as pd
import datetime
from datetime import datetime
import cv2

from src.lstmTraining import getIDFromClassName, getClassnameFromID
from src.network import CNNModel, Net
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import config
from config.config import config

def save_checkpoint(state, is_best, filename='checkpoint_cnn_lstm.pth.tar'):
    saved_model_path = os.path.join("../saved_model/cnn_lstm/model_best.pth.tar")
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, saved_model_path)


def train(file_name, input_dir, csv_file_path, json_file_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    total_num_iteration_for_LOOCV = 0
    total_acc_for_LOOCV = 0

    # Defining Model, Optimizer and Loss
    model = Net().to(device)

    print('cuda available: ', torch.cuda.is_available())

    df = pd.read_csv(csv_file_path)

    # Get class Frequency as a dictionary
    classFrequencyDict = df['activity'].value_counts().to_dict()
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
    class_weights = torch.tensor(classFrequenciesList).float().to(device)

    criterion = nn.CrossEntropyLoss(weight = class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['decay'])

    path = "../saved_model/cnn_lstm/model_best.pth.tar"
    start_epoch = 0
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch += checkpoint['epoch']
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(path, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(path))

    # Get names of all h5 files
    h5Files = [f.split('.h5')[:-1] for f in os.listdir(os.path.join(os.getcwd(), '../','data', file_name, 'h5py')) if f.endswith('.h5')]

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

        # Test
        file_index = test_index[-1]
        date = datetime.strptime(h5Files[file_index][0], '%d-%b-%Y')
        file_path = os.path.join(h5Directory, date.strftime('%d-%b-%Y') + '.h5')

        dataset = datasetHDF5(objectsChannel, curr_file_path=file_path)
        testLoader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=8)

        for file_index in train_index:
            date = datetime.strptime(h5Files[file_index][0], '%d-%b-%Y')
            file_path = os.path.join(h5Directory, date.strftime('%d-%b-%Y') + '.h5')

            dataset = datasetHDF5(objectsChannel, curr_file_path = file_path)
            trainLoader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=8)

            training(config['num_epochs'], trainLoader,  testLoader,optimizer, model, criterion, start_epoch)

        total_acc_for_LOOCV += evaluate(testLoader, model)
        break

    print("Avg. Accuracy is {}".format(total_acc_for_LOOCV))

def training(num_epochs, trainLoader, testLoader, optimizer, model, criterion, start_epoch, accumulation_steps = config['accumulation_steps']):
    writer = SummaryWriter(os.path.join('../logs', file_name, 'cnn_lstm'))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for epoch in range(num_epochs):
        print('epoch', epoch + start_epoch)
        running_loss = 0
        for i, (image, label) in enumerate(trainLoader):
            image = image.permute(0,3, 1, 2)  # from NHWC to NCHW
            if i % 20 == 19:
                print(i)
            image = image.float().to(device)
            label = label.to(device)

            # Forward pass to get output/logits
            output = model(image)

            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(output, label)
            running_loss += loss
            loss = loss / accumulation_steps  # Normalize our loss (if averaged)
            loss.backward()  # Backward pass
            if (i) % accumulation_steps == accumulation_steps -1 :  # Wait for several backward steps
                optimizer.step()  # Now we can do an optimizer step
                optimizer.zero_grad()  # Reset gradients tensors
        print('epoch: {} Loss: {}'.format(epoch, running_loss))

        if epoch % 10 == 0:
            accuracy, per_class_accuracy = evaluate(testLoader, model)

            # Logging mean class accuracy
            d = {}
            for i in range(len(per_class_accuracy)):
                d[getClassnameFromID(i)] = per_class_accuracy[i]

            writer.add_scalars('Mean_class_Accuracy', d, epoch + 1)

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, True)

            # Logging Gradients
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                writer.add_histogram(tag + '/grad', value.grad.data.cpu().numpy(), epoch + 1)

            # plot weights historgram
            for key in model.linear_layers.state_dict():
                writer.add_histogram(key, model.linear_layers.state_dict()[key].data.cpu().numpy(), epoch + 1)

        # Loggin loss
        writer.add_scalar('Loss', running_loss, epoch + 1)
        print('%d loss: %.3f' %
              (epoch + 1,  running_loss))

# Calculate Accuracy
def evaluate(testLoader, model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    nb_classes = config['output_dim']
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    correct = 0
    total = 0
    # Iterate through test dataset
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
            correct += (predicted.to(device) == labels.to(device)).sum()
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    print('per class accuracy')
    per_class_acc = confusion_matrix.diag() / confusion_matrix.sum(1)
    per_class_acc = per_class_acc.cpu().numpy()
    per_class_acc[np.isnan(per_class_acc)] = -1
    print(per_class_acc)
    # pd.isnull(np.array([np.nan, -1], dtype=float))

    # df_cm = pd.DataFrame(confusion_matrix, index=[getClassnameFromID(i) for i in range(confusion_matrix.shape[0])],
    #                      columns=[getClassnameFromID(i) for i in range(confusion_matrix.shape[0])], dtype=float)
    # plt.figure(figsize=(20, 20))
    # sn.heatmap(df_cm, annot=True)
    # plt.show()

    accuracy = 100 * correct / total

    # Print Accuracy
    print('Accuracy: {}'.format(accuracy))
    return accuracy, per_class_acc

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




