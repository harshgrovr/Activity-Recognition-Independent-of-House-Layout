import shutil
from pathlib import Path

import h5py
import os
import torch
import torchvision
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from src.dataLoader import datasetHDF5, datasetFolder
import sys

from sklearn.model_selection import LeaveOneOut
import pandas as pd
import datetime
from datetime import datetime
import cv2

from src.lstmTraining import getIDFromClassName, getClassnameFromID
from src.lstmTrainingWithLOOCV import getUniqueStartIndex, getStartAndEndIndex
from src.network import CNNLSTM, LSTMModel, Network
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

def create_inout_sequences(input_data, tw ):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data.iloc[i:i+tw, input_data.columns.isin(['time_of_the_day'])]
        train_seq = train_seq.values
        train_label = input_data.iloc[i:i+tw, input_data.columns.isin(['activity'])]
        train_label = train_label.values
        (values, counts) = np.unique(train_label, return_counts=True)
        ind = np.argmax(counts)
        train_seq = train_seq[ind]
        train_label = values[ind]
        inout_seq.append([train_seq, train_label])
    return inout_seq

def train(file_name, input_dir, csv_file_path, json_file_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    total_num_iteration_for_LOOCV = 0
    avg_per_class_accuracy = 0
    avg_acc = 0
    avg_f1 = 0

    # Defining Model, Optimizer and Loss
    model = CNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['decay'])


    images_path = os.path.join(os.getcwd(), '../', 'data', file_name, 'images')


    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True, num_workers=8)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testLoader = torch.utils.data.DataLoader(testset, batch_size=64,
                                             shuffle=False, num_workers=8)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    for epoch in range(config['num_epochs']):
        print('epoch', epoch)
        running_loss = 0
        nb_classes = config['output_dim']
        confusion_matrix = torch.zeros(nb_classes, nb_classes)
        for i, (image, label) in enumerate(trainLoader):
            image = image.float().to(device)
            label = label.to(device)
            optimizer.zero_grad()
            # image = image.permute(0, 3, 1, 2)  # from NHWC to NCHW
            output = model(image)
            loss = criterion(output, label)
            running_loss += loss.item()
            loss.backward()  # Backward pass
            optimizer.step()  # Now we can do an optimizer step

        if epoch % 5 == 0:
            correct = 0
            total = 0
            batches = 0
            f1 = 0
            total_acc = 0
            total_f1 =0
            with torch.no_grad():
                for i, (image, label) in enumerate(trainLoader):
                    image = image.float().to(device)
                    label = label.to(device)
                    # image = image.permute(0, 3, 1, 2)  # from NHWC to NCHW
                    output = model(image)
                    _, predicted = torch.max(output.data, 1)
                    total += label.size(0)

                    if torch.cuda.is_available():
                        correct += (predicted.cpu() == label.cpu()).sum().item()
                    else:
                        correct += (predicted == label).sum().item()
                    for t, p in zip(label.view(-1), predicted.view(-1)):
                        confusion_matrix[t.long(), p.long()] += 1
                    f1 += f1_score(label.cpu(), predicted.cpu(), average='macro')
                    if (i == 20):
                        print('f1 for current batch is',f1/label.size(0))
                    # print(predicted, label)

                    batches = i
                f1 = f1 / batches
                per_class_acc = confusion_matrix.diag() / confusion_matrix.sum(1)
                per_class_acc = per_class_acc.cpu().numpy()
                per_class_acc[np.isnan(per_class_acc)] = 0
                d = {}
                for i in range(len(per_class_acc)):
                    d[getClassnameFromID(i)] = per_class_acc[i]
                # print(correct.cpu().item(), total)
                accuracy = 100 * correct / total
                print('\n\n ******** TRAIN ************** \n\n')
                print('Train Epoch: {}. train Loss: {}. train Accuracy: {}, . train-pca: {}.'.format(epoch, running_loss, accuracy, d))


                for i, (image, label) in enumerate(testLoader):
                    image = image.float().to(device)
                    label = label.to(device)
                    # image = image.permute(0, 3, 1, 2)  # from NHWC to NCHW
                    output = model(image)
                    _, predicted = torch.max(output.data, 1)
                    total += label.size(0)

                    if torch.cuda.is_available():
                        correct += (predicted.cpu() == label.cpu()).sum()
                    else:
                        correct += (predicted == label).sum()
                    for t, p in zip(label.view(-1), predicted.view(-1)):
                        confusion_matrix[t.long(), p.long()] += 1
                    if (i == 20):
                        print('f1 for current batch is',f1/label.size(0))
                    f1 += f1_score(label.cpu(), predicted.cpu(), average='macro')

                    # print(predicted, label)

                    batches = i
                f1 = f1 / batches
                confusion_matrix = confusion_matrix/ batches
                per_class_acc = confusion_matrix.diag() / confusion_matrix.sum(1)
                per_class_acc = per_class_acc.cpu().numpy()
                per_class_acc[np.isnan(per_class_acc)] = 0
                d = {}
                for i in range(len(per_class_acc)):
                    d[getClassnameFromID(i)] = per_class_acc[i]
                # print(correct.cpu().item(), total)
                accuracy = 100 * correct / total
                avg_per_class_accuracy += per_class_acc
                avg_acc += accuracy
                avg_f1 += f1
                print('\n\n ******** TEST ************** \n\n')
                print('Test-epoch: {}. test-Loss: {}. test-Accuracy: {}, . test-pca: {}.'.format(epoch, running_loss, accuracy, d))
    total_num_iteration_for_LOOCV += 1
    # print('\n\n ******** AVERAGE ************** \n\n')
    # print('avg_per_class_accuracy: {}. avg_acc: {}. avg_f1: {}'.format(avg_per_class_accuracy/total_num_iteration_for_LOOCV, avg_acc/total_num_iteration_for_LOOCV, avg_f1/total_num_iteration_for_LOOCV))
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

