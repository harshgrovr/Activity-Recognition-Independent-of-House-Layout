import torch
import torchvision.models as models
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from Better_LSTM_PyTorch.better_lstm import LSTM
from config.config import config


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        vgg16 = models.vgg16_bn(pretrained=False)
        layers = list(vgg16.features.children())[:-1]
        layers[0] = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1)
        self.features = nn.Sequential(*layers)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 14 * 14, 128),
            nn.ReLU(),
            nn.Linear(128, 22),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(-1,  512 * 14 * 14)
        out = self.classifier(x)

        return out


class LSTMModel(nn.Module):
    def init_hidden(self, batch_size):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        if torch.cuda.is_available():
            print('****  init_hidden ********')
            hn = torch.zeros(self.layer_dim , self.batch_size, self.hidden_dim).to(device)
            # Initialize cell state
            cn = torch.zeros(self.layer_dim , self.batch_size, self.hidden_dim).to(device)
        else:
            hn = torch.zeros(self.layer_dim, self.batch_size, self.hidden_dim)
            # Initialize cell state
            cn = torch.zeros(self.layer_dim, self.batch_size, self.hidden_dim)
        return hn, cn

    def __init__(self):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = config['hidden_dim']

        # Number of hidden layers
        self.layer_dim = config['layer_dim']
        self.output_dim = config['output_dim']
        self.input_dim = config['input_dim']
        self.seq_dim = config['seq_dim']
        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = None
        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(5, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Readout layer
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x, hn, cn):
        # Initialize hidden state with zeros
        batch_size, C, H, W = x.size()
        batch_size = int(batch_size / config['seq_dim'])
        x = self.cnn_layers(x)
        x = x.view(batch_size, config['seq_dim'], -1)

        lstm_input_size = x.size(2)
        self.input_dim = lstm_input_size
        if self.lstm is None:
            print('model initialized')
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, batch_first=True, dropout= 0.4).to(device)

        output, (hn, cn) = self.lstm(x, (hn, cn))

        output = self.fc(output[:, :, :])
        return output, (hn, cn)



class HARmodel(nn.Module):
    """Model for human-activity-recognition."""
    def __init__(self,  num_classes):
        super().__init__()

        # Extract features, 1D conv layers
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, 5),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(64, 64, 5),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(64, 64, 5),
            nn.ReLU(),
            )
        # Classify output, fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64*218, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, num_classes),
            )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 64*218)
        out = self.classifier(x)

        return out


class CNNLSTM(nn.Module):

    def init_hidden(self, batch_size):
        self.batch_size = batch_size
        if torch.cuda.is_available():
            hn = torch.zeros(self.layer_dim , self.batch_size, self.hidden_dim).cuda()
            # Initialize cell state
            cn = torch.zeros(self.layer_dim, self.batch_size, self.hidden_dim).cuda()
        else:
            hn = torch.zeros(self.layer_dim, self.batch_size, self.hidden_dim)
            # Initialize cell state
            cn = torch.zeros(self.layer_dim, self.batch_size, self.hidden_dim)
        return hn, cn

    def __init__(self, num_classes):
        super(CNNLSTM, self).__init__()
        self.model = None
        self.layer_dim = config['layer_dim']
        self.hidden_dim = config['hidden_dim']
        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(5, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


    # Defining the forward pass NCHW
    def forward(self, x, hn, cn):
        batch_size, C, H, W = x.size()
        batch_size = int(batch_size/config['seq_dim'])
        x = self.cnn_layers(x)
        x = x.view(batch_size, config['seq_dim'], -1)
        lstm_input_size = x.size(2)
        if self.model is None:
            print('model initialized')
            self.model = LSTMModel()

        output, (hn, cn) = self.model((x, (hn,cn)))
        return output, (hn, cn)
