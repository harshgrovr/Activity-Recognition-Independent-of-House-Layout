import torch
import torchvision.models as models
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torch.nn import init

from Better_LSTM_PyTorch.better_lstm import LSTM
from config.config import config

import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.model = models.vgg16_bn(pretrained=False)
        for param in self.model.parameters():
            param.requires_grad = False
        layers = list(self.model.features.children())
        layers[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        for param in layers[0].parameters():
            param.requires_grad = False

        self.model.features = nn.Sequential(*layers)
        n_inputs = self.model.classifier[6].in_features
        # Add on classifier
        self.model.classifier[6] = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, config['output_dim']))

        print(self.model)

    def forward(self, x):
        x = self.model(x)
        return x


class LSTMModel(nn.Module):
    def init_hidden(self, batch_size):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    def init_weights(self, m):
        if type(m) == nn.Conv2d or type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)
        return m

    def initLstmWeights(self, lstm):
        for name, param in lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal(param)
        return lstm

    def __init__(self):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = config['hidden_dim']

        # Number of hidden layers
        self.layer_dim = config['layer_dim']
        self.output_dim = config['output_dim']
        self.input_dim = config['input_dim']
        self.seq_dim = config['seq_dim']
        self.lstm = None

        # VGG
        vgg16 = models.vgg16_bn(pretrained=False)
        layers = list(vgg16.features.children())

        # Freeze all layers except top 3 cnn
        for layer in layers[:-11]:
            if not isinstance(layer, (nn.ReLU, nn.MaxPool2d)):
                layer.weight.requires_grad = False
                layer.bias.requires_grad = False

        # for i, layer in enumerate(layers[-11:]):
        #     if not isinstance(layer, (nn.ReLU, nn.MaxPool2d)):
        #         print(layer.weight)
        #         break

        # Initialize Unfreezed Conv layers' biases and weights
        for i, layer in enumerate(layers[-11:]):
            if not isinstance(layer, (nn.ReLU, nn.MaxPool2d)):
                self.init_weights(layer)

        # for i, layer in enumerate(layers[-11:]):
        #     if not isinstance(layer, (nn.ReLU, nn.MaxPool2d)):
        #         print(layer.weight)
        #         break

        # Change Initial Layer and add 1 * 1 Cnn as the last conv layer
        layers[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        layers[0].weight.requires_grad = False
        layers[0].bias.requires_grad = False
        layers[-3] = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=1)
        layers[-2] = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.sensorCNN = nn.Sequential(*layers)
        self.objectCNN = nn.Sequential(*layers)
        self.imageCNN = nn.Sequential(*layers)
        self.oneByOneCNN = nn.Conv2d(384, config['seq_dim'], kernel_size=1)

        # # Check model train params, like requires gradient etc
        # vgg16 = self.sensorCNN
        # for layer in list(vgg16.children()):
        #     print(layer)
        #     if not isinstance(layer, (nn.ReLU, nn.MaxPool2d)):
        #         print(layer.weight.requires_grad)
        #
        #     # Find total parameters and trainable parameters
        # total_params = sum(p.numel() for p in vgg16.parameters())
        # print('total parameters', total_params)
        # total_trainable_params = sum(
        #     p.numel() for p in vgg16.parameters() if p.requires_grad)
        # print('total_trainable_params', total_trainable_params)

        # Readout layer
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.fc = self.init_weights(self.fc)
        self.relu = nn.ReLU()
        self.lstm = None


    def forward(self, image, label, objectChannel, sensorChannel, textData, hn, cn):

        # Get original Batch_Size
        batch_size, C, H, W = image.size()
        batch_size = int(batch_size/config['seq_dim'])

        # Pass Image, Object and Input to VGG
        objectOutput = self.objectCNN(objectChannel)
        imageOutput = self.imageCNN(image)
        sensorOutput = self.sensorCNN(sensorChannel)


        # Concatenate the output
        concatOutput = torch.cat((imageOutput, objectOutput, sensorOutput), dim=1)
        concatOutput = self.oneByOneCNN(concatOutput)
        concatOutput = concatOutput.view(batch_size, config['seq_dim'], -1)
        concatOutput = torch.cat((concatOutput, textData), dim=2)
        self.input_dim = concatOutput.size(2)

        # Create LSTM model based upon concatSize
        if self.lstm is None:
            print('model initialized')
            print('input_dim',self.input_dim)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, batch_first=True, dropout= 0.4).to(device)

            # Initialize LSTM weights and biases
            self.lstm = self.initLstmWeights(self.lstm)


        output, (hn, cn) = self.lstm(concatOutput, (hn, cn))

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


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=True, strides=1, dilation=2, padding=1, kernel_size=5):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=kernel_size, padding=padding, stride=strides, dilation=dilation)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=kernel_size, padding=2 * padding, dilation=(2 * dilation))
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_channels = 64
        strides = 1
        dilation = 2
        padding = 4
        kernel_size = 5
        input_channel = 3
        self.fc = nn.Linear(config['hidden_dim'], config['output_dim'])
        self.lstm = None

        b1 = Residual(input_channels=input_channel, num_channels=num_channels,
                      use_1x1conv=True, strides=strides, dilation=dilation, padding=padding, kernel_size=kernel_size)

        b2 = Residual(input_channels=num_channels, num_channels=2 * num_channels,
                      use_1x1conv=True, strides=strides, dilation=2 * dilation, padding=2 * padding,
                      kernel_size=kernel_size)

        b3 = Residual(input_channels=2 * num_channels, num_channels=4 * num_channels,
                      use_1x1conv=True, strides=strides, dilation=4 * dilation, padding=4 * padding,
                      kernel_size=kernel_size)

        self.net = nn.Sequential(b1, b2, b3, nn.AdaptiveMaxPool2d((2, 2)))
        self.apply(weight_init)

    def forward(self, x):
        x = self.net(x)
        x = x.view(config['batch_size'], config['seq_dim'], -1)
        if self.lstm is None:
            self.lstm = nn.LSTM(x.size(2), config['hidden_dim'], 1, batch_first=True).to(self.device)
            for param in self.lstm.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)

        h0 = torch.zeros(config['layer_dim'], x.size(0), config['hidden_dim']).to(self.device)
        # Initialize cell state
        c0 = torch.zeros(config['layer_dim'], x.size(0), config['hidden_dim']).to(self.device)

        output, (hn, cn) = self.lstm(x, (h0,c0))
        output = output[:, :, :]
        output = self.fc(output)

        return output, (hn, cn)
