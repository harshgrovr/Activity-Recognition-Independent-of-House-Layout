import torch
import torchvision.models as models
import torch.nn as nn
from torch.autograd import Variable


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        vgg16 = models.vgg16_bn(pretrained=False)
        layers = list(vgg16.features.children())[:-1]
        layers[0] = nn.Conv2d(22, 64, kernel_size=3, stride=1, padding=1)
        self.features = nn.Sequential(*layers)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 46 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, 18),
            nn.ReLU(),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(-1,  512 * 46 * 56)
        out = self.classifier(x)

        return out


class LSTMModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Initialize hidden state with zeros

        if torch.cuda.is_available():
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda()
            # Initialize cell state
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda()

        else:
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
            # Initialize cell state
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)

        # time steps
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # Index hidden state of last time step
        out = self.relu(self.fc(out[:, -1, :]))
        out = self.softmax(out)
        return out


