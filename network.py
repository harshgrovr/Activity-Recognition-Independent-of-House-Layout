
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(22, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Dropout(),
            nn.Conv2d(64, 128, 5),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(),
            nn.Conv2d(128, 256, 5),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 89 * 110, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 18),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(-1,  256 * 89 * 110)
        out = self.classifier(x)

        return out



