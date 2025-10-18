
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*16*16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)


        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
