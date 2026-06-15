# Small LeNet-style CNN that classifies one MNIST digit.
# The same network is used by every framework so the comparison is fair.

import torch.nn as nn


class SharedCNN(nn.Module):
    def __init__(self, n: int = 10, with_softmax: bool = True):
        super().__init__()
        self.with_softmax = with_softmax
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),   # 1x28x28 -> 6x24x24
            nn.MaxPool2d(2, 2),   # 6x24x24 -> 6x12x12
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),  # 6x12x12 -> 16x8x8
            nn.MaxPool2d(2, 2),   # 16x8x8  -> 16x4x4
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, n),
        )
        if with_softmax:
            self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 16 * 4 * 4)
        x = self.classifier(x)
        if self.with_softmax:
            x = self.softmax(x)
        return x
