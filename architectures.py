import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self):
        # TODO: I am unsure how to adjust the stride, kernel and padding so that the input is the same as the output dimension.

        super(SimpleCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(16, 8, 3, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=1, padding=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=1, padding=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
