import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # encoder
        self.enc_conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=4, padding=0)
        self.enc_maxpool1 = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.enc_conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=0)
        self.enc_maxpool2 = nn.MaxPool2d(2, stride=2, return_indices=True)

        self.dec_unpool2 = nn.MaxUnpool2d(2, stride=2)
        self.dec_conv2 = nn.ConvTranspose2d(16, 8, 3, stride=2, padding=0)
        self.dec_unpool3 = nn.MaxUnpool2d(2, stride=2)
        self.dec_conv3 = nn.ConvTranspose2d(8, 1, kernel_size=8, stride=4, padding=0)
        self.dec_conv4 = nn.ConvTranspose2d(1, 1, kernel_size=5, stride=1, padding=0)

    def forward(self, x):
        print(x.size())
        x = F.relu(self.enc_conv1(x))
        x, i1 = self.enc_maxpool1(x)
        x = F.relu(self.enc_conv2(x))

        x, i2 = self.enc_maxpool2(x)
        x = self.dec_unpool2(x, i2)

        x = F.relu(self.dec_conv2(x))
        x = self.dec_unpool3(x, i1)
        x = F.relu(self.dec_conv3(x))
        x = F.sigmoid(self.dec_conv4(x))
        print(x.size())
        return x
