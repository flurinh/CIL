import torch.nn as nn
import torch.nn.functional as F
from unet_parts import *

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
        self.dec_conv5 = nn.ConvTranspose2d(1, 1, kernel_size=1, stride=1, padding=0)
        self.dec_conv6 = nn.ConvTranspose2d(1, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = F.relu(self.enc_conv1(x))
        x, i1 = self.enc_maxpool1(x)
        x = F.relu(self.enc_conv2(x))

        # x, i2 = self.enc_maxpool2(x)
        # x = self.dec_unpool2(x, i2)

        x = F.relu(self.dec_conv2(x))
        x = self.dec_unpool3(x, i1)
        x = F.relu(self.dec_conv3(x))
        x = F.relu(self.dec_conv4(x))
        x = F.relu(self.dec_conv5(x))
        x = F.softmax(self.dec_conv6(x))
        return x




class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return F.sigmoid(x)
