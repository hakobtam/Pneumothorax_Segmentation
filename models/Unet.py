# Code has been taken from https://github.com/milesial/Pytorch-UNet for building u-net architecture

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, batch_norm=True):
        super(DoubleConv, self).__init__()
        if batch_norm:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.conv(x.type(torch.FloatTensor))
        return x


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, up_mode=None):
        super(Up, self).__init__()

        if up_mode is not None:
            self.up = nn.Upsample(scale_factor=2, mode=up_mode, align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(
                    self, 
                    n_channels=1, 
                    n_classes=1,
                    num_filters=32,
                    up_mode = 'biliniar',
                    batch_norm=True,
                ):
        super(UNet, self).__init__()
        self.inc = InConv(n_channels, num_filters)
        self.down1 = Down(num_filters, num_filters * 2)
        self.down2 = Down(num_filters * 2, num_filters * 4)
        self.down3 = Down(num_filters * 2, num_filters * 8)
        self.down4 = Down(num_filters * 8, num_filters * 8)
        self.up1 = Up(num_filters * (8 + 8), num_filters * 4, up_mode)
        self.up2 = Up(num_filters * (4 + 4), num_filters * 2, up_mode)
        self.up3 = Up(num_filters * (2 + 2), num_filters, up_mode)
        self.up4 = Up(num_filters * (1 + 1), num_filters, up_mode)
        self.outc = OutConv(num_filters, n_classes)

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
        return self.outc(x)