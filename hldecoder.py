
import math
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from hlconv import hlconv
def normlayer(x):
    return nn.GroupNorm(2,x)
class DeepLabDecoder(nn.Module):
    def __init__(self, conv_operator='std_conv', kernel_size=5, batch_norm=normlayer):
        super(DeepLabDecoder, self).__init__()
        hlConv2d = hlconv[conv_operator]
        BatchNorm2d = batch_norm

        self.first_dconv = nn.Sequential(
            nn.Conv2d(24, 48, 1, bias=False),
            BatchNorm2d(48),
            nn.ReLU6(inplace=True)
        )

        self.last_dconv = nn.Sequential(
            hlConv2d(304, 256, kernel_size, 1, BatchNorm2d),
            hlConv2d(256, 256, kernel_size, 1, BatchNorm2d)
        )

        self._init_weight()

    def forward(self, l, l_low):
        l_low = self.first_dconv(l_low)
        l = F.interpolate(l, size=l_low.size()[2:], mode='bilinear', align_corners=True)
        l = torch.cat((l, l_low), dim=1)
        l = self.last_dconv(l)
        return l

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)


# max-pooling indices-guided decoding
class IndexedDecoder(nn.Module):
    def __init__(self, inp, oup, conv_operator='std_conv', kernel_size=5, batch_norm=normlayer):
        super(IndexedDecoder, self).__init__()
        hlConv2d = hlconv[conv_operator]
        BatchNorm2d = batch_norm

        self.upsample = nn.MaxUnpool2d((2, 2), stride=2)
        # inp, oup, kernel_size, stride, batch_norm
        self.dconv = hlConv2d(inp, oup, kernel_size, 1, BatchNorm2d)

        self._init_weight()

    def forward(self, l_encode, l_low, indices=None):
        l_encode = self.upsample(l_encode, indices) if indices is not None else l_encode
        l_encode = torch.cat((l_encode, l_low), dim=1)
        return self.dconv(l_encode)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)


    def visualize(self, x, indices=None):
        l = self.upsample(x, indices) if indices is not None else x
        l = l.mean(dim=1).squeeze()
        l = l.cpu().numpy()
        l = l / l.max() * 255.
        plt.figure()
        plt.imshow(l, cmap='viridis')
        plt.show()


class IndexedUpsamlping(nn.Module):
    def __init__(self, inp, oup, conv_operator='std_conv', kernel_size=5, batch_norm=normlayer):
        super(IndexedUpsamlping, self).__init__()
        self.oup = oup

        hlConv2d = hlconv[conv_operator]
        BatchNorm2d = batch_norm

        # inp, oup, kernel_size, stride, batch_norm
        self.dconv = hlConv2d(inp, oup, kernel_size, 1, BatchNorm2d)

        self._init_weight()

    def forward(self, l_encode, l_low, indices=None):
        _, c, _, _ = l_encode.size()
        if indices is not None:
            l_encode = indices * F.interpolate(l_encode, size=l_low.size()[2:], mode='nearest')
        l_cat = torch.cat((l_encode, l_low), dim=1)
        return self.dconv(l_cat)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)

    def visualize(self, x, indices=None):
        l = self.upsample(x, indices) if indices is not None else x
        l = l.mean(dim=1).squeeze()
        l = l.detach().cpu().numpy()
        l = l / l.max() * 255.
        plt.figure()
        plt.imshow(l, cmap='viridis')
        plt.axis('off')
        plt.show()