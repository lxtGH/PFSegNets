
import torch
import torch.nn as nn
from network import resnet_d as Resnet_Deep
from network.nn.mynn import initialize_weights, Norm2d, Upsample
from network.resnext import resnext50_32x4d, resnext101_32x8d, resnext101_32x8
import time

from network.extension.cc_attention import CrissCrossAttention


class _CCHead(nn.Module):
    def __init__(self, nclass, norm_layer=nn.BatchNorm2d):
        super(_CCHead, self).__init__()
        self.rcca = _RCCAModule(2048, 512, norm_layer)
        self.out = nn.Conv2d(512, nclass, 1)

    def forward(self, x):
        x = self.rcca(x)
        x = self.out(x)
        return x


class _RCCAModule(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(_RCCAModule, self).__init__()
        self.recurrence = 2
        inter_channels = in_channels // 4
        self.conva = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(True))
        self.cca = CrissCrossAttention(inter_channels)
        self.convb = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(True))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + inter_channels, out_channels, 3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.Dropout2d(0.1))

    def forward(self, x):
        out = self.conva(x)
        for i in range(self.recurrence):
            out = self.cca(out)
        out = self.convb(out)
        out = torch.cat([x, out], dim=1)
        out = self.bottleneck(out)

        return out


class CCNet(nn.Module):
    """
    Implement Encoding model
    A: stride8
    B: stride16
    with skip connections
    """
    def __init__(self, num_classes, trunk='resnet-50-deep', criterion=None, variant="D"):
        super(CCNet, self).__init__()
        self.criterion = criterion
        self.variant = variant

        if trunk == 'resnet-50-deep':
            resnet = Resnet_Deep.resnet50()
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        elif trunk == 'resnet-101-deep':
            resnet = Resnet_Deep.resnet101()
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        elif trunk == 'resnext-50-32x4d':
            resnet = resnext50_32x4d()
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        elif trunk == 'resnext-101-32x8d':
            resnet = resnext101_32x8d()
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        elif trunk == 'resnext-101-32x8':
            resnet = resnext101_32x8()
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        else:
            raise ValueError('Only support resnet50 and resnet101 for now')

        self.layer0 = resnet.layer0
        self.layer1, self.layer2, self.layer3, self.layer4 = \
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        if self.variant == 'D':
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        elif self.variant == 'D16':
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        else:
            print("Not using Dilation ")

        self.head = _CCHead(nclass=num_classes, norm_layer=Norm2d)

        initialize_weights(self.head)

    def forward(self, x, gts=None, cal_inference_time=False):

        start = time.time()
        x_size = x.size()  # 800
        x0 = self.layer0(x)  # 400
        x1 = self.layer1(x0)  # 400
        x2 = self.layer2(x1)  # 100
        x3 = self.layer3(x2)  # 100
        x4 = self.layer4(x3)  # 100
        out = self.head(x4)

        main_out = Upsample(out, x_size[2:])

        end = time.time()
        if cal_inference_time:
            return end-start

        if self.training:
            return self.criterion(main_out, gts)

        return main_out


def CCNet_v1_r50(num_classes, criterion, variant='D'):
    """
    ResNet-50 Based Network
    """
    return CCNet(num_classes, trunk='resnet-50-deep', criterion=criterion, variant=variant)


def CCNet_v1_r101(num_classes, criterion, variant='D'):
    """
    ResNet-50 Based Network
    """
    return CCNet(num_classes, trunk='resnet-101-deep', criterion=criterion, variant=variant)