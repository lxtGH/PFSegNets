#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Implementation of Point Rend ResNet series.
# Author: Hao He(hehao2019@ia.ac.cn)

import torch
import torch.nn as nn
import torch.nn.functional as F
from network.nn.mynn import initialize_weights, Norm2d, Upsample
from network import resnet_d as Resnet_Deep
from network.resnext import resnext50_32x4d, resnext101_32x8d, resnext101_32x8
from network.nn.sampling_points import sampling_points, point_sample
from network.nn.operators import _AtrousSpatialPyramidPoolingModule


class PointHead(nn.Module):
    def __init__(self, in_c=515, num_classes=3, k=3, beta=0.75):
        super().__init__()
        self.mlp = nn.Conv1d(in_c, num_classes, 1)
        self.k = k
        self.beta = beta

    def forward(self, x, res2, out):
        """
        1. Fine-grained features are interpolated from res2 for DeeplabV3
        2. During training we sample as many points as there are on a stride 16 feature map of the input
        3. To measure prediction uncertainty
           we use the same strategy during training and inference: the difference between the most
           confident and second most confident class probabilities.
        """
        if not self.training:
            return self.inference(x, res2, out)

        points = sampling_points(out, x.shape[-1] // 16, self.k, self.beta)

        coarse = point_sample(out, points, align_corners=False)
        fine = point_sample(res2, points, align_corners=False)

        feature_representation = torch.cat([coarse, fine], dim=1)

        rend = self.mlp(feature_representation)

        return {"rend": rend, "points": points}

    @torch.no_grad()
    def inference(self, x, res2, out):
        """
        During inference, subdivision uses N=8096
        (i.e., the number of points in the stride 16 map of a 1024×2048 image)
        """
        num_points = 8096

        while out.shape[-1] != x.shape[-1]:
            out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=True)

            points_idx, points = sampling_points(out, num_points, training=self.training)

            coarse = point_sample(out, points)
            fine = point_sample(res2, points)

            feature_representation = torch.cat([coarse, fine], dim=1)

            rend = self.mlp(feature_representation)

            B, C, H, W = out.shape
            points_idx = points_idx.unsqueeze(1).expand(-1, C, -1)
            out = (out.reshape(B, C, -1)
                      .scatter_(2, points_idx, rend)
                      .view(B, C, H, W))

        return {"fine": out}


class PointRendDeeplabv3p(nn.Module):
    """
    Implement DeepLabV3 model
    A: stride8
    B: stride16
    with skip connections
    """

    def __init__(self, num_classes, trunk='seresnext-50', criterion=None, variant='D',
                 skip='m1', skip_num=48, k=3, beta=0.75):
        super(PointRendDeeplabv3p, self).__init__()
        self.criterion = criterion
        self.variant = variant
        self.skip = skip
        self.skip_num = skip_num

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
                    if trunk == 'resnext-101-32x8d' or trunk == 'resnext-50-32x4d':
                        m.kernel_size = (1, 1)
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    if trunk == 'resnext-101-32x8d' or trunk == 'resnext-50-32x4d':
                        m.kernel_size = (1, 1)
                    m.stride = (1, 1)
        elif self.variant == 'D16':
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    if trunk == 'resnext-101-32x8d' or trunk == 'resnext-50-32x8d':
                        m.kernel_size = (1, 1)
                    m.stride = (1, 1)
        else:
            print('Not using dilation')

        if self.variant == 'D':
            self.aspp = _AtrousSpatialPyramidPoolingModule(2048, 256, output_stride=8)
        elif self.variant == 'D16':
            self.aspp = _AtrousSpatialPyramidPoolingModule(2048, 256, output_stride=16)

        if self.skip == 'm1':
            self.bot_fine = nn.Conv2d(256, self.skip_num, kernel_size=1, bias=False)
        elif self.skip == 'm2':
            self.bot_fine = nn.Conv2d(512, self.skip_num, kernel_size=1, bias=False)
        else:
            raise Exception('Not a valid skip')

        self.bot_aspp = nn.Conv2d(1280, 256, kernel_size=1, bias=False)

        self.final = nn.Sequential(
            nn.Conv2d(256 + self.skip_num, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False))

        self.point_head = PointHead(in_c=512+num_classes, num_classes=num_classes, k=k, beta=beta)

        initialize_weights(self.aspp)
        initialize_weights(self.bot_aspp)
        initialize_weights(self.bot_fine)
        initialize_weights(self.final)
        initialize_weights(self.point_head)

    def forward(self, x, gts=None):

        x_size = x.size()  # 800
        x0 = self.layer0(x)  # 400
        x1 = self.layer1(x0)  # 400
        x2 = self.layer2(x1)  # 100
        x3 = self.layer3(x2)  # 100
        x4 = self.layer4(x3)  # 100
        xp = self.aspp(x4)

        dec0_up = self.bot_aspp(xp)
        if self.skip == 'm1':
            dec0_fine = self.bot_fine(x1)
            dec0_up = Upsample(dec0_up, x1.size()[2:])
        else:
            dec0_fine = self.bot_fine(x2)
            dec0_up = Upsample(dec0_up, x2.size()[2:])

        dec0 = [dec0_fine, dec0_up]
        dec0 = torch.cat(dec0, 1)
        dec1 = self.final(dec0)
        ret = {}
        ret['image'] = x
        ret['coarse'] = dec1
        ret.update(self.point_head(x, x2, ret['coarse']))
        if self.training:
            return self.criterion(ret, gts)

        return ret['fine']


class PointRendDeeplabv3(nn.Module):
    """
    Implement DeepLabV3 model
    A: stride8
    B: stride16
    with skip connections
    """

    def __init__(self, num_classes, trunk='seresnext-50', criterion=None, variant='D',
                 skip='m1', skip_num=48, k=3, beta=0.75):
        super(PointRendDeeplabv3, self).__init__()
        self.criterion = criterion
        self.variant = variant
        self.skip = skip
        self.skip_num = skip_num

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
                    if trunk == 'resnext-101-32x8d' or trunk == 'resnext-50-32x4d':
                        m.kernel_size = (1, 1)
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    if trunk == 'resnext-101-32x8d' or trunk == 'resnext-50-32x4d':
                        m.kernel_size = (1, 1)
                    m.stride = (1, 1)
        elif self.variant == 'D16':
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    if trunk == 'resnext-101-32x8d' or trunk == 'resnext-50-32x8d':
                        m.kernel_size = (1, 1)
                    m.stride = (1, 1)
        else:
            print('Not using dilation')

        if self.variant == 'D':
            self.aspp = _AtrousSpatialPyramidPoolingModule(2048, 256, output_stride=8)
        elif self.variant == 'D16':
            self.aspp = _AtrousSpatialPyramidPoolingModule(2048, 256, output_stride=16)

        # if self.skip == 'm1':
        #     self.bot_fine = nn.Conv2d(256, self.skip_num, kernel_size=1, bias=False)
        # elif self.skip == 'm2':
        #     self.bot_fine = nn.Conv2d(512, self.skip_num, kernel_size=1, bias=False)
        # else:
        #     raise Exception('Not a valid skip')

        self.bot_aspp = nn.Sequential(nn.Conv2d(1280, 256, kernel_size=1, bias=False),
                                      Norm2d(256))

        self.final = nn.Conv2d(256, num_classes, kernel_size=1, bias=False)

        self.point_head = PointHead(in_c=512+num_classes, num_classes=num_classes, k=k, beta=beta)


    def forward(self, x, gts=None):

        x_size = x.size()  # 800
        x0 = self.layer0(x)  # 400
        x1 = self.layer1(x0)  # 400
        x2 = self.layer2(x1)  # 100
        x3 = self.layer3(x2)  # 100
        x4 = self.layer4(x3)  # 100
        xp = self.aspp(x4)

        dec0_up = self.bot_aspp(xp)
        # if self.skip == 'm1':
        #     dec0_fine = self.bot_fine(x1)
        #     dec0_up = Upsample(dec0_up, x1.size()[2:])
        # else:
        #     dec0_fine = self.bot_fine(x2)
        #     dec0_up = Upsample(dec0_up, x2.size()[2:])

        # dec0 = [dec0_fine, dec0_up]
        # dec0 = torch.cat(dec0, 1)
        dec1 = self.final(dec0_up)
        ret = {}
        ret['image'] = x
        ret['coarse'] = dec1
        ret.update(self.point_head(x, x2, ret['coarse']))
        if self.training:
            return self.criterion(ret, gts)

        return ret['fine']


def DeepR50V3PlusDPointRend_m1_deeply(num_classes, criterion):
    """
    ResNet-50 Based Network
    """
    return PointRendDeeplabv3p(num_classes, trunk='resnet-50-deep', criterion=criterion, variant='D', skip='m1',
                               k=3, beta=0.75)

def DeepR50V3PlusD16PointRend_m1_deeply(num_classes, criterion):
    """
    ResNet-50 Based Network
    """
    return PointRendDeeplabv3p(num_classes, trunk='resnet-50-deep', criterion=criterion, variant='D16', skip='m1',
                               k=3, beta=0.75)

def DeepR50V3DPointRend_m1_deeply(num_classes, criterion):
    """
    ResNet-50 Based Network
    """
    return PointRendDeeplabv3(num_classes, trunk='resnet-50-deep', criterion=criterion, variant='D', skip='m1',
                               k=3, beta=0.75)

def DeepR50V3D16PointRend_m1_deeply(num_classes, criterion):
    """
    ResNet-50 Based Network
    """
    return PointRendDeeplabv3(num_classes, trunk='resnet-50-deep', criterion=criterion, variant='D16', skip='m1',
                               k=3, beta=0.75)

def DeepR101V3PlusDPointRend_m1_deeply(num_classes, criterion):
    """
    ResNet-101 Based Network
    """
    return PointRendDeeplabv3p(num_classes, trunk='resnet-101-deep', criterion=criterion, variant='D', skip='m1',
                               k=3, beta=0.75)

def DeepRx101V3PlusDPointRend_m1_deeply(num_classes, criterion):
    """
    ResNet-101 Based Network
    """
    return PointRendDeeplabv3p(num_classes, trunk='resnext-101-32x8d', criterion=criterion, variant='D', skip='m1',
                               k=3, beta=0.75)

def DeepRx101wodV3PlusDPointRend_m1_deeply(num_classes, criterion):
    """
    ResNet-101 Based Network
    """
    return PointRendDeeplabv3p(num_classes, trunk='resnext-101-32x8', criterion=criterion, variant='D', skip='m1',
                               k=3, beta=0.75)