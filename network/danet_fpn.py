#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Implementation of DANet ResNet series.
# Author: Xiangtai(lxt@pku.edu.cn)

import torch
from torch import nn
import torch.nn.functional as F
from network import resnet_d as Resnet_Deep
from network.nn.mynn import initialize_weights, Norm2d, Upsample
from network.resnext import resnext50_32x4d, resnext101_32x8d, resnext101_32x8
from network.nn.operators import PAM_Module, CAM_Module, DABlock



def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv3x3_bn_relu(in_planes, out_planes, stride=1, normal_layer=nn.BatchNorm2d):
    return nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            normal_layer(out_planes),
            nn.ReLU(inplace=True),
    )


class FPNDecoder(nn.Module):

    def __init__(self, num_class, norm_layer=nn.BatchNorm2d, fpn_inplanes=[256, 512, 1024, 2048], fpn_dim=128,
                 fpn_dsn=False):
        super(FPNDecoder, self).__init__()

        self.fpn_dsn = fpn_dsn
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes:
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv2d(fpn_inplane, fpn_dim, 1),
                    norm_layer(fpn_dim),
                    nn.ReLU(inplace=False)
                )
            )
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        self.fpn_out_align = []
        self.dsn = []

        for i in range(len(fpn_inplanes) - 1):
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))


            if self.fpn_dsn:
                self.dsn.append(
                    nn.Sequential(
                        nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1),
                        norm_layer(fpn_dim),
                        nn.ReLU(),
                        nn.Dropout2d(0.1),
                        nn.Conv2d(fpn_dim, num_class, kernel_size=1, stride=1, padding=0, bias=True)
                    )
                )

        self.fpn_out = nn.ModuleList(self.fpn_out)
        self.fpn_out_align = nn.ModuleList(self.fpn_out_align)

        if self.fpn_dsn:
            self.dsn = nn.ModuleList(self.dsn)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1)
        )

    def forward(self, conv_out):
        aspp_out = conv_out[-1]
        f = aspp_out
        f = self.fpn_in[-1](f)
        fpn_feature_list = [f]
        out = []

        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)
            f = F.upsample(f, size=conv_x.size()[2:], mode='bilinear', align_corners=True)
            f = conv_x + f

            fpn_feature_list.append(self.fpn_out[i](f))
            if self.fpn_dsn:
                out.append(self.dsn[i](f))

        fpn_feature_list.reverse()  # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]

        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=True))

        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)

        return x


class DABlock_ours(nn.Module):
    def __init__(self, out_planes=512, in_planes=2048, norm_layer=nn.BatchNorm2d):
        super(DABlock_ours, self).__init__()
        self.sa = PAM_Module(in_planes)
        self.sc = CAM_Module(in_planes)
        inner_planes = in_planes // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_planes, inner_planes, 3, padding=1, bias=False),
                                    norm_layer(inner_planes),
                                    nn.ReLU(inplace=True))

        self.conv5c = nn.Sequential(nn.Conv2d(in_planes, inner_planes, 3, padding=1, bias=False),
                                    norm_layer(inner_planes),
                                    nn.ReLU(inplace=True))

        self.sa = PAM_Module(inner_planes)
        self.sc = CAM_Module(inner_planes)
        self.conv51 = nn.Sequential(nn.Conv2d(inner_planes, inner_planes, 3, padding=1, bias=False),
                                    norm_layer(inner_planes),
                                    nn.ReLU(inplace=True))
        self.conv52 = nn.Sequential(nn.Conv2d(inner_planes, inner_planes, 3, padding=1, bias=False),
                                    norm_layer(inner_planes),
                                    nn.ReLU(inplace=True))

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_planes, 1))
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_planes, 1))


    def forward(self, x, return_atten_map=False, hp=-1, wp=-1):
        feat1 = self.conv5a(x)
        if return_atten_map:
            return self.sa(feat1, return_atten_map, hp, wp)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)
        feat_sum = sa_output + sc_output

        return feat_sum


class DANet(nn.Module):
    """
    """
    def __init__(self, num_classes, trunk='seresnext-50', criterion=None, variant="D",
                 ):
        super(DANet, self).__init__()
        self.criterion = criterion
        self.variant = variant
        self.fpn_dsn = False
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

        self.da_head = DABlock_ours(512, norm_layer=Norm2d)
        self.point_flow_decoder = FPNDecoder(num_classes, fpn_inplanes=[256, 512, 1024, 512], norm_layer=Norm2d)

        initialize_weights(self.da_head)

    def forward(self, x, gts=None, return_atten_map=False, hp=-1, wp=-1, cal_inference_time=False):

        x_size = x.size()  # 800
        x0 = self.layer0(x)  # 400
        x1 = self.layer1(x0)  # 400
        x2 = self.layer2(x1)  # 100
        x3 = self.layer3(x2)  # 100
        x4 = self.layer4(x3)  # 100
        if return_atten_map:
            return self.da_head(x4, return_atten_map, hp, wp)

        da_out = self.da_head(x4)
        pointflow_input = [x1, x2, x3, da_out]

        x = self.point_flow_decoder(pointflow_input)
        main_out = Upsample(x, x_size[2:])
        if self.training:
            if not self.fpn_dsn:
                return self.criterion(main_out, gts)
            return self.criterion(x, gts)
        return main_out


def DANet_r101(num_classes, criterion):
    """
    ResNet-101 Based Network
    """
    return DANet(num_classes, trunk='resnet-101-deep', criterion=criterion, variant='D')


def DANet_r50(num_classes, criterion):
    """
    ResNet-101 Based Network
    """
    return DANet(num_classes, trunk='resnet-50-deep', criterion=criterion, variant='D')


def DANet_d16_r50(num_classes, criterion):
    """
    ResNet-101 Based Network
    """
    return DANet(num_classes, trunk='resnet-50-deep', criterion=criterion, variant='D16')


def DANet_rx101(num_classes, criterion):
    """
    ResNet-101 Based Network
    """
    return DANet(num_classes, trunk='resnext-101-32x8', criterion=criterion, variant='D')
