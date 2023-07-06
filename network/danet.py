#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Implementation of DANet ResNet series.
# Author: Xiangtai(lxt@pku.edu.cn)


from torch import nn
import torch
import torch.nn.functional as F
from network import resnet_d as Resnet_Deep
from network.nn.mynn import initialize_weights, Norm2d, Upsample
from network.resnext import resnext50_32x4d, resnext101_32x8d, resnext101_32x8
from network.nn.operators import PAM_Module, CAM_Module, DABlock
import time
from network.nn.contour_point_gcn import ContourPointGCN, ContourPointGCNV3, ContourPointGCNV5


class Edge_extractorWofirstext(nn.Module):
    def __init__(self, inplane, skip_num, norm_layer, num_groups=8):
        super(Edge_extractorWofirstext, self).__init__()
        self.skip_mum = skip_num
        self.pre_extractor = nn.Sequential(
            nn.Conv2d(inplane, inplane, kernel_size=3,
                      padding=1, groups=1, bias=False),
            norm_layer(inplane),
            nn.ReLU(inplace=True)
        )
        self.extractor = nn.Sequential(
            nn.Conv2d(inplane + skip_num, inplane, kernel_size=3,
                      padding=1, groups=8, bias=False),
            norm_layer(inplane),
            nn.ReLU(inplace=True)
        )

    def forward(self, aspp, layer1):            # 200        # 100
        seg_edge = torch.cat([F.interpolate(aspp, size=layer1.size()[2:], mode='bilinear',
                                            align_corners=True), layer1], dim=1)                      # 200
        seg_edge = self.extractor(seg_edge)          # 200
        seg_body = F.interpolate(aspp, layer1.size()[2:], mode='bilinear', align_corners=True) - seg_edge

        return seg_edge, seg_body

class DABlock_ours(nn.Module):
    def __init__(self, in_planes=2048, inner_planes=None, norm_layer=nn.BatchNorm2d):
        super(DABlock_ours, self).__init__()
        self.sa = PAM_Module(in_planes)
        self.sc = CAM_Module(in_planes)
        if inner_planes == None:
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


    def forward(self, x, return_atten_map=False, hp=-1, wp=-1):
        feat1 = self.conv5a(x)
        if return_atten_map:
            return self.sa(feat1, return_atten_map, hp, wp)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)

        feat_sum = sa_conv + sc_conv

        return feat_sum

class DANet_rdm_pgm_cascadewos_wofirst(nn.Module):
    """
    """

    def __init__(self, num_classes, trunk='seresnext-50', criterion=None, variant='D',
                 skip='m1', skip_num=48, edge_group=8, num_cascade=3, down_sample=False,
                 gcn_first=True, num_points=128, threshold=0.9, num_aspp=1):
        super(DANet_rdm_pgm_cascadewos_wofirst, self).__init__()

        self.criterion = criterion
        self.variant = variant
        self.skip = skip
        self.skip_num = skip_num
        self.num_cascade = num_cascade
        self.down_sample = down_sample
        self.gcn_first = gcn_first
        self.num_points = num_points
        self.threshold = threshold
        self.num_aspp = num_aspp

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

        self.bot_fine = nn.Conv2d(256, 48, kernel_size=1, bias=False)

        self.body_fines = nn.ModuleList()
        for i in range(self.num_cascade):
            inchannels = 2 ** (11 - i)
            self.body_fines.append(nn.Conv2d(inchannels, 48, kernel_size=1, bias=False))
        self.body_fuse = [nn.Conv2d(256 + 48, 256, kernel_size=1, bias=False) for _ in range(self.num_cascade)]
        self.body_fuse = nn.ModuleList(self.body_fuse)

        self.edge_extractors = [Edge_extractorWofirstext(256, norm_layer=Norm2d, skip_num=48, num_groups=8)
                                for _ in range(self.num_cascade)]
        self.edge_extractors = nn.ModuleList(self.edge_extractors)

        self.refines = [ContourPointGCN(256, self.num_points, self.threshold) for _ in range(self.num_cascade)]
        self.refines = nn.ModuleList(self.refines)

        self.edge_out_pre = [nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True)) for _ in range(self.num_cascade)]
        self.edge_out_pre = nn.ModuleList(self.edge_out_pre)
        self.edge_out = nn.ModuleList([nn.Conv2d(256, 1, kernel_size=1, bias=False)
                                       for _ in range(self.num_cascade)])

        self.body_out_pre = [nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True)) for _ in range(self.num_cascade)]
        self.body_out_pre = nn.ModuleList(self.body_out_pre)
        self.body_out = nn.ModuleList([nn.Conv2d(256, num_classes, kernel_size=1, bias=False)
                                       for _ in range(self.num_cascade)])

        self.final_seg_out_pre = [nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True)) for _ in range(self.num_cascade)]
        self.final_seg_out_pre = nn.ModuleList(self.final_seg_out_pre)
        self.final_seg_out = nn.ModuleList([nn.Conv2d(256, num_classes, kernel_size=1, bias=False)
                                            for _ in range(self.num_cascade)])

        self.da_head = DABlock_ours(num_classes, norm_layer=Norm2d)
        self.bot_aspp = nn.Conv2d(512, 256, kernel_size=1, bias=False)

    def forward(self, x, gts=None, return_atten_map=False, hp=-1, wp=-1, cal_inference_time=False):

        start = time.time()
        x_size = x.size()  # 800
        x0 = self.layer0(x)  # 400
        x1 = self.layer1(x0)  # 400
        x2 = self.layer2(x1)  # 100
        x3 = self.layer3(x2)  # 100
        x4 = self.layer4(x3)  # 100
        if return_atten_map:
            return self.da_head(x4, return_atten_map, hp, wp)
        da_out = self.da_head(x4)
        feats = []
        feats.append(x0)
        feats.append(x1)
        feats.append(x2)
        feats.append(x3)
        feats.append(x4)
        fine_size = feats[1].size()  # 200

        seg_edges = []
        seg_edge_outs = []
        seg_bodys = []
        seg_body_outs = []
        seg_finals = []
        seg_final_outs = []
        aspp = self.bot_aspp(da_out)  # 100
        final_fuse_feat = F.interpolate(aspp, size=fine_size[2:], mode='bilinear', align_corners=True)  # 200

        low_feat = self.bot_fine(feats[1])  # 200

        for i in range(self.num_cascade):
            if i == 0:
                last_seg_feat = aspp  # 100
            else:
                last_seg_feat = seg_finals[-1]  # 200
                if self.down_sample:
                    last_seg_feat = F.interpolate(last_seg_feat, size=aspp.size()[2:],
                                                  mode='bilinear', align_corners=True)  # 100

            seg_edge, seg_body = self.edge_extractors[i](last_seg_feat, low_feat)  # 200

            high_fine = F.interpolate(self.body_fines[i](feats[-(i + 1)]), size=fine_size[2:], mode='bilinear',
                                      align_corners=True)  # 200
            seg_body = self.body_fuse[i](torch.cat([seg_body, high_fine], dim=1))  # 200
            seg_body_pre = self.body_out_pre[i](seg_body)
            seg_body_out = F.interpolate(self.body_out[i](seg_body_pre), size=x_size[2:],
                                         mode='bilinear', align_corners=True)  # 800
            seg_bodys.append(seg_body_pre)
            seg_body_outs.append(seg_body_out)

            seg_edge_pre = self.edge_out_pre[i](seg_edge)  # 200
            seg_edge_out_pre = self.edge_out[i](seg_edge_pre)
            seg_edge_out = F.interpolate(seg_edge_out_pre, size=x_size[2:],
                                         mode='bilinear', align_corners=True)  # 800
            seg_edges.append(seg_edge_pre)
            seg_edge_outs.append(seg_edge_out)

            seg_out = seg_body + seg_edge  # 200
            seg_out = self.refines[i](seg_out, torch.sigmoid(seg_edge_out_pre.clone().detach()))
            seg_final_pre = self.final_seg_out_pre[i](seg_out)
            seg_final_out = F.interpolate(self.final_seg_out[i](seg_final_pre), size=x_size[2:],
                                          mode='bilinear', align_corners=True)
            seg_finals.append(seg_final_pre)
            seg_final_outs.append(seg_final_out)

        end = time.time()
        if cal_inference_time:
            return end - start

        if self.training:
            return self.criterion((seg_final_outs, seg_body_outs, seg_edge_outs), gts)

        return seg_final_outs[-1]


class DABlock_merge(nn.Module):
    def __init__(self,  in_planes=2048, out_planes=512, norm_layer=nn.BatchNorm2d):
        super(DABlock_merge, self).__init__()
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
    def __init__(self, num_classes, trunk='seresnext-50', criterion=None, variant="D"):
        super(DANet, self).__init__()
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

        self.da_head = DABlock_merge(2048, 512, norm_layer=Norm2d)
        self.final = nn.Conv2d(512, num_classes,kernel_size=1)
        initialize_weights(self.da_head)
        initialize_weights(self.final)

    def forward(self, x, gts=None, return_atten_map=False, hp=-1, wp=-1, cal_inference_time=False):

        start = time.time()
        x_size = x.size()  # 800
        x0 = self.layer0(x)  # 400
        x1 = self.layer1(x0)  # 400
        x2 = self.layer2(x1)  # 100
        x3 = self.layer3(x2)  # 100
        x4 = self.layer4(x3)  # 100
        if return_atten_map:
            return self.da_head(x4, return_atten_map, hp, wp)
        da_out = self.da_head(x4)
        da_out = self.final(da_out)
        main_out = Upsample(da_out, x_size[2:])

        end = time.time()
        if cal_inference_time:
            return end-start

        if self.training:
            return self.criterion(main_out, gts)

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

def DANet_r50_casc3_aspp1_wofirst(num_classes, criterion, num_groups=8, num_cascade=1, down_sample=True,
                                          gcn_first=True, num_points=128, threshold=0.9, num_aspp=1):
    """
    ResNet-50 Based Network with stride=8, contrast and cascade model
    """
    return DANet_rdm_pgm_cascadewos_wofirst(num_classes, trunk='resnet-50-deep', criterion=criterion,
                                         variant='D16', skip='m1', edge_group=num_groups, num_cascade=num_cascade,
                                         down_sample=down_sample, gcn_first=gcn_first, num_points=num_points,
                                         threshold=threshold, num_aspp=num_aspp)

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
