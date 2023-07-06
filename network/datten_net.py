import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
from network import resnet_d as Resnet_Deep
from network.extension.dcn.deform_conv_module import DFConv2d
from network.nn.mynn import Norm2d, Upsample

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


class _DAPBlock(nn.Module):
    """ ConvNet block for building DenseASPP. """

    def __init__(self, input_num, num1, num2, dilation_rate, drop_out, bn_start=True,modulation=True, BatchNorm=nn.BatchNorm2d):
        super(_DAPBlock, self).__init__()
        self.modulation = modulation
        self.bn_start = bn_start
        self.bn1 = BatchNorm(input_num, momentum=0.0003)
        self.relu1 = nn.ReLU(inplace = True)
        self.conv_1 = nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1)
        self.bn2 = BatchNorm(num1, momentum=0.0003)
        self.relu2 = nn.ReLU(inplace = True)

        self.deform_conv = DFConv2d(num1, num2, with_modulated_dcn=False, kernel_size=3)

        self.conv_3 =nn.Conv2d(in_channels=num2, out_channels=num2, kernel_size=3,groups=num2,dilation=dilation_rate,
                               padding=dilation_rate)
        self.drop_out = drop_out

    def forward(self,input):
        if self.bn_start == True:
            input = self.bn1(input)
        feature = self.relu1(input)
        feature = self.conv_1(feature)
        feature = self.bn2(feature)
        feature1 = self.deform_conv(feature)
        feature1 = self.conv_3(feature1)
        return feature1


class DAPHead(nn.Module):
    def __init__(self,num_features, d_feature0, d_feature1, dropout0, modulation=True, BatchNorm=nn.BatchNorm2d):
        super(DAPHead,self).__init__()
        self.num_features = num_features
        self.d_feature0 = d_feature0
        self.d_feature1 = d_feature1
        self.init_feature = 2048 - 5*d_feature1
        self.dropout0 = dropout0
        self.modulation_all = modulation
        self.BatchNorm=BatchNorm

        self.init_conv_dap = nn.Conv2d(self.num_features,self.init_feature, kernel_size=(3,3),padding=1)
        self.num_features = self.init_feature

        self.DAP_3 = _DAPBlock(input_num=self.num_features, num1=self.d_feature0, num2=self.d_feature1,
                                      dilation_rate=3, drop_out=self.dropout0, bn_start=False,modulation=self.modulation_all,BatchNorm=BatchNorm)

        self.DAP_5 = _DAPBlock(input_num=self.num_features + self.d_feature1 * 1, num1=self.d_feature0, num2=self.d_feature1,
                                      dilation_rate=5, drop_out=self.dropout0, bn_start=True, modulation=self.modulation_all,BatchNorm=BatchNorm)

        self.DAP_7 = _DAPBlock(input_num=self.num_features + self.d_feature1 * 2, num1=self.d_feature0, num2=self.d_feature1,
                                       dilation_rate=7, drop_out=self.dropout0, bn_start=True, modulation=self.modulation_all,BatchNorm=BatchNorm)

        self.DAP_9 = _DAPBlock(input_num=self.num_features + self.d_feature1 * 3, num1=self.d_feature0, num2=self.d_feature1,
                                       dilation_rate=9, drop_out=self.dropout0, bn_start=True, modulation=self.modulation_all,BatchNorm=BatchNorm)

        self.DAP_11 = _DAPBlock(input_num=self.num_features + self.d_feature1 * 4, num1=self.d_feature0, num2=self.d_feature1,
                                       dilation_rate=11, drop_out=self.dropout0, bn_start=True, modulation=self.modulation_all, BatchNorm=BatchNorm)

    def forward(self,feature):

            feature = self.init_conv_dap(feature)
            dap3 = self.DAP_3(feature)
            feature = torch.cat((dap3, feature), dim=1)

            dap5 = self.DAP_5(feature)
            feature = torch.cat((dap5, feature), dim=1)

            dap7 = self.DAP_7(feature)
            feature = torch.cat((dap7, feature), dim=1)

            dap9 = self.DAP_9(feature)
            feature = torch.cat((dap9, feature), dim=1)

            dap11 = self.DAP_11(feature)
            feature = torch.cat((dap11, feature), dim=1)
            return feature


class UperNetHead(nn.Module):

    def __init__(self, inplane, num_class, norm_layer=nn.BatchNorm2d, fpn_inplanes=[256, 512, 1024, 2048], fpn_dim=256,
                fpn_dsn=False):
        super(UperNetHead, self).__init__()

        self.ppm = nn.Sequential(DAPHead(2048, 512, 128, .1 ,modulation=True),
          nn.Conv2d(2048, fpn_dim, 1)
        )

        self.fpn_dsn = fpn_dsn
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv2d(fpn_inplane, fpn_dim, 1),
                    norm_layer(fpn_dim),
                    nn.ReLU(inplace=False)
                )
            )
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        self.dsn = []
        self.fpn_dcn = []
        for i in range(len(fpn_inplanes) - 1):
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))

        self.fpn_out = nn.ModuleList(self.fpn_out)
        self.fpn_dcn = nn.ModuleList(self.fpn_dcn)
        if self.fpn_dsn:
            self.dsn = nn.ModuleList(self.dsn)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1)
        )

    def forward(self, conv_out):
        psp_out = self.ppm(conv_out[-1])

        f = psp_out
        fpn_feature_list = [psp_out]
        out = []
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)  # lateral branch
            f = F.upsample(f, size=conv_x.size()[2:], mode='bilinear', align_corners=True)
            f = conv_x + f
            fpn_feature_list.append(self.fpn_out[i](f))
            if self.fpn_dsn:
                out.append(self.dsn[i](f))

        fpn_feature_list.reverse()  # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]

        result = fpn_feature_list[0]
        for i in range(1, len(fpn_feature_list)):
            result = result + nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=True)

        x = self.conv_last(result)

        return x, out


class DynamicAttenNet(nn.Module):
    def __init__(self, num_classes, trunk='resnet-101', criterion=None, variant='D',
                 skip='m1', skip_num=48, fpn_dsn=False, dcn_type="v1"):
        super(DynamicAttenNet, self).__init__()
        self.criterion = criterion
        self.variant = variant
        self.skip = skip
        self.skip_num = skip_num
        self.fpn_dsn = fpn_dsn
        self.dcn_type = dcn_type
        if trunk == trunk == 'resnet-50-deep':
            resnet = Resnet_Deep.resnet50()
        elif trunk == 'resnet-101-deep':
            resnet = Resnet_Deep.resnet101()
        else:
            raise ValueError("Not a valid network arch")

        resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer0 = resnet.layer0
        self.layer1, self.layer2, self.layer3, self.layer4 = \
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        del resnet

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

        if trunk == 'resnet-18-deep':
            inplane_head = 512
            self.head = UperNetHead(inplane_head, num_class=num_classes, norm_layer=Norm2d,
                                         fpn_inplanes=[64, 128, 256, 512], fpn_dim=128, )
        else:
            inplane_head = 2048
            self.head = UperNetHead(inplane_head, num_class=num_classes, norm_layer=Norm2d,)

    def forward(self, x, gts=None):
        x_size = x.size()  # 800
        x0 = self.layer0(x)  # 400
        x1 = self.layer1(x0)  # 400
        x2 = self.layer2(x1)  # 100
        x3 = self.layer3(x2)  # 100
        x4 = self.layer4(x3)  # 100
        x = self.head([x1, x2, x3, x4])
        main_out = Upsample(x[0], x_size[2:])
        if self.training:
            if not self.fpn_dsn:
                return self.criterion(main_out, gts)
            return self.criterion(x, gts)
        return main_out


#m = DAP(2048, 512,128,.1,modulation = True).cuda()
#a = torch.Tensor(2,2048, 128, 128).cuda()
#o = m(a)
#print(o.size())

def DynamicAttenNetDeepR50(num_classes, criterion):
    """
    ResNet-50 Based Network
    """
    return DynamicAttenNet(num_classes, trunk='resnet-50-deep', criterion=criterion, variant='D', skip='m1')