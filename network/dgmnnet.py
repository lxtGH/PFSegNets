import torch
import torch.nn as nn
from network import resnet_d as Resnet_Deep
from network.nn.mynn import Norm2d, Upsample
import time
from network.nn.operators import _ConvBNReLU as ConvBNReLU
from network.extension.dcn.deform_unfold_module import DeformUnfold


class MoreMultiDynamicWeights(nn.Module):
    def __init__(self, channels, norm_layer=None):
        super(MoreMultiDynamicWeights, self).__init__()
        self.cat_conv = nn.Conv2d(channels, 45, 3, padding=1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        self.unfold1 = nn.Unfold(kernel_size=(3, 3), padding=1)
        self.unfold2 = nn.Unfold(kernel_size=(3, 3), padding=6, dilation=6)
        self.unfold3 = nn.Unfold(kernel_size=(3, 3), padding=12, dilation=12)
        self.unfold4 = nn.Unfold(kernel_size=(3, 3), padding=24, dilation=24)
        self.unfold5 = nn.Unfold(kernel_size=(3, 3), padding=36, dilation=36)
        self.scale_conv = nn.Sequential(nn.Conv2d(channels * 6, channels, 1, padding=0, bias=False),
                                        norm_layer(channels))

    def forward(self, x):
        blur_depth = x
        guidance = self.cat_conv(blur_depth)
        N, C, H, W = guidance.size()

        dynamic_filter1 = guidance[:, :9, :, :]
        dynamic_filter2 = guidance[:, 9:18, :, :]
        dynamic_filter3 = guidance[:, 18:27, :, :]
        dynamic_filter4 = guidance[:, 27:36, :, :]
        dynamic_filter5 = guidance[:, -9:, :, :]
        dynamic_filter1 = self.softmax(dynamic_filter1.permute(0, 2, 3, 1).contiguous().view(N * H * W, -1))
        dynamic_filter2 = self.softmax(dynamic_filter2.permute(0, 2, 3, 1).contiguous().view(N * H * W, -1))
        dynamic_filter3 = self.softmax(dynamic_filter3.permute(0, 2, 3, 1).contiguous().view(N * H * W, -1))
        dynamic_filter4 = self.softmax(dynamic_filter4.permute(0, 2, 3, 1).contiguous().view(N * H * W, -1))
        dynamic_filter5 = self.softmax(dynamic_filter5.permute(0, 2, 3, 1).contiguous().view(N * H * W, -1))

        xd_unfold1 = self.unfold1(blur_depth)
        xd_unfold2 = self.unfold2(blur_depth)
        xd_unfold3 = self.unfold3(blur_depth)
        xd_unfold4 = self.unfold4(blur_depth)
        xd_unfold5 = self.unfold5(blur_depth)
        xd_unfold1 = xd_unfold1.contiguous().view(N, blur_depth.size()[1], 9, H * W).permute(0, 3, 1, 2).contiguous().view(N * H * W, blur_depth.size()[1], 9)
        xd_unfold2 = xd_unfold2.contiguous().view(N, blur_depth.size()[1], 9, H * W).permute(0, 3, 1, 2).contiguous().view(N * H * W, blur_depth.size()[1], 9)
        xd_unfold3 = xd_unfold3.contiguous().view(N, blur_depth.size()[1], 9, H * W).permute(0, 3, 1, 2).contiguous().view(N * H * W, blur_depth.size()[1], 9)
        xd_unfold4 = xd_unfold4.contiguous().view(N, blur_depth.size()[1], 9, H * W).permute(0, 3, 1, 2).contiguous().view(N * H * W, blur_depth.size()[1], 9)
        xd_unfold5 = xd_unfold5.contiguous().view(N, blur_depth.size()[1], 9, H * W).permute(0, 3, 1, 2).contiguous().view(N * H * W, blur_depth.size()[1], 9)

        out1 = torch.bmm(xd_unfold1, dynamic_filter1.unsqueeze(2))  # N*H*W, xd.size()[1], 1
        out1 = out1.view(N, H, W, blur_depth.size()[1]).permute(0, 3, 1, 2).contiguous()  # N, xd.size()[1], H, W
        out2 = torch.bmm(xd_unfold2, dynamic_filter2.unsqueeze(2))  # N*H*W, xd.size()[1], 1
        out2 = out2.view(N, H, W, blur_depth.size()[1]).permute(0, 3, 1, 2).contiguous()  # N, xd.size()[1], H, W
        out3 = torch.bmm(xd_unfold3, dynamic_filter3.unsqueeze(2))  # N*H*W, xd.size()[1], 1
        out3 = out3.view(N, H, W, blur_depth.size()[1]).permute(0, 3, 1, 2).contiguous()  # N, xd.size()[1], H, W
        out4 = torch.bmm(xd_unfold4, dynamic_filter4.unsqueeze(2))  # N*H*W, xd.size()[1], 1
        out4 = out4.view(N, H, W, blur_depth.size()[1]).permute(0, 3, 1, 2).contiguous()  # N, xd.size()[1], H, W
        out5 = torch.bmm(xd_unfold5, dynamic_filter5.unsqueeze(2))  # N*H*W, xd.size()[1], 1
        out5 = out5.view(N, H, W, blur_depth.size()[1]).permute(0, 3, 1, 2).contiguous()  # N, xd.size()[1], H, W

        out = self.scale_conv(torch.cat((x, out1, out2, out3, out4, out5), 1))

        return out


class DynamicModuleHead(nn.Module):
    def __init__(self, channels, norm_layer=None):
        super(DynamicModuleHead, self).__init__()

        self.cat_conv1 = nn.Conv2d(channels, 27, 3, padding=1, bias=False)
        self.cat_conv2 = nn.Conv2d(channels, 27, 3, padding=6, dilation=6, bias=False)
        self.cat_conv3 = nn.Conv2d(channels, 27, 3, padding=12, dilation=12, bias=False)
        self.cat_conv4 = nn.Conv2d(channels, 27, 3, padding=24, dilation=24, bias=False)
        self.cat_conv5 = nn.Conv2d(channels, 27, 3, padding=36, dilation=36, bias=False)

        self.unfold1 = DeformUnfold(kernel_size=(3, 3), padding=1)
        self.unfold2 = DeformUnfold(kernel_size=(3, 3), padding=6, dilation=6)
        self.unfold3 = DeformUnfold(kernel_size=(3, 3), padding=12, dilation=12)
        self.unfold4 = DeformUnfold(kernel_size=(3, 3), padding=24, dilation=24)
        self.unfold5 = DeformUnfold(kernel_size=(3, 3), padding=36, dilation=36)

        self.softmax = nn.Softmax(dim=-1)

        self.scale_conv = nn.Sequential(nn.Conv2d(channels * 6, channels, 1, padding=0, bias=False),
                                        norm_layer(channels))

    def forward(self, x):
        blur_depth = x
        N, C, H, W = x.size()

        dynamic_filter_offset1 = self.cat_conv1(blur_depth)
        dynamic_filter1 = dynamic_filter_offset1[:, :9, :, :]
        offset1 = dynamic_filter_offset1[:, -18:, :, :]

        dynamic_filter_offset2 = self.cat_conv2(blur_depth)
        dynamic_filter2 = dynamic_filter_offset2[:, :9, :, :]
        offset2 = dynamic_filter_offset2[:, -18:, :, :]

        dynamic_filter_offset3 = self.cat_conv3(blur_depth)
        dynamic_filter3 = dynamic_filter_offset3[:, :9, :, :]
        offset3 = dynamic_filter_offset3[:, -18:, :, :]

        dynamic_filter_offset4 = self.cat_conv4(blur_depth)
        dynamic_filter4 = dynamic_filter_offset4[:, :9, :, :]
        offset4 = dynamic_filter_offset4[:, -18:, :, :]

        dynamic_filter_offset5 = self.cat_conv5(blur_depth)
        dynamic_filter5 = dynamic_filter_offset5[:, :9, :, :]
        offset5 = dynamic_filter_offset5[:, -18:, :, :]

        dynamic_filter1 = self.softmax(dynamic_filter1.permute(0, 2, 3, 1).contiguous().view(N * H * W, -1))
        dynamic_filter2 = self.softmax(dynamic_filter2.permute(0, 2, 3, 1).contiguous().view(N * H * W, -1))
        dynamic_filter3 = self.softmax(dynamic_filter3.permute(0, 2, 3, 1).contiguous().view(N * H * W, -1))
        dynamic_filter4 = self.softmax(dynamic_filter4.permute(0, 2, 3, 1).contiguous().view(N * H * W, -1))
        dynamic_filter5 = self.softmax(dynamic_filter5.permute(0, 2, 3, 1).contiguous().view(N * H * W, -1))

        xd_unfold1 = self.unfold1(blur_depth, offset1)
        xd_unfold2 = self.unfold2(blur_depth, offset2)
        xd_unfold3 = self.unfold3(blur_depth, offset3)
        xd_unfold4 = self.unfold4(blur_depth, offset4)
        xd_unfold5 = self.unfold5(blur_depth, offset5)

        xd_unfold1 = xd_unfold1.contiguous().view(N, blur_depth.size()[1], 9, H * W).permute(0, 3, 1, 2).contiguous().view(N * H * W, blur_depth.size()[1], 9)
        xd_unfold2 = xd_unfold2.contiguous().view(N, blur_depth.size()[1], 9, H * W).permute(0, 3, 1, 2).contiguous().view(N * H * W, blur_depth.size()[1], 9)
        xd_unfold3 = xd_unfold3.contiguous().view(N, blur_depth.size()[1], 9, H * W).permute(0, 3, 1, 2).contiguous().view(N * H * W, blur_depth.size()[1], 9)
        xd_unfold4 = xd_unfold4.contiguous().view(N, blur_depth.size()[1], 9, H * W).permute(0, 3, 1, 2).contiguous().view(N * H * W, blur_depth.size()[1], 9)
        xd_unfold5 = xd_unfold5.contiguous().view(N, blur_depth.size()[1], 9, H * W).permute(0, 3, 1, 2).contiguous().view(N * H * W, blur_depth.size()[1], 9)

        out1 = torch.bmm(xd_unfold1, dynamic_filter1.unsqueeze(2))  # N*H*W, xd.size()[1], 1
        out1 = out1.view(N, H, W, blur_depth.size()[1]).permute(0, 3, 1, 2).contiguous()  # N, xd.size()[1], H, W
        out2 = torch.bmm(xd_unfold2, dynamic_filter2.unsqueeze(2))  # N*H*W, xd.size()[1], 1
        out2 = out2.view(N, H, W, blur_depth.size()[1]).permute(0, 3, 1, 2).contiguous()  # N, xd.size()[1], H, W
        out3 = torch.bmm(xd_unfold3, dynamic_filter3.unsqueeze(2))  # N*H*W, xd.size()[1], 1
        out3 = out3.view(N, H, W, blur_depth.size()[1]).permute(0, 3, 1, 2).contiguous()  # N, xd.size()[1], H, W
        out4 = torch.bmm(xd_unfold4, dynamic_filter4.unsqueeze(2))  # N*H*W, xd.size()[1], 1
        out4 = out4.view(N, H, W, blur_depth.size()[1]).permute(0, 3, 1, 2).contiguous()  # N, xd.size()[1], H, W
        out5 = torch.bmm(xd_unfold5, dynamic_filter5.unsqueeze(2))  # N*H*W, xd.size()[1], 1
        out5 = out5.view(N, H, W, blur_depth.size()[1]).permute(0, 3, 1, 2).contiguous()  # N, xd.size()[1], H, W

        out = self.scale_conv(torch.cat((x, out1, out2, out3, out4, out5), 1))

        return out


class DGMNNet(nn.Module):
    """
    Implement Encoding model
    A: stride8
    B: stride16
    with skip connections
    """
    def __init__(self, num_classes, trunk='resnet-50-deep', criterion=None, variant="D", dynamic=False):
        super(DGMNNet, self).__init__()
        self.criterion = criterion
        self.variant = variant

        if trunk == 'resnet-50-deep':
            resnet = Resnet_Deep.resnet50()
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        elif trunk == 'resnet-101-deep':
            resnet = Resnet_Deep.resnet101()
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

        self.fc0 = ConvBNReLU(2048, 512, 3, 1, 1, 1, norm_layer=Norm2d)
        if dynamic is not True:
            self.head = MoreMultiDynamicWeights(512, Norm2d)
        else:
            self.head = DynamicModuleHead(512, Norm2d)
        self.fc1 = nn.Sequential(
            ConvBNReLU(512, 256, 1, 1, 1, 1, norm_layer=Norm2d),
            nn.Dropout2d(p=0.1))
        self.fc2 = nn.Conv2d(256, num_classes, 1)

    def forward(self, x, gts=None, cal_inference_time=False):

        start = time.time()
        x_size = x.size()  # 800
        x0 = self.layer0(x)  # 400
        x1 = self.layer1(x0)  # 400
        x2 = self.layer2(x1)  # 100
        x3 = self.layer3(x2)  # 100
        x4 = self.layer4(x3)  # 100
        x = self.fc0(x4)
        x = self.head(x)
        x = self.fc1(x)
        out = self.fc2(x)

        main_out = Upsample(out, x_size[2:])

        end = time.time()
        if cal_inference_time:
            return end-start

        if self.training:
            return self.criterion(main_out, gts)

        return main_out


def DGMN_r50_fix_head(num_classes, criterion):
    """
    ResNet-50 Based Network
    """
    return DGMNNet(num_classes, trunk='resnet-50-deep', criterion=criterion, variant='D')


def DGMN_r101_dynamic_head(num_classes, criterion):
    """
    ResNet-50 Based Network
    """
    return DGMNNet(num_classes, trunk='resnet-101-deep', criterion=criterion, variant='D', dynamic=True)


def DGMN_r50_dynamic_head(num_classes, criterion):
    """
    ResNet-50 Based Network
    """
    return DGMNNet(num_classes, trunk='resnet-50-deep', criterion=criterion, variant='D', dynamic=True)
