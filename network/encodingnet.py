import torch
import torch.nn as nn
import torch.nn.functional as F
from network import resnet_d as Resnet_Deep
from network.nn.mynn import initialize_weights, Norm2d, Upsample
from network.resnext import resnext50_32x4d, resnext101_32x8d, resnext101_32x8

import time


class _EncHead(nn.Module):
    def __init__(self, in_channels, nclass, se_loss=False, lateral=True,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(_EncHead, self).__init__()
        self.lateral = lateral
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, 512, 3, padding=1, bias=False),
            norm_layer(512, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        if lateral:
            self.connect = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(512, 512, 1, bias=False),
                    norm_layer(512, **({} if norm_kwargs is None else norm_kwargs)),
                    nn.ReLU(True)),
                nn.Sequential(
                    nn.Conv2d(1024, 512, 1, bias=False),
                    norm_layer(512, **({} if norm_kwargs is None else norm_kwargs)),
                    nn.ReLU(True)),
            ])
            self.fusion = nn.Sequential(
                nn.Conv2d(3 * 512, 512, 3, padding=1, bias=False),
                norm_layer(512, **({} if norm_kwargs is None else norm_kwargs)),
                nn.ReLU(True)
            )
        self.encmodule = EncModule(512, nclass, ncodes=32, se_loss=se_loss,
                                   norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        self.conv6 = nn.Sequential(
            nn.Dropout(0.1, False),
            nn.Conv2d(512, nclass, 1)
        )

    def forward(self, inputs):
        x1,x2,x3= inputs
        feat = self.conv5(x3)
        if self.lateral:
            c2 = self.connect[0](x1)
            c3 = self.connect[1](x2)
            feat = self.fusion(torch.cat([feat, c2, c3], 1))
        outs = list(self.encmodule(feat))
        outs[0] = self.conv6(outs[0])
        return outs[0]


class EncModule(nn.Module):
    def __init__(self, in_channels, nclass, ncodes=32, se_loss=True,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(EncModule, self).__init__()
        self.se_loss = se_loss
        self.encoding = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            norm_layer(in_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True),
            Encoding(D=in_channels, K=ncodes),
            nn.BatchNorm1d(ncodes),
            nn.ReLU(True),
            Mean(dim=1)
        )
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid()
        )
        if self.se_loss:
            self.selayer = nn.Linear(in_channels, nclass)

    def forward(self, x):
        en = self.encoding(x)
        b, c, _, _ = x.size()
        gamma = self.fc(en)
        y = gamma.view(b, c, 1, 1)
        outputs = [F.relu_(x + x * y)]
        if self.se_loss:
            outputs.append(self.selayer(en))
        return tuple(outputs)


class Encoding(nn.Module):
    def __init__(self, D, K):
        super(Encoding, self).__init__()
        # init codewords and smoothing factor
        self.D, self.K = D, K
        self.codewords = nn.Parameter(torch.Tensor(K, D), requires_grad=True)
        self.scale = nn.Parameter(torch.Tensor(K), requires_grad=True)
        self.reset_params()

    def reset_params(self):
        std1 = 1. / ((self.K * self.D) ** (1 / 2))
        self.codewords.data.uniform_(-std1, std1)
        self.scale.data.uniform_(-1, 0)

    def forward(self, X):
        # input X is a 4D tensor
        assert (X.size(1) == self.D)
        B, D = X.size(0), self.D
        if X.dim() == 3:
            # BxDxN -> BxNxD
            X = X.transpose(1, 2).contiguous()
        elif X.dim() == 4:
            # BxDxHxW -> Bx(HW)xD
            X = X.view(B, D, -1).transpose(1, 2).contiguous()
        else:
            raise RuntimeError('Encoding Layer unknown input dims!')
        # assignment weights BxNxK
        A = F.softmax(self.scale_l2(X, self.codewords, self.scale), dim=2)
        # aggregate
        E = self.aggregate(A, X, self.codewords)
        return E

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'N x' + str(self.D) + '=>' + str(self.K) + 'x' \
               + str(self.D) + ')'

    @staticmethod
    def scale_l2(X, C, S):
        S = S.view(1, 1, C.size(0), 1)
        X = X.unsqueeze(2).expand(X.size(0), X.size(1), C.size(0), C.size(1))
        C = C.unsqueeze(0).unsqueeze(0)
        SL = S * (X - C)
        SL = SL.pow(2).sum(3)
        return SL

    @staticmethod
    def aggregate(A, X, C):
        A = A.unsqueeze(3)
        X = X.unsqueeze(2).expand(X.size(0), X.size(1), C.size(0), C.size(1))
        C = C.unsqueeze(0).unsqueeze(0)
        E = A * (X - C)
        E = E.sum(1)
        return E


class Mean(nn.Module):
    def __init__(self, dim, keep_dim=False):
        super(Mean, self).__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, input):
        return input.mean(self.dim, self.keep_dim)


class EncodingNet(nn.Module):
    """
    Implement Encoding model
    A: stride8
    B: stride16
    with skip connections
    """
    def __init__(self, num_classes, trunk='resnet-50-deep', criterion=None, variant="D"):
        super(EncodingNet, self).__init__()
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

        self.encoding = _EncHead(2048, nclass=num_classes, norm_layer=Norm2d)

        initialize_weights(self.encoding)

    def forward(self, x, gts=None, cal_inference_time=False):

        start = time.time()
        x_size = x.size()  # 800
        x0 = self.layer0(x)  # 400
        x1 = self.layer1(x0)  # 400
        x2 = self.layer2(x1)  # 100
        x3 = self.layer3(x2)  # 100
        x4 = self.layer4(x3)  # 100
        out = self.encoding([x2, x3, x4])

        main_out = Upsample(out, x_size[2:])

        end = time.time()
        if cal_inference_time:
            return end-start

        if self.training:
            return self.criterion(main_out, gts)

        return main_out


def EncodingNet_v1_r50(num_classes, criterion, variant='D'):
    """
    ResNet-50 Based Network
    """
    return EncodingNet(num_classes, trunk='resnet-50-deep', criterion=criterion, variant=variant)


def EncodingNet_v1_r101(num_classes, criterion, variant='D'):
    """
    ResNet-50 Based Network
    """
    return EncodingNet(num_classes, trunk='resnet-101-deep', criterion=criterion, variant=variant)