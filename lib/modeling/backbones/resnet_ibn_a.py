import copy

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F


from ...layers.pooling import GeM


__all__ = ['ResNet_IBN', 'resnet50_ibn_a', 'resnet101_ibn_a',
           'resnet152_ibn_a', 'se_resnet101_ibn_a']


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class cross_channel_attention(nn.Module):

    def __init__(self, output_channel, f_2_channel=2048, f_4_channel=2048):
        super(cross_channel_attention, self).__init__()

        self.f_2_channel = f_2_channel # 512
        self.f_4_channel = f_4_channel # 2048

        self.conv1_1 = nn.Conv2d(self.f_2_channel, output_channel, kernel_size=3, stride=2, padding=1)
        self.conv1_2 = nn.Conv2d(self.f_4_channel, output_channel, kernel_size=3, stride=1, padding=1)
        self.bn1_1 = nn.BatchNorm2d(output_channel)
        self.bn1_2 = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(output_channel, output_channel, kernel_size=1, stride=1),
            nn.BatchNorm2d(output_channel),
            nn.ReLU()
        )

        self.downsample = nn.Sequential(
            nn.Conv2d(output_channel*2, output_channel//2,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(output_channel//2),
            nn.ReLU()
        )
        self.upsample = nn.Sequential(
            nn.Conv2d(output_channel//2, output_channel,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(output_channel)
        )
        self.sigmoid = nn.Sigmoid()



    def forward(self, x_1, x_2):

        x_1 = self.conv1_1(x_1)
        x_1 = self.bn1_1(x_1)
        x_1 = self.relu(x_1)

        x_2 = self.conv1_2(x_2)
        x_2 = self.bn1_2(x_2)
        x_2 = self.relu(x_2)


        x_1_AP = self.gap(x_1)
        x_2_AP = self.gap(x_2)

        x_cat = torch.cat((x_1_AP, x_2_AP), dim=1)
        x_cat = self.downsample(x_cat)
        x_cat = self.upsample(x_cat)
        alpha = self.sigmoid(x_cat)

        x_fin = alpha * x_1 + (1-alpha) * x_2
        x_fin = self.conv2(x_fin)

        return x_fin


class spatial_feature_aggreation(nn.Module):
    def __init__(self, normalize_feature=True):
        super(spatial_feature_aggreation,self).__init__()
        self.normalize_feature = normalize_feature

    def forward(self, f_ori, f_mask): # 32,2048

        h = f_ori.size(0)  # 32
        w = f_ori.size(1)  # 2048
        if self.normalize_feature:
            f_ori_nor = F.normalize(f_ori)
            f_mask_nor = F.normalize(f_mask)
        correlation_matrix = torch.matmul(f_mask_nor.T, f_ori_nor)
        weights = torch.mean(correlation_matrix, dim=1)
        f_fusion = f_ori * weights.unsqueeze(0) + 0.3 * f_ori

        return f_fusion



class IBN(nn.Module):
    def __init__(self, planes):
        super(IBN, self).__init__()
        half1 = int(planes/2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2)
    
    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, int(channel/reduction), bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(int(channel/reduction), channel, bias=False),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, ibn=False, stride=1, downsample=None, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if ibn:
            self.bn1 = IBN(planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck_IBN(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, ibn=False, stride=1, downsample=None):
        super(Bottleneck_IBN, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if ibn:
            self.bn1 = IBN(planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_IBN(nn.Module):

    def __init__(self, last_stride, block, layers, num_classes=1000):
        scale = 64
        self.inplanes = scale
        super(ResNet_IBN, self).__init__()

        self.conv1 = nn.Conv2d(3, scale, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(scale)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, scale, layers[0])
        self.layer2 = self._make_layer(block, scale*2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, scale*4, layers[2], stride=2) # 256
        self.layer4 = self._make_layer(block, scale*8, layers[3], stride=last_stride) # (bottleneck, 512, 3, 1)

        self.layer3_y = copy.deepcopy(self.layer3)
        self.layer4_y = copy.deepcopy(self.layer4)

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(scale * 8 * block.expansion, num_classes)

        self.CCA = cross_channel_attention(f_2_channel=512, f_4_channel=2048, output_channel=2048)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1): # ①64，3,1 ②128,4,2 ③256,6,2 ④512，3,1
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion: # 512*4=2048
            downsample = nn.Sequential( # 用于计算残差，并加在最后的结果上
                nn.Conv2d(self.inplanes, planes * block.expansion, # 64,64*4=256
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        ibn = True
        if planes == 512:
            ibn = False
        layers.append(block(self.inplanes, planes, ibn, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks): # [2,3,5,2]
            layers.append(block(self.inplanes, planes, ibn))

        return nn.Sequential(*layers)

    def featuremap(self, x,y):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.maxpool(y)

        x = self.layer1(x)
        x = self.layer2(x)
        y = self.layer1(y)
        y = self.layer2(y)
        y_2 = y

        x = self.layer3(x)
        x = self.layer4(x)
        y = self.layer3_y(y)
        y = self.layer4_y(y)
        y = self.CCA(y_2, y)

        return x, y

    def forward(self, x,y):

        f_ori, f_mask = self.featuremap(x,y)
        return f_ori, f_mask

    def load_param(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu')
        for i in param_dict:
            if 'fc' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

def resnet50_ibn_a(last_stride, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(last_stride, Bottleneck_IBN, [3, 4, 6, 3], **kwargs)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101_ibn_a(last_stride, pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(last_stride, Bottleneck_IBN, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model

def se_resnet101_ibn_a(last_stride):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(last_stride, SEBottleneck,  [3, 4, 23, 3])
    return model


def resnet152_ibn_a(last_stride, pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(last_stride, Bottleneck_IBN, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model