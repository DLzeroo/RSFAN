import copy
import torch
from torch import nn
from .resnet_ibn_a import ResNet_IBN, resnet50_ibn_a


class double_branch(nn.Module):
    def __init__(self):
        super(double_branch, self).__init__()

        resnet50 = resnet50_ibn_a(pretrained=True)

        self.backbone = nn.Sequential(
            resnet50.conv1,
            resnet50.bn1,
            resnet50.relu,
            resnet50.maxpool,
            resnet50.layer1,
            resnet50.layer2
        )

        self.ori_branch = nn.Sequential()