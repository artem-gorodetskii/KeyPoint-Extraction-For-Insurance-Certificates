# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

'''
Copy-paste from https://github.com/jwyang/faster-rcnn.pytorch/blob/master/lib/model/faster_rcnn/resnet.py 
with modifications:
    * remove last two layers (fc, conv)
    * modified conv1, maxpool layers
    * add conv2, bn2, relu2 layers for output final feature maps
'''

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
           'se_resnet18', 'se_resnet34', 'se_resnet50', 'se_resnet101', 'se_resnet152']

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

# AAG: SE block added
class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        #self.bn1 = nn.BatchNorm2d(planes)
        # AAG: changing Batch Normalization on Group Normalization with 32 groups
        self.bn1 = nn.GroupNorm(32, planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        #self.bn2 = nn.BatchNorm2d(planes)
        # AAG: changing Batch Normalization on Group Normalization with 32 groups
        self.bn2 = nn.GroupNorm(32, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# AAG: basic SE block was added
class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, r=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        # AAG: changing Batch Normalization on Group Normalization with 32 groups
        self.bn1 = nn.GroupNorm(32, planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        # AAG: changing Batch Normalization on Group Normalization with 32 groups
        self.bn2 = nn.GroupNorm(32, planes)
        self.downsample = downsample
        self.stride = stride

        # add SE block
        self.se = SE_Block(planes, r)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # add SE operation
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        #self.bn1 = nn.BatchNorm2d(planes)
        # AAG: changing Batch Normalization on Group Normalization with 32 groups
        self.bn1 = nn.GroupNorm(32, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=1, bias=False)
        #self.bn2 = nn.BatchNorm2d(planes)
        # AAG: changing Batch Normalization on Group Normalization with 32 groups
        self.bn2 = nn.GroupNorm(32, planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        #self.bn3 = nn.BatchNorm2d(planes * 4)
        # AAG: changing Batch Normalization on Group Normalization with 32 groups
        self.bn3 = nn.GroupNorm(32, planes * 4)
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

# AAG: SEBottleneck block with standart SE operation was added
class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, r=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        # AAG: changing Batch Normalization on Group Normalization with 32 groups
        self.bn1 = nn.GroupNorm(32, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=1, bias=False)
        # AAG: changing Batch Normalization on Group Normalization with 32 groups
        self.bn2 = nn.GroupNorm(32, planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        # AAG: changing Batch Normalization on Group Normalization with 32 groups
        self.bn3 = nn.GroupNorm(32, planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        # Add SE block
        self.se = SE_Block(planes * 4, r)

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
        # Add SE operation
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, output_channels=512):

        self.expansion = block.expansion
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # change
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        #self.bn1 = nn.BatchNorm2d(64)
        # AAG: changing Batch Normalization on Group Normalization with 32 groups
        self.bn1 = nn.GroupNorm(32, 64)
        self.relu1 = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # it is slightly better whereas slower to set stride = 1
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=1)

        # AAG:  We remove these layers, beacause we will not use block expansion and all ResNet models will have output_channels=512, also we will apply a set of Dense projections. 
        if self.expansion>1:
            self.conv2 = nn.Conv2d(512 * block.expansion, output_channels, kernel_size=3, stride=1, padding=1, bias=False)  # add
            self.bn2 = nn.GroupNorm(32, output_channels)
            self.relu2 = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d): # AAG: BN was changed on GN
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                #nn.BatchNorm2d(planes * block.expansion),
                nn.GroupNorm(32, planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # 512*block.expansion, H/16, W/16

        # AAG:  use only of exapnsion > 1
        if self.expansion>1:
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu2(x)  # N, 512, H/16, W/16

        return x


def resnet18(output_channels=512):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], output_channels=output_channels)
    return model

# AAG: added
def se_resnet18(output_channels=512):
    """Constructs a SEResNet-18 model.
    """
    model = ResNet(SEBasicBlock, [2, 2, 2, 2], output_channels=output_channels)
    return model


def resnet34(output_channels=512):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], output_channels=output_channels)
    return model

# AAG: added
def se_resnet34(output_channels=512):
    """Constructs a SEResNet-34 model.
    """
    model = ResNet(SEBasicBlock, [3, 4, 6, 3], output_channels=output_channels)
    return model


def resnet50(output_channels=512):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], output_channels=output_channels)
    return model

# AAG: added
def se_resnet50(output_channels=512):
    """Constructs a SEResNet-50 model.
    """
    model = ResNet(SEBottleneck, [3, 4, 6, 3], output_channels=output_channels)
    return model


def resnet101(output_channels=512):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], output_channels=output_channels)
    return model

# AAG: added
def se_resnet101(output_channels=512):
    """Constructs a SEResNet-101 model.
    """
    model = ResNet(SEBottleneck, [3, 4, 23, 3], output_channels=output_channels)
    return model

def resnet152(output_channels=512):
    """Constructs a ResNet-152 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], output_channels=output_channels)
    return model

# AAG: added
def se_resnet152(output_channels=512):
    """Constructs a SEResNet-152 model.
    """
    model = ResNet(SEBottleneck, [3, 8, 36, 3], output_channels=output_channels)
    return model
