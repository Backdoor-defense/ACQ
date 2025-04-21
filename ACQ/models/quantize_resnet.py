import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import math


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Clip_ReLU(nn.ReLU):
    def __init__(self, inplace: bool = False, clip_min=None, clip_max=None, num_step=math.pow(2, 8)):
        super().__init__(inplace)
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.num_step = num_step
        bias = torch.full(self.clip_min.shape, 0.)
        weight = torch.full(self.clip_min.shape, 1.)
        self.bias = nn.Parameter(bias, requires_grad=True)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def forward(self, input: Tensor) -> Tensor:
        shape = list(input.shape)
        shape[0] = 1
        shape = tuple(shape)
        self.shape = shape

        clip_min = self.clip_min.view(shape)
        clip_max = self.clip_max.view(shape)
        self.S = (torch.max(clip_max)-torch.min(clip_min))/self.num_step+1e-8
        input = F.relu(input, inplace=False).clamp(
            clip_min, clip_max)
        return self.quantize(input).clamp(
            clip_min, clip_max)

    def quantize(self, input: Tensor):
        q = torch.round((input)/self.S)
        r = self.S*q
        r = self.weight.view(self.shape)*r+self.bias.view(self.shape)
        return r

    def quantize_nonaverage(self, input: Tensor):
        q = torch.round(torch.log2(input/self.S))
        r = self.S*torch.pow(2, q)
        return r


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, clip_max=None, clip_min=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu1 = Clip_ReLU(
            inplace=True, clip_min=clip_min[0], clip_max=clip_max[0])
        self.relu2 = Clip_ReLU(
            inplace=True, clip_min=clip_min[1], clip_max=clip_max[1])
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.clip_min = clip_min
        self.clip_max = clip_max

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)

        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu2(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, clip_max=None, clip_min=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = Clip_ReLU(
            inplace=True, clip_min=torch.tensor(clip_min['layer0.2']).to('cuda'), clip_max=torch.tensor(clip_max['layer0.2']).to('cuda'))
        self.layer0 = nn.Sequential(self.conv1, self.bn1, self.relu)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        layers_ = list(clip_max.keys())
        self.layer1 = self._make_layer(
            block, 64, layers[0],
            clip_min=[torch.tensor(clip_min[layer]).to('cuda')
                      for layer in layers_ if 'layer1' in layer],
            clip_max=[torch.tensor(clip_max[layer]).to('cuda')
                      for layer in layers_ if 'layer1' in layer]
        )

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0],
                                       clip_min=[torch.tensor(clip_min[layer]).to('cuda')
                                                 for layer in layers_ if 'layer2' in layer],
                                       clip_max=[torch.tensor(clip_max[layer]).to('cuda')
                                                 for layer in layers_ if 'layer2' in layer])

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1],
                                       clip_min=[torch.tensor(clip_min[layer]).to('cuda')
                                                 for layer in layers_ if 'layer3' in layer],
                                       clip_max=[torch.tensor(clip_max[layer]).to('cuda')
                                                 for layer in layers_ if 'layer3' in layer]
                                       )

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2],
                                       clip_min=[torch.tensor(clip_min[layer]).to('cuda')
                                                 for layer in layers_ if 'layer4' in layer],
                                       clip_max=[torch.tensor(clip_max[layer]).to('cuda')
                                                 for layer in layers_ if 'layer4' in layer]
                                       )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        self.output_num = 5

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, clip_max=None, clip_min=None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, clip_max=clip_max[0:2], clip_min=clip_min[0:2]))
        self.inplanes = planes * block.expansion

        for block_idx in range(1, blocks):
            if block_idx < blocks - 1:
                layers.append(block(self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer,
                                    clip_max=clip_max[2 *
                                                      block_idx:2*(block_idx+1)],
                                    clip_min=clip_min[2*block_idx:2*(block_idx+1)]))
            else:
                layers.append(block(self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer,
                                    clip_max=clip_max[2*block_idx:],
                                    clip_min=clip_min[2*block_idx:]))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.layer0(x)

        x_4 = self.layer1(x)

        x_3 = self.layer2(x_4)

        x_2 = self.layer3(x_3)

        x_1 = self.layer4(x_2)

        x = self.avgpool(x_1)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(block, layers, num_classes, clip_max, clip_min, **kwargs):
    model = ResNet(block, layers, num_classes,
                   clip_max=clip_max, clip_min=clip_min, **kwargs)
    return model


def QUANT_ResNet18(num_classes: int = 10, clip_max=None, clip_min=None, **kwargs):
    return _resnet(BasicBlock, [2, 2, 2, 2], num_classes, clip_max=clip_max, clip_min=clip_min, **kwargs)


def QUANT_ResNet34(num_classes: int = 10, **kwargs):
    return _resnet(BasicBlock, [3, 4, 6, 3], num_classes, **kwargs)


def QUANT_ResNet50(num_classes: int = 10, **kwargs):
    return _resnet(Bottleneck, [3, 4, 6, 3], num_classes, **kwargs)


def CLIP_ResNet101(num_classes: int = 10, **kwargs):
    return _resnet(Bottleneck, [3, 4, 23, 3], num_classes, **kwargs)


def CLIP_ResNet152(num_classes: int = 10, **kwargs):
    return _resnet(Bottleneck, [3, 8, 36, 3], num_classes, **kwargs)
