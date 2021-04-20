import torch
import torch.nn as nn
import pdb
import numpy as np
import torchsummary as summary
from backbone import *
import config as cf

'''
Information about architecture config:
Tuple is structured by (filters, kernel_size, stride)
feature_size는 오직 stride에 의해 결정됨.

List is structured by 'B' indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss (3 times)
"U" is for upsampling the feature map and concatenating with a previous layer
'''

config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # 여기까지 Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels//2, kernel_size=1),
                    CNNBlock(channels//2, channels, kernel_size=3, padding=1),
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)
        return x


class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, in_channels*2, kernel_size=3, padding=1),
            CNNBlock(2 * in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1),  # predict feature_map 에서는 batch_norm 사용 하지 않음
        )
        self.num_classes = num_classes

    def forward(self, x):
        # x = (n, 512, 13, 13) or (n, 512, 26, 26) or (n, 512, 52, 52)
        return(
            self.pred(x)  # (n, 27, 13, 13), (n, 27, 26, 26), (n, 27, 52, 52)
            .reshape(x.shape[0], 3, self.num_classes+5, x.shape[2], x.shape[3])  # (n, 3, 9, 13, 13), (n, 3, 9, 26, 26), (n, 3, 9, 52, 52)
            .permute(0, 1, 3, 4, 2)  # (n, 3, 13, 13, 9), (n, 3, 26, 26, 9), (n, 3, 52, 52, 9)
        )


class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=4, pretrained_weight=''):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()
        if pretrained_weight:
            print('Loading Pretrained model!')
            self.load_pretrained_layers(pretrained_weight)

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in config:
            if isinstance(module, tuple):  # General Convolution
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(in_channels,
                             out_channels,
                             kernel_size=kernel_size,
                             stride=stride,
                             padding=1 if kernel_size == 3 else 0,  # same padding
                             )
                )
                in_channels = out_channels

            elif isinstance(module, list):  # Resiual block
                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels,num_repeats=num_repeats,))

            elif isinstance(module, str):
                if module == "S":  # predict layer
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),  # 1024
                        CNNBlock(in_channels, in_channels//2, kernel_size=1),
                        ScalePrediction(in_channels//2, num_classes=self.num_classes),
                    ]  # 결국 Convolution 4번하고 마지막 conv로 predict feature_map을 구함

                    in_channels = in_channels // 2  # 512

                elif module == "U":  # up sampling
                    layers.append(nn.Upsample(scale_factor=2),)  # default = nearest
                    #  upsampling을 한 뒤에 config에 없는 concat을 진행을 하기 때문에
                    # config의 upsampling 다음에 나오는 Conv의 in_channels를 맞춰주기 위해 3을 곱함
                    in_channels = in_channels * 3

        return layers

    def forward(self, x):
        outputs = []  # for each scale
        route_connections = []
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue

            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)  # 값 저장

            elif isinstance(layer, nn.Upsample):
                # upsample 한 후의 결과와 route_connections 맨 뒤에 저장된 값과 concat
                x = torch.cat([x, route_connections[-1]], dim=1)  # concatenate with channels  (n, 768, 26, 26), (n, 384, 52, 52)
                route_connections.pop()

        return outputs  # [(n, 3, 13, 13, 9), (n, 3, 26, 26, 9), (n, 3, 52, 52, 9)]
#
    def load_pretrained_layers(self, pretrained_weight):
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        check_point = torch.load(pretrained_weight, map_location=cf.DEVICE)
        pretrained_state_dict = check_point['state_dict']
        pretrained_param_names = list(check_point['state_dict'].keys())

        for i, param in enumerate(param_names[:312]):  # fc layer 전까지 weight 적용
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]
        self.load_state_dict(state_dict)


if __name__ == '__main__':
    num_classes = 4
    IMAGE_SIZE = 416
    model = YOLOv3(num_classes=num_classes)
    x = torch.randn((2,3,IMAGE_SIZE, IMAGE_SIZE))
    out = model(x)
    assert model(x)[0].shape == (2, 3, IMAGE_SIZE // 32, IMAGE_SIZE // 32, num_classes + 5)
    assert model(x)[1].shape == (2, 3, IMAGE_SIZE // 16, IMAGE_SIZE // 16, num_classes + 5)
    assert model(x)[2].shape == (2, 3, IMAGE_SIZE // 8, IMAGE_SIZE // 8, num_classes + 5)
    summary.summary(model, input_size=(3, 416, 416), device='cpu')  # Total params: 61,539,889
    # print(model)
    print("Success!")

