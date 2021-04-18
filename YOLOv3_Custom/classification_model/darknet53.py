import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class DarkResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(DarkResidualBlock, self).__init__()
        reduced_dim = in_channels // 2
        self.layers1 = Conv(in_channels, reduced_dim, 1, 1, 0)
        self.layers2 = Conv(reduced_dim, in_channels, 3, 1, 1)

    def forward(self, x):
        residual = x
        out = self.layers1(x)
        out = self.layers2(out)
        out += residual  # leaky relu 이후에 skip connection이 되었는데 문제가 되지 않은것 같다.
        return out

class Darknet53(nn.Module):
    def __init__(self, block, num_classes):
        super(Darknet53, self).__init__()

        self.conv1 = Conv(3, 32, 3, 1, 1)
        self.conv2 = Conv(32, 64, 3, 2, 1)
        self.residual_block1 = self._make_layer(block, in_channels=64, num_blocks=1)
        self.conv3 = Conv(64, 128, 3, 2, 1)
        self.residual_block2 = self._make_layer(block, in_channels=128, num_blocks=2)
        self.conv4 = Conv(128, 256, 3, 2, 1)
        self.residual_block3 = self._make_layer(block, in_channels=256, num_blocks=8)
        self.conv5 = Conv(256, 512, 3, 2, 1)
        self.residual_block4 = self._make_layer(block, in_channels=512, num_blocks=8)
        self.conv6 = Conv(512, 1024, 3, 2, 1)
        self.residual_block5 = self._make_layer(block, in_channels=1024, num_blocks=4)
        self.global_avg_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(1024, num_classes)


    def _make_layer(self, block, in_channels, num_blocks):
        layers = []
        for i in range(num_blocks):
            layers.append(block(in_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.residual_block1(out)
        out = self.conv3(out)
        out = self.residual_block2(out)
        out = self.conv4(out)
        out = self.residual_block3(out)
        out = self.conv5(out)
        out = self.residual_block4(out)
        out = self.conv6(out)
        out = self.residual_block5(out)
        out = self.global_avg_pool(out)
        out = out.view(-1, 1024)
        out = self.fc(out)

        return out


def darknet53(num_classes):
    return Darknet53(DarkResidualBlock, num_classes)


if __name__ == '__main__':
    model = darknet53(1000)
    inputs = torch.rand((4, 3, 256, 256))
    outputs = model(inputs)
    assert outputs.shape == (4, 1000)
    print("Success!!")
