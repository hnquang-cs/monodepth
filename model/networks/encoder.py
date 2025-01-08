import torch.nn as nn

class BottleNeck(nn.Module):
    """
    BottleNeck building block implementation as description in the original paper "Deep residual learning for image recognition".
    Paper link: https://arxiv.org/abs/1512.03385.

    Args:
        in_block_channels (int): size of a block input's channel.
        plane_size (int): plane's size of a layer.
        is_downsample (bool): True if the block is a downsampling block. The blocks "conv3_1", "conv4_1", and "conv5_1" are downsampling block according to the paper. 
    """
    def __init__(
            self,
            in_block_channels=int,
            plane_size=int,
            is_downsampling=bool,
        ) -> None:
        super().__init__()

        # Activation function
        self.relu = nn.ReLU(inplace=True)

        # Conv1x1 layer in BottleNeck
        self.conv1 = nn.Conv2d(in_channels=in_block_channels, out_channels=plane_size, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(num_features=plane_size)

        # Conv3x3 layer in BottleNeck
        self.conv2 = nn.Conv2d(in_channels=plane_size, out_channels=plane_size, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=plane_size)
        
        # Conv1x1 layer in BottleNeck
        stride = 2 if is_downsampling else 1
        self.conv3 = nn.Conv2d(in_channels=plane_size, out_channels=plane_size*4, kernel_size=1, stride=stride)
        self.bn3 = nn.BatchNorm2d(num_features=plane_size*4)

        # Residual path downsamepler
        self.downsampler = nn.Sequential(
            nn.Conv2d(in_channels=in_block_channels, out_channels=plane_size*4, kernel_size=1, stride=2 if is_downsampling else 1),
            nn.BatchNorm2d(num_features=plane_size*4)
        )
        
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

        if self.downsampler is not None:
            residual = self.downsampler(residual)
        
        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet50 re-implementation as description in the original paper "Deep residual learning for image recognition".
    Paper link: https://arxiv.org/abs/1512.03385.

    Args:
        in_channels (int): input image's channel size.
    """
    def __init__(
            self,
            in_channels=int,
        ) -> None:
        super().__init__()

        # Layers of resnet50 as description in table 1 in the original paper.
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace= True)
        )
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            *self.make_layer(in_channels=64, plane_size=64, num_of_block=3, is_downsampling=False).children()
        )
        self.conv3 = self.make_layer(in_channels=256, plane_size=128, num_of_block=4, is_downsampling=True)
        self.conv4 = self.make_layer(in_channels=512, plane_size=256, num_of_block=6, is_downsampling=True)
        self.conv5 = self.make_layer(in_channels=1024, plane_size=512, num_of_block=3, is_downsampling=True)

    def make_layer(
            self,
            in_channels,
            plane_size,
            num_of_block,
            is_downsampling
        ) -> nn.Sequential:
        blocks = []
        # First block
        blocks.append(
            BottleNeck(in_block_channels=in_channels, plane_size=plane_size, is_downsampling=is_downsampling)
        )
        # Other blocks
        for _ in range(1, num_of_block):
            blocks.append(
                BottleNeck(in_block_channels=4*plane_size, plane_size=plane_size, is_downsampling=False)
            )
        # Return layer composed of several blocks
        return nn.Sequential(*blocks)

    def forward(self, x):
        feature_stages = []
        feature_stages.append(x)
        feature_stages.append(self.conv1(feature_stages[-1]))
        feature_stages.append(self.conv2(feature_stages[-1]))
        feature_stages.append(self.conv3(feature_stages[-1]))
        feature_stages.append(self.conv4(feature_stages[-1]))
        feature_stages.append(self.conv5(feature_stages[-1]))
        return feature_stages