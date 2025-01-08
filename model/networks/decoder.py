import torch.nn as nn
import torch

class UpConv(nn.Module):
    """
    Up-Convolution layer that upscales input composed of several channels
    then applies 2D convolutions over the upscaled signal.

    Args:
        in_channels (int): size of input image's channel.
        scale_factor (int): The channel size should be scaled down with a factor of n^2 for the image to be scaled up with a factor of n.
    """
    def __init__(self, in_channels:int, out_channels:int, scale_factor:int):
        super().__init__()
        self.scale_factor = scale_factor

        # Layers
        self.residual_conv= nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        input = nn.functional.interpolate(x, scale_factor=self.scale_factor)
        residual = self.residual_conv(input)
        out = self.conv1(input)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out

class Decoder(nn.Module):
    """
    Decoder module that applies several 2d convolutions and 2d up-convolutions to the output of encoder module.
    This module return 3 disparity maps corresponding to 3 scales (1:4, 1:2, 1:1) of original image.
    """
    def __init__(
            self,
        ) -> None:
        super().__init__()
        # Stage 5
        self.uconv5 = UpConv(in_channels=2048, out_channels=512, scale_factor=2)
        # Stage 4
        self.iconv4 = nn.Sequential(
            nn.Conv2d(in_channels=512+1024, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True)
        )
        self.uconv4 = UpConv(in_channels=512, out_channels=256, scale_factor=2)
        # Stage 3
        self.iconv3 = nn.Sequential(
            nn.Conv2d(in_channels=256+512, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True)
        )
        self.uconv3 = UpConv(in_channels=256, out_channels=128, scale_factor=2)
        # Stage 2
        self.iconv2 = nn.Sequential(
            nn.Conv2d(in_channels=128+256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True)
        )
        self.disp_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.uconv2 = UpConv(in_channels=128, out_channels=64, scale_factor=2)
        # Stage 1
        self.iconv1 = nn.Sequential(
            nn.Conv2d(in_channels=64+64+1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )
        self.disp_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.uconv1 = UpConv(in_channels=64, out_channels=32, scale_factor=2)
        # Stage 0
        self.uconv_en1 = UpConv(in_channels=64, out_channels=32, scale_factor=2)
        self.iconv0 = nn.Sequential(
            nn.Conv2d(in_channels=32+32+1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True)
        )
        self.disp_conv0 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        feature_stages = x
        # Stage 5
        out_de_5 = self.uconv5(feature_stages[5])
        # Stage 4
        out_de_4 = torch.cat(tensors=(out_de_5, feature_stages[4]), dim=1)
        out_de_4 = self.iconv4(out_de_4)
        out_de_4 = self.uconv4(out_de_4)
        # Stage 3
        out_de_3 = torch.cat(tensors=(out_de_4, feature_stages[3]), dim=1)
        out_de_3 = self.iconv3(out_de_3)
        out_de_3 = self.uconv3(out_de_3)
        # Stage 2
        out_de_2 = torch.cat(tensors=(out_de_3, feature_stages[2]), dim=1)
        out_de_2 = self.iconv2(out_de_2)
        disp_de_2 = self.disp_conv2(out_de_2)
        out_de_2 = self.uconv2(out_de_2)
        # Stage 1
        out_de_1 = torch.cat(
            tensors=(
                out_de_2,
                feature_stages[1],
                nn.functional.interpolate(disp_de_2, scale_factor=2, mode="bilinear")
            ),
            dim=1
        )
        out_de_1 = self.iconv1(out_de_1)
        disp_de_1 = self.disp_conv1(out_de_1)
        out_de_1 = self.uconv1(out_de_1)
        # Stage 0 (same size as original image)
        up_feature_en1 = self.uconv_en1(feature_stages[1])
        out_de_0 = torch.cat(
            tensors=(
                out_de_1,
                up_feature_en1,
                nn.functional.interpolate(disp_de_1, scale_factor=2, mode="bilinear")
            ),
            dim=1
        )
        out_de_0 = self.iconv0(out_de_0)
        disp_de_0 = self.disp_conv0(out_de_0)
            
        # Return 3 disparity maps corresponding to 3 scale (1:4, 1:2, 1:1)
        return [disp_de_2, disp_de_1, disp_de_0]