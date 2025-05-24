import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, base_channels=64):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.inc = ConvolutionBlock(in_channels, base_channels)
        self.down1 = ConvolutionBlock(base_channels, base_channels * 2)
        self.down2 = ConvolutionBlock(base_channels * 2, base_channels * 4)
        self.down3 = ConvolutionBlock(base_channels * 4, base_channels * 8)
        self.down4 = ConvolutionBlock(base_channels * 8, base_channels * 16)

        self.up1 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.conv1 = ConvolutionBlock(base_channels * 16, base_channels * 8)

        self.up2 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.conv2 = ConvolutionBlock(base_channels * 8, base_channels * 4)

        self.up3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.conv3 = ConvolutionBlock(base_channels * 4, base_channels * 2)

        self.up4 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.conv4 = ConvolutionBlock(base_channels * 2, base_channels)

        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Contracting path
        d1 = self.inc(x)
        d2 = self.down1(self.max_pool(d1))
        d3 = self.down2(self.max_pool(d2))
        d4 = self.down3(self.max_pool(d3))
        bottom = self.down4(self.max_pool(d4))

        # Expanding path
        u1 = self.up1(bottom)
        u1 = torch.cat([d4, u1], dim=1)
        u1 = self.conv1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([d3, u2], dim=1)
        u2 = self.conv2(u2)

        u3 = self.up3(u2)
        u3 = torch.cat([d2, u3], dim=1)
        u3 = self.conv3(u3)

        u4 = self.up4(u3)
        u4 = torch.cat([d1, u4], dim=1)
        u4 = self.conv4(u4)

        return self.outc(u4)

    # def _crop_and_concat(self, enc_feat, up_feat):
    #     # Crop encoder feature map to match upsampled size (valid convolutions shrink size)
    #     _, _, H, W = up_feat.shape
    #     enc_feat_cropped = self._center_crop(enc_feat, H, W)
    #     return torch.cat([enc_feat_cropped, up_feat], dim=1)


    # def _center_crop(self, layer, target_height, target_width):
    #     _, _, h, w = layer.size()
    #     delta_h = (h - target_height) // 2
    #     delta_w = (w - target_width) // 2
    #     return layer[:, :, delta_h:delta_h + target_height, delta_w:delta_w + target_width]
