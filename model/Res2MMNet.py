import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from functools import partial
from typing import Optional, Callable
from timm.layers import DropPath, to_2tuple, trunc_normal_
from .MwinMambaBlock import MultiWinMambaBlocks


class TransitionBlock(nn.Module):
    """
    Transition block for changing number of channels.
    Structure: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels=1, out_channels=32, kernel_size=3, padding=1, stride=1, bias=False):
        super(TransitionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                             padding=padding ,padding_mode='reflect' , stride=stride, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                             padding=padding ,padding_mode='reflect' , stride=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return out

class TransitionBlockReverse(nn.Module):
    """
    Two-stage transition block: first 32->32, then 32->1, each with bias, no BatchNorm, only ReLU after first conv.
    Structure: Conv(32->32, bias=True) -> ReLU -> Conv(32->1, bias=True)
    """
    def __init__(self, in_channels=32, mid_channels=32, out_channels=1, kernel_size=3, padding=1, stride=1):
        super(TransitionBlockReverse, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size,
                               padding=padding, padding_mode='reflect', stride=stride, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size,
                               padding=padding , padding_mode='reflect', stride=1, bias=True)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return out


class Res2MambaBlock(nn.Module):
    def __init__(self, feat_channels=32, img_size = 256):
        super(Res2MambaBlock, self).__init__()

        # Branch 1: Preserve original resolution (Stride 1)
        self.branch1 = nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1)
        self.B1Mamba = MultiWinMambaBlocks(dim=feat_channels, window_sizes=[img_size//8, img_size//8, img_size])

        # Branch 2: Downsample by a factor of 2 (Stride 2), use 3x3 convolution with stride=2
        self.branch2 = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1)
        )
        self.B2Mamba = MultiWinMambaBlocks(dim=feat_channels, window_sizes=[img_size//16, img_size//16, img_size//2])
        self.branch2_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1)
        )

        # Branch 3: Downsample by a factor of 4, two convolutions with stride=2
        self.branch3 = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1)
        )
        self.B3Mamba = MultiWinMambaBlocks(dim=feat_channels, window_sizes=[img_size//32, img_size//32, img_size//4])
        self.branch3_upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1)
        ) 

    def forward(self, x):
        # Branch 1
        out1 = self.branch1(x)
        # Branch 2
        out2 = self.branch2(x)
        # Branch 3
        out3 = self.branch3(x)
        out3_mamba = self.B3Mamba(out3)
        out3_2 = self.branch3_upsample1(out3_mamba)
        out2_mamba = self.B2Mamba(out2+out3_2)
        out2_1 = self.branch2_upsample(out2_mamba)
        out1_mamba = self.B1Mamba(out1+out2_1)

        return out1_mamba


class Res2MambaBlock2(nn.Module):
    def __init__(self, feat_channels=32, img_size = 256):
        super(Res2MambaBlock2, self).__init__()

        # Branch 1: Preserve original resolution (Stride 1)
        self.branch1 = nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1,padding_mode='reflect' )
        self.B1Mamba = MultiWinMambaBlocks(dim=feat_channels, window_sizes=[img_size//8, img_size//8, img_size])

        # Branch 2: Downsample by a factor of 2 (Stride 2), use 3x3 convolution with stride=2
        self.branch2 = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, stride=2, padding=1,padding_mode='reflect' ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1,padding_mode='reflect' )
        )
        self.B2Mamba = MultiWinMambaBlocks(dim=feat_channels, window_sizes=[img_size//16, img_size//16, img_size//2])
        self.branch2_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1,padding_mode='reflect' )
        )

        # Branch 3: Downsample by a factor of 4, two convolutions with stride=2
        self.branch3 = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, stride=2, padding=1,padding_mode='reflect' ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, stride=2, padding=1,padding_mode='reflect' ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1,padding_mode='reflect' )
        )
        self.B3Mamba = MultiWinMambaBlocks(dim=feat_channels, window_sizes=[img_size//32, img_size//32, img_size//4])
        self.branch3_upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1,padding_mode='reflect' )
        )
        self.branch3_upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1,padding_mode='reflect' )
        )   
        
        self.branch1_out = nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1,padding_mode='reflect' )
        self.branch2_out = nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1,padding_mode='reflect' )
        self.branch3_out = nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1,padding_mode='reflect' )
        self.concat_out = nn.Conv2d(feat_channels * 3, feat_channels, kernel_size=1)

    def forward(self, x):
        # Branch 1
        out1 = self.branch1(x)
        # Branch 2
        out2 = self.branch2(x)
        # Branch 3
        out3 = self.branch3(x)
        out3 = self.B3Mamba(out3)
        out3 = self.branch3_upsample1(out3)
        out2 = self.B2Mamba(out2+out3)
        out2 = self.branch2_upsample(out2)
        out1 = self.B1Mamba(out1+out2)
        out3 = self.branch3_upsample2(out3)
        out3 = self.branch3_out(out3)
        out2 = self.branch2_out(out2)
        out1 = self.branch1_out(out1)
        # After upsampling, out1, out2, and out3 have the same spatial dimensions; concatenate along the channel dimension
        out = torch.cat([out1, out2, out3], dim=1)
        out = self.concat_out(out)
        return out


class Res2MMCNNBlock(nn.Module):
    def __init__(self, feat_channels=32,img_size = 256):
        super(Res2MMCNNBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1 ,padding_mode='reflect'  , bias=False)
        self.bn1 = nn.BatchNorm2d(feat_channels)
        self.res2mab = Res2MambaBlock2(feat_channels, img_size = img_size) 
        
        self.conv2 = nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1 ,padding_mode='reflect'  , bias=False)
        self.bn2 = nn.BatchNorm2d(feat_channels)
        self.conv3 = nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1 ,padding_mode='reflect'  , bias=False)
        self.bn3 = nn.BatchNorm2d(feat_channels)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.res2mab(out)
        out = out + residual
        
        residual = out
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))
        
        return out + residual 



class Res2MMNet(nn.Module):
    """
    Comparison model with only one feature scale (feat_1 retained), no Soft-Thresholding.
    Flow: Input -> Forward Trans (feat_1) -> Inverse Trans (feat_1) -> + Input
    """
    def __init__(self, in_channels=1, base_features=32, Block_num=4, img_W=256, img_H=256):
        super(Res2MMNet, self).__init__()
        self.conv_head = TransitionBlock(in_channels=in_channels, out_channels=base_features)
        self.body = nn.Sequential(*[Res2MMCNNBlock(feat_channels=base_features, img_size=img_H) for _ in range(Block_num)])
        self.conv_tail = TransitionBlockReverse(in_channels=base_features, out_channels=in_channels)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x_head = self.conv_head(x)
        x_body = self.body(x_head)
        x_tail = self.conv_tail(x_body+x_head)
        x_out = self.sigmod(x_tail)
        return x_out