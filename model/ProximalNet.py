import torch
import torch.nn as nn
import math
from .MwinMambaBlock import MultiWinMambaBlocks
import torch.nn.functional as F


class ResBlock(nn.Module):
    """
    Enhanced Residual Block with BatchNorm and improved structure.
    Structure: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> (+ Input)
    If if_norm=True: Conv -> ReLU -> Conv -> (+ Input) (no BatchNorm)
    """
    def __init__(self, channels, kernel_size=3, padding=1, bias=False, if_norm=True):
        super(ResBlock, self).__init__()
        self.if_norm = if_norm
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, bias=bias)
        if if_norm:
            self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, bias=bias)
        if if_norm:
            self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        if self.if_norm:
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
        else:
            out = self.relu(self.conv1(x))
            out = self.conv2(out)
        return identity + out  # Local Residual Connection

class ResBlockSequence(nn.Module):
    """
    Sequence of residual blocks for deeper feature extraction.
    """
    def __init__(self, channels, num_blocks=3, if_norm=True):
        super(ResBlockSequence, self).__init__()
        self.blocks = nn.ModuleList([
            ResBlock(channels, if_norm=if_norm) for _ in range(num_blocks)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class TransitionBlock(nn.Module):
    """
    Transition block for changing number of channels.
    Structure: Conv -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False):
        super(TransitionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                             padding=padding, stride=stride, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))



class SoftThresholding(nn.Module):
    """
    Implementation based on TransCS paper (Shen et al., 2022).
    Equation (13): x_soft = sign(x) * ELU(|x| - theta)
    Key difference from standard ISTA:
    Uses ELU instead of ReLU to avoid zero-gradient issues and maintain
    information flow even when signal is below threshold.
    """

    def __init__(self, num_features):
        super(SoftThresholding, self).__init__()
        # Initialize learnable threshold parameter
        # Using a scalar threshold for simplicity, but can be expanded to (1, C, 1, 1)
        self.threshold = nn.Parameter(torch.tensor(0.01), requires_grad=True)

    def forward(self, x):
        # 1. Calculate magnitude: |x|
        magnitude = torch.abs(x)

        # 2. Shift by threshold: |x| - zeta
        shifted = magnitude - self.threshold

        # 3. Apply ELU activation (Paper Eq. 13 uses ELU instead of ReLU)
        # ELU(z) = z if z > 0 else alpha * (exp(z) - 1)
        activation = F.elu(shifted)

        # 4. Restore sign: sign(x) * ELU(|x| - zeta)
        return torch.sign(x) * activation



class SafeSoftThresholding(nn.Module):
    """
    Learnable smooth soft thresholding operator with safe interval constraints.
    Formula: y = sign(x) * Softplus(|x| - lambda)
    Constraint: lambda in [min_val, max_val]
    """
    def __init__(self, num_features, init_val=0.1, min_val=0.01, max_val=0.5):
        super(SafeSoftThresholding, self).__init__()
        
        self.min_val = min_val
        self.max_val = max_val
        self.num_features = num_features
        
        # --- 1. Parameter Initialization ---
        # We want the initial value of lambda to be exactly init_val
        # Formula: lambda = min + (max - min) * sigmoid(w)
        # Inverse to get w (logit): w = ln(p / (1-p)), where p = (init - min) / (max - min)
        if not (min_val < init_val < max_val):
            raise ValueError(f"init_val {init_val} must be between {min_val} and {max_val}")
            
        # Calculate the normalized position p (between 0~1)
        p = (init_val - min_val) / (max_val - min_val)
        # Calculate inverse (Logit)
        init_w = math.log(p / (1.0 - p))
        
        # Define the learnable parameter w, shape (1, C, 1, 1) for broadcasting
        self.threshold_param = nn.Parameter(torch.full((1, num_features, 1, 1), init_w))

    def forward(self, x):
        """
        Args:
            x: Input tensor (B, C, H, W)
        Returns:
            out: Tensor after thresholding (B, C, H, W)
        """
        # --- 2. Compute constrained actual threshold ---
        # Sigmoid maps parameter to (0, 1), then linearly maps to (min, max)
        lambd = self.min_val + (self.max_val - self.min_val) * torch.sigmoid(self.threshold_param)
        
        # --- 3. Apply smooth soft thresholding ---
        # Compute magnitude
        magnitude = torch.abs(x)
        
        # Smooth shrinkage: Softplus(|x| - lambda)
        # beta=20 makes it very close to ReLU but maintains smooth gradients
        magnitude_out = F.softplus(magnitude - lambd, beta=20)
        
        # Restore the sign
        return torch.sign(x) * magnitude_out


class ProximalMamba_18(nn.Module):
    """
    Deep Proximal Network with hierarchical feature extraction.
    Architecture: Multi-scale forward transform -> Soft Thresholding -> Multi-scale inverse transform
    Flow: Input -> Forward Trans (32->64->128) -> TransCS Soft Threshold -> Inverse Trans (128->64->32) -> + Input
    """
    def __init__(self, img_size=256, in_channels=1, base_features=16):
        super(ProximalMamba_18, self).__init__()

        # Feature dimensions for hierarchical structure
        self.feat_1 = base_features      # 32/2
        self.feat_2 = base_features * 2  # 64/2
        self.feat_3 = base_features * 4  # 128/2

        # --- 1. Forward Transform Module F(x) - Multi-scale feature extraction ---
        # Stage 1: Initial feature extraction
        self.fwd_head = TransitionBlock(in_channels, self.feat_1)
        self.fwd_stage1 = ResBlockSequence(self.feat_1, num_blocks=2)

        # Stage 2: Feature expansion and deeper processing
        self.fwd_trans12 = TransitionBlock(self.feat_1, self.feat_2, stride=1)  # No downsampling for simplicity
        self.fwd_stage2 = ResBlockSequence(self.feat_2, num_blocks=3)

        # Stage 3: Deep feature representation
        self.fwd_trans23 = TransitionBlock(self.feat_2, self.feat_3, stride=1)
        self.fwd_stage3 = ResBlockSequence(self.feat_3, num_blocks=4, if_norm=True)

        self.enMamba = MultiWinMambaBlocks(dim=self.feat_3, window_sizes=[img_size//16, img_size//16, img_size])

        # --- 2. Proximal Operator (TransCS Style) ---
        self.soft_threshold = SoftThresholding(self.feat_3)

        self.deMamba = MultiWinMambaBlocks(dim=self.feat_3, window_sizes=[img_size//16, img_size//16, img_size])

        # --- 3. Inverse Transform Module F_tilde(x) - Multi-scale reconstruction ---
        # Stage 3: Deep feature reconstruction
        self.inv_stage3 = ResBlockSequence(self.feat_3, num_blocks=4, if_norm=True)

        # Stage 2: Feature reduction and reconstruction
        self.inv_trans32 = TransitionBlock(self.feat_3, self.feat_2)
        self.inv_stage2 = ResBlockSequence(self.feat_2, num_blocks=3)

        # Stage 1: Final reconstruction
        self.inv_trans21 = TransitionBlock(self.feat_2, self.feat_1)
        self.inv_stage1 = ResBlockSequence(self.feat_1, num_blocks=2)
        self.inv_tail = nn.Conv2d(self.feat_1, in_channels, kernel_size=3, padding=1)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        # Save input for global residual connection
        input_identity = x

        # === Forward Transform: Multi-scale feature extraction ===
        # Stage 1
        feat1_input = self.fwd_head(x)
        feat1 = self.fwd_stage1(feat1_input)

        # Stage 2
        feat2 = self.fwd_trans12(feat1)
        feat2 = self.fwd_stage2(feat2)

        # Stage 3
        feat3 = self.fwd_trans23(feat2)
        feat3 = self.fwd_stage3(feat3)

        feat3 = self.enMamba(feat3)
        
        # === Proximal Operator: Soft Thresholding ===
        sparse_feat3 = self.soft_threshold(feat3)

        recon3 = self.deMamba(sparse_feat3)

        # === Inverse Transform: Multi-scale reconstruction ===
        # Stage 3
        recon3 = self.inv_stage3(recon3)

        # Stage 2
        recon2 = self.inv_trans32(recon3)
        recon2 = self.inv_stage2(recon2)

        # Stage 1
        recon1 = self.inv_trans21(recon2)
        recon1 = self.inv_stage1(recon1)
        output = self.inv_tail(recon1)

        output = self.sigmod(input_identity + output)
   
        # === Global Residual Connection ===
        return output, sparse_feat3


class ProximalMamba_8(nn.Module):
    """
    Deep Proximal Network (feat_3 removed, only feat_1 and feat_2 retained).
    Architecture: Multi-scale forward transform -> Soft Thresholding -> Multi-scale inverse transform
    Flow: Input -> Forward Trans (feat_1->feat_2) -> TransCS Soft Threshold -> Inverse Trans (feat_2->feat_1) -> + Input
    """
    def __init__(self, img_size=256, in_channels=1, base_features=32):
        super(ProximalMamba_8, self).__init__()

        # Only two feature scales retained
        self.feat_1 = base_features      # e.g. 16
        self.feat_2 = base_features * 2  # e.g. 32

        # --- 1. Forward Transform Module F(x) - 2-level feature extraction ---
        # Stage 1: Initial feature extraction
        self.fwd_head = TransitionBlock(in_channels, self.feat_1)
        self.fwd_stage1 = ResBlockSequence(self.feat_1, num_blocks=2)

        # Stage 2: Feature expansion and deeper processing
        self.fwd_trans12 = TransitionBlock(self.feat_1, self.feat_2, stride=1)
        self.fwd_stage2 = ResBlockSequence(self.feat_2, num_blocks=2, if_norm=True)
        self.enMamba = MultiWinMambaBlocks(dim=self.feat_2, window_sizes=[img_size//8, img_size//8, img_size])


        # --- 2. Proximal Operator (TransCS Style) ---
        self.soft_threshold = SafeSoftThresholding(self.feat_2)

        self.deMamba = MultiWinMambaBlocks(dim=self.feat_2, window_sizes=[img_size//8, img_size//8, img_size])
        # --- 3. Inverse Transform Module F_tilde(x) - 2-level reconstruction ---
        # Stage 2: Feature reconstruction
        self.inv_stage2 = ResBlockSequence(self.feat_2, num_blocks=2, if_norm=True)
        # Final reconstruction
        self.inv_trans21 = TransitionBlock(self.feat_2, self.feat_1)
        self.inv_stage1 = ResBlockSequence(self.feat_1, num_blocks=2)
        self.inv_tail = nn.Conv2d(self.feat_1, in_channels, kernel_size=3, padding=1)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        # Save input for global residual connection
        input_identity = x

        # === Forward Transform ===
        feat1_input = self.fwd_head(x)
        feat1 = self.fwd_stage1(feat1_input)

        feat2 = self.fwd_trans12(feat1)
        feat2 = self.fwd_stage2(feat2)
        feat2 = self.enMamba(feat2)

        # === Proximal Operator: Soft Thresholding ===
        sparse_feat2 = self.soft_threshold(feat2)

        recon2 = self.deMamba(sparse_feat2)

        # === Inverse Transform: 2-level reconstruction ===
        recon2 = self.inv_stage2(recon2)
        recon1 = self.inv_trans21(recon2)
        recon1 = self.inv_stage1(recon1)
        output = self.inv_tail(recon1)

        output = self.sigmod(input_identity + output)
   
        # === Global Residual Connection ===
        return output, sparse_feat2