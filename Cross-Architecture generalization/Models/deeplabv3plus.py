import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

# ---------------------------------------------------------------
# ----------------------- START of SQCAM ---------------------- #
# ---------------------------------------------------------------

def soft_quantiles(x, quantiles=[0.25, 0.5, 0.75, 0.95], tau1=0.1, tau2=0.1):
    N = x.shape[-1]
    x_i = x.unsqueeze(-1)
    x_j = x.unsqueeze(-2)
    P = torch.sigmoid((x_i - x_j) / tau1)
    r = 1 + P.sum(-1)
    q_vals = []
    for q in quantiles:
        target_rank = 1 + q * (N - 1)
        w = torch.softmax(-(r - target_rank).abs() / tau2, dim=-1)
        q_val = (w * x).sum(-1)
        q_vals.append(q_val)
    return q_vals


class ChannelReducer(nn.Module):
    def __init__(self, in_ch, in_H, in_W):
        super().__init__()
        self.in_ch = in_ch
        out_H, out_W = 12, 16 
        if in_H == out_H and in_W == out_W:
            self.reduce = nn.Identity()
        else:
            kernel_H = in_H // out_H
            kernel_W = in_W // out_W
            self.reduce = nn.AvgPool2d(kernel_size=(kernel_H, kernel_W),
                                       stride=(kernel_H, kernel_W))
    def forward(self, x):
        return self.reduce(x)


class SoftQuantileChannelAttention(nn.Module):
    def __init__(self, channels, in_H, in_W, quantiles=[0.25, 0.5, 0.75, 0.95]):
        super().__init__()
        self.quantiles = quantiles
        self.Q = len(quantiles)
        self.C = channels

        # Internal reducer
        self.reducer = ChannelReducer(channels, in_H, in_W)

        # Fusion layers for attention stats
        hidden = max(channels // 2, 1)
        self.conv1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden, channels, kernel_size=(1, self.Q), bias=False)
        self.sigmoid = nn.Sigmoid()

        # 1x1 conv for concatenated [x, x*scale] → out_ch
        self.fuse_conv = nn.Conv2d(2*channels, channels, kernel_size=1, bias=False)
        self._init_fuse_conv_identity()

    def _init_fuse_conv_identity(self):
        # Initialize as identity: x passes through, x*scale initially ignored
        with torch.no_grad():
            self.fuse_conv.weight.zero_()
            for c in range(self.C):
                self.fuse_conv.weight[c, c, 0, 0] = 1.0  # pass-through x

    def forward(self, x):
        B, C, H, W = x.shape

        # Apply reducer
        reduced = self.reducer(x)   # → B,C,H',W'

        # Flatten spatial dims
        flat = reduced.view(B, C, -1)

        # Soft quantiles
        q_list = soft_quantiles(flat, quantiles=self.quantiles)

        # Build B,C,1,Q
        stats = torch.stack(q_list, dim=-1).unsqueeze(-2)

        # Fuse #1
        t = self.conv1(stats)
        t = self.relu(t)
        t = self.conv2(t)        
        scale = self.sigmoid(t)  # B,C,1,1

        # Concatenate x and x*scale
        concat = torch.cat([x, x*scale], dim=1)  # B,2C,H,W

        # Fuse #2
        out = self.fuse_conv(concat)  # B,C,H,W

        return out

# ---------------------------------------------------------------
#    --------------------- END of SQCAM ---------------------   #
# ---------------------------------------------------------------

# ==================================================================================
# =========================         DeepLabV3+    PART   ===========================
# ==================================================================================

# ===================== ASPP ===================== 
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, atrous_rates=(6,12,18)):
        super().__init__()
        self.atrous1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.atrous2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rates[0],
                      dilation=atrous_rates[0], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.atrous3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rates[1],
                      dilation=atrous_rates[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.atrous4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rates[2],
                      dilation=atrous_rates[2], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # Image pooling branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.project = nn.Sequential(
            nn.Conv2d(out_channels*5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        size = x.shape[2:]
        x1 = self.atrous1(x)
        x2 = self.atrous2(x)
        x3 = self.atrous3(x)
        x4 = self.atrous4(x)
        x5 = F.interpolate(self.global_pool(x), size=size, mode='bilinear', align_corners=False)
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return self.project(x)

# ===================== Decoder =====================
class Decoder(nn.Module):
    def __init__(self, low_level_channels, num_classes, out_channels=256):
        super().__init__()
        self.low_proj = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels + 48, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Conv2d(out_channels, num_classes, 1)

    def forward(self, low_level_feat, high_level_feat):
        low = self.low_proj(low_level_feat)
        high = F.interpolate(high_level_feat, size=low.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([low, high], dim=1)
        x = self.fuse(x)
        x = self.classifier(x)
        return x

# ===================== DeepLabV3+ =====================
class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=1, backbone='resnet50', output_stride=16):
        super().__init__()
        if backbone == 'resnet50':
            self.backbone = resnet50(pretrained=True, replace_stride_with_dilation=[False, True, True])
            in_channels = 2048
            low_level_channels = 256
        else:
            raise NotImplementedError("Other backbones not implemented yet")

        # ASPP
        if output_stride == 16:
            rates = (6, 12, 18)
        else:
            raise ValueError("output_stride must be 16")

        self.aspp = ASPP(in_channels, 256, atrous_rates=rates)
        self.decoder = Decoder(low_level_channels, num_classes)
        
        # SQCAM Integration :  # -------- SQCAM @ Layer4 (before ASPP) -------- #    
        # self.sqcam_layer4 = SoftQuantileChannelAttention(channels=2048, in_H=24, in_W=32) # ← Uncomment to enable SQCAM at Layer4

    def forward(self, x):
        # Extract features from ResNet backbone
        x_size = x.shape[2:]

        # ResNet forward 
        x0 = self.backbone.conv1(x)
        x0 = self.backbone.bn1(x0)
        x0 = self.backbone.relu(x0)
        x0 = self.backbone.maxpool(x0)

        x1 = self.backbone.layer1(x0)  # low-level
        x2 = self.backbone.layer2(x1)
        x3 = self.backbone.layer3(x2)
        x4 = self.backbone.layer4(x3)  # high-level

        # SQCAM Integration :  # -------- SQCAM @ Layer4 (before ASPP) -------- #  
        # x4 = self.sqcam_layer4(x4)  # ← Uncomment to enable SQCAM at Layer4
        
        high = self.aspp(x4)
        out = self.decoder(x1, high)
        out = F.interpolate(out, size=x_size, mode='bilinear', align_corners=False)
        return out