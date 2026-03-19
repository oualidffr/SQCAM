import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

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
# =========================       ResNet18_FPN    PART   ===========================
# ==================================================================================

class ResNet18_Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet18(pretrained=pretrained)
        # Initial layers
        self.conv1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        # ResNet stages
        self.layer1 = resnet.layer1  
        self.layer2 = resnet.layer2  
        self.layer3 = resnet.layer3  
        self.layer4 = resnet.layer4

    def forward(self, x):
        x = self.conv1(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c2, c3, c4, c5


class FPN(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()
        # Lateral 1x1 convs
        self.lateral2 = nn.Conv2d(64,  out_channels, 1)
        self.lateral3 = nn.Conv2d(128, out_channels, 1)
        self.lateral4 = nn.Conv2d(256, out_channels, 1)
        self.lateral5 = nn.Conv2d(512, out_channels, 1)
        # 3x3 smoothing convs
        self.smooth2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        #self.smooth3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        #self.smooth4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        #self.smooth5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, c2, c3, c4, c5):
        # Top-down pathway
        p5 = self.lateral5(c5)
        p4 = self.lateral4(c4) + F.interpolate(p5, scale_factor=2, mode="nearest")
        p3 = self.lateral3(c3) + F.interpolate(p4, scale_factor=2, mode="nearest")
        p2 = self.lateral2(c2) + F.interpolate(p3, scale_factor=2, mode="nearest")
        # 3x3 smoothing
        #p5 = self.smooth5(p5)
        #p4 = self.smooth4(p4)
        #p3 = self.smooth3(p3)
        p2 = self.smooth2(p2)

        return p2, p3, p4, p5

class FPN_Segmentation(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = ResNet18_Backbone(pretrained=pretrained)
        self.fpn = FPN(out_channels=256)
       
        # SQCAM Integration : ---------------- (for P2 only) ----------------
        #self.sqcam_p2 = SoftQuantileChannelAttention(channels=256, in_H=48, in_W=64) # ← Uncomment to enable SQCAM for P2
  
        # Simple segmentation head
        self.head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )

    def forward(self, x):
        input_size = x.shape[-2:]
        c2, c3, c4, c5 = self.backbone(x)
        p2, p3, p4, p5 = self.fpn(c2, c3, c4, c5)

        # SQCAM Integration : ---------------- (for P2 only) ----------------
        #p2 = self.sqcam_p2(p2) # ← Uncomment to enable SQCAM for P2

        out = self.head(p2)
        out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)
        return out