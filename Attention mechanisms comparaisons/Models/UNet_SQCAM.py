import torch
import torch.nn as nn
import torch.nn.functional as F

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

class CGR(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        num_groups = max(1, out_ch // 8)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class DoubleCGR(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block1 = CGR(in_ch, out_ch)
        self.block2 = CGR(out_ch, out_ch)
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x

# ---------------- UNet with SQCAM ---------------- 
class SmallUNet_AllCA(nn.Module):

    def __init__(self, in_ch=3, base_ch=16, quantiles=[0.25, 0.5, 0.75, 0.95]):
        super().__init__()
        self.base_ch = base_ch
        q = quantiles

        # Encoder
        self.enc1 = DoubleCGR(in_ch, base_ch)            # e1: C=base_ch, HxW=192x256
        self.ca_e1 = SoftQuantileChannelAttention(base_ch, in_H=192, in_W=256, quantiles=q)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleCGR(base_ch, base_ch*2)        # e2: C=base_ch*2, 96x128
        self.ca_e2 = SoftQuantileChannelAttention(base_ch*2, in_H=96, in_W=128, quantiles=q)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleCGR(base_ch*2, base_ch*4)      # e3: C=base_ch*4, 48x64
        self.ca_e3 = SoftQuantileChannelAttention(base_ch*4, in_H=48, in_W=64, quantiles=q)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = DoubleCGR(base_ch*4, base_ch*8)      # e4: C=base_ch*8, 24x32
        self.ca_e4 = SoftQuantileChannelAttention(base_ch*8, in_H=24, in_W=32, quantiles=q)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleCGR(base_ch*8, base_ch*16)  # b: C=base_ch*16, 12x16
        self.ca_bottleneck = SoftQuantileChannelAttention(base_ch*16, in_H=12, in_W=16, quantiles=q)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base_ch*16, base_ch*8, 2, stride=2)
        self.dec4 = DoubleCGR(base_ch*16, base_ch*8)
        self.ca_d4 = SoftQuantileChannelAttention(base_ch*8, in_H=24, in_W=32, quantiles=q)

        self.up3 = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, stride=2)
        self.dec3 = DoubleCGR(base_ch*8, base_ch*4)
        self.ca_d3 = SoftQuantileChannelAttention(base_ch*4, in_H=48, in_W=64, quantiles=q)

        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
        self.dec2 = DoubleCGR(base_ch*4, base_ch*2)
        self.ca_d2 = SoftQuantileChannelAttention(base_ch*2, in_H=96, in_W=128, quantiles=q)

        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)
        self.dec1 = DoubleCGR(base_ch*2, base_ch)
        self.ca_d1 = SoftQuantileChannelAttention(base_ch, in_H=192, in_W=256, quantiles=q)

        # Final conv
        self.final_conv = nn.Conv2d(base_ch, 1, 1)
        self.final_act = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)                       # 192x256
        e1 = self.ca_e1(e1)
        e2 = self.enc2(self.pool1(e1))          # 96x128
        e2 = self.ca_e2(e2)
        e3 = self.enc3(self.pool2(e2))          # 48x64
        e3 = self.ca_e3(e3)
        e4 = self.enc4(self.pool3(e3))          # 24x32
        e4 = self.ca_e4(e4)

        # Bottleneck
        b  = self.bottleneck(self.pool4(e4))    # 12x16
        b  = self.ca_bottleneck(b)

        # Decoder
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))  # 24x32
        d4 = self.ca_d4(d4)

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))  # 48x64
        d3 = self.ca_d3(d3)

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))  # 96x128
        d2 = self.ca_d2(d2)

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))  # 192x256 
        d1 = self.ca_d1(d1)

        out = self.final_act(self.final_conv(d1))
        return out