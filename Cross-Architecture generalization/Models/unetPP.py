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

# ==================================================================================
# =========================          UNet++    PART      ===========================
# ==================================================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        num_groups = max(1, out_ch // 8)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True),
            nn.GroupNorm(num_groups, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True),
            nn.GroupNorm(num_groups, out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

# ---------------------------------------------------------------
# U-Net++ with SQCAM (commented = Baseline)
# ---------------------------------------------------------------
class UNetPP_SQCAM(nn.Module):
    def __init__(self, in_ch=3, num_classes=1, base_ch=16):
        super().__init__()
        nb = [base_ch, base_ch*2, base_ch*4, base_ch*8, base_ch*16]
        self.pool = nn.MaxPool2d(2)
        self.up   = lambda x: F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        # ---------------- Encoder X_i,0 ----------------
        self.conv0_0 = ConvBlock(in_ch, nb[0])
        self.conv1_0 = ConvBlock(nb[0], nb[1])
        self.conv2_0 = ConvBlock(nb[1], nb[2])
        self.conv3_0 = ConvBlock(nb[2], nb[3])
        self.conv4_0 = ConvBlock(nb[3], nb[4]) # Bottle.

        # ---------------- Decoder nested nodes ---------
        self.conv0_1 = ConvBlock(nb[0]+nb[1], nb[0])
        self.conv1_1 = ConvBlock(nb[1]+nb[2], nb[1])
        self.conv2_1 = ConvBlock(nb[2]+nb[3], nb[2])
        self.conv3_1 = ConvBlock(nb[3]+nb[4], nb[3]) 
        #
        self.conv0_2 = ConvBlock(nb[0]*2+nb[1], nb[0])
        self.conv1_2 = ConvBlock(nb[1]*2+nb[2], nb[1])
        self.conv2_2 = ConvBlock(nb[2]*2+nb[3], nb[2]) 
        #
        self.conv0_3 = ConvBlock(nb[0]*3+nb[1], nb[0])
        self.conv1_3 = ConvBlock(nb[1]*3+nb[2], nb[1]) 
        #
        self.conv0_4 = ConvBlock(nb[0]*4+nb[1], nb[0]) 

        """ # ---------------- SQCAM Integration ---------------- #

                        # Encoder attention #
        self.ca_e0 = SoftQuantileChannelAttention(nb[0], 192, 256)
        self.ca_e1 = SoftQuantileChannelAttention(nb[1], 96, 128)
        self.ca_e2 = SoftQuantileChannelAttention(nb[2], 48, 64)
        self.ca_e3 = SoftQuantileChannelAttention(nb[3], 24, 32)

                        # Bottle attention #
        self.ca_bottleneck = SoftQuantileChannelAttention(nb[4], 12, 16)

                        # Decoder attention #
        self.ca_d0_1 = SoftQuantileChannelAttention(nb[3], 24, 32)
        self.ca_d0_2 = SoftQuantileChannelAttention(nb[2], 48, 64)
        self.ca_d0_3 = SoftQuantileChannelAttention(nb[1], 96, 128)
        self.ca_d0_4 = SoftQuantileChannelAttention(nb[0], 192, 256)

        # ------------ Uncomment for Integration ------------ # """

        # RETURNS (Paper Deep Supervision)
        self.ds0_1 = nn.Conv2d(nb[0], num_classes, kernel_size=1)
        self.ds0_2 = nn.Conv2d(nb[0], num_classes, kernel_size=1)
        self.ds0_3 = nn.Conv2d(nb[0], num_classes, kernel_size=1)
        self.ds0_4 = nn.Conv2d(nb[0], num_classes, kernel_size=1)


    def forward(self, x):

        # ---------------- Encoder ----------------
        x0_0 = self.conv0_0(x)
        #x0_0 = self.ca_e0(x0_0) # ← Uncomment

        x1_0 = self.conv1_0(self.pool(x0_0))
        #x1_0 = self.ca_e1(x1_0) # ← Uncomment

        x2_0 = self.conv2_0(self.pool(x1_0))
        #x2_0 = self.ca_e2(x2_0) # ← Uncomment

        x3_0 = self.conv3_0(self.pool(x2_0))
        #x3_0 = self.ca_e3(x3_0) # ← Uncomment

        x4_0 = self.conv4_0(self.pool(x3_0))
        #x4_0 = self.ca_bottleneck(x4_0) # ← Uncomment

        # ---------------- Nested Decoder ----------------
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        #    
        #x3_1 = self.ca_d0_1(x3_1) # ← Uncomment

        
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        #
        #x2_2 = self.ca_d0_2(x2_2) # ← Uncomment

        
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        #
        #x1_3 = self.ca_d0_3(x1_3) # ← Uncomment

        
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        #
        #x0_4 = self.ca_d0_4(x0_4) # ← Uncomment

        #   #   #    #    #   #   #   #   
        out1 = self.ds0_1(x0_1)
        out2 = self.ds0_2(x0_2)
        out3 = self.ds0_3(x0_3)
        out4 = self.ds0_4(x0_4)
        return [out1, out2, out3, out4]