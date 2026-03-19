import torch
import torch.nn as nn
import torch.nn.functional as F

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

# ---------------- SE Block ----------------
class SEBlock(nn.Module):

    def __init__(self, channels, in_H=None, in_W=None, quantiles=None, reduction=2):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=True)
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = torch.sigmoid(self.fc(y)).view(b, c, 1, 1)
        return x * y

# ---------------- UNet with SE ---------------- 
class SmallUNet_AllCA_vSE(nn.Module):
                                    
    def __init__(self, in_ch=3, base_ch=16, quantiles=[0.25, 0.5, 0.75, 0.95]):
        super().__init__()
        self.base_ch = base_ch
        q = quantiles
        # Encoder
        self.enc1 = DoubleCGR(in_ch, base_ch)
        self.ca_e1 = SEBlock(base_ch, in_H=192, in_W=256, quantiles=q)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleCGR(base_ch, base_ch*2)
        self.ca_e2 = SEBlock(base_ch*2, in_H=96, in_W=128, quantiles=q)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleCGR(base_ch*2, base_ch*4)
        self.ca_e3 = SEBlock(base_ch*4, in_H=48, in_W=64, quantiles=q)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleCGR(base_ch*4, base_ch*8)
        self.ca_e4 = SEBlock(base_ch*8, in_H=24, in_W=32, quantiles=q)
        self.pool4 = nn.MaxPool2d(2)
        # Bottleneck
        self.bottleneck = DoubleCGR(base_ch*8, base_ch*16)
        self.ca_bottleneck = SEBlock(base_ch*16, in_H=12, in_W=16, quantiles=q)
        # Decoder
        self.up4 = nn.ConvTranspose2d(base_ch*16, base_ch*8, 2, stride=2)
        self.dec4 = DoubleCGR(base_ch*16, base_ch*8)
        self.ca_d4 = SEBlock(base_ch*8, in_H=24, in_W=32, quantiles=q)
        self.up3 = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, stride=2)
        self.dec3 = DoubleCGR(base_ch*8, base_ch*4)
        self.ca_d3 = SEBlock(base_ch*4, in_H=48, in_W=64, quantiles=q)
        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
        self.dec2 = DoubleCGR(base_ch*4, base_ch*2)
        self.ca_d2 = SEBlock(base_ch*2, in_H=96, in_W=128, quantiles=q)
        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)
        self.dec1 = DoubleCGR(base_ch*2, base_ch)
        self.ca_d1 = SEBlock(base_ch, in_H=192, in_W=256, quantiles=q)
        # Final conv
        self.final_conv = nn.Conv2d(base_ch, 1, 1)
        self.final_act = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e1 = self.ca_e1(e1)
        e2 = self.enc2(self.pool1(e1))
        e2 = self.ca_e2(e2)
        e3 = self.enc3(self.pool2(e2))
        e3 = self.ca_e3(e3)
        e4 = self.enc4(self.pool3(e3))
        e4 = self.ca_e4(e4)
        # Bottleneck
        b  = self.bottleneck(self.pool4(e4))
        b  = self.ca_bottleneck(b)
        # Decoder
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d4 = self.ca_d4(d4)
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d3 = self.ca_d3(d3)
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d2 = self.ca_d2(d2)
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        d1 = self.ca_d1(d1)
        #
        out = self.final_act(self.final_conv(d1))
        return out