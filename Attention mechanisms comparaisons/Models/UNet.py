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

# ---------------- UNet Model ---------------- 
class SmallUNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=16):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(CGR(in_ch, base_ch), CGR(base_ch, base_ch))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(CGR(base_ch, base_ch*2), CGR(base_ch*2, base_ch*2))
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(CGR(base_ch*2, base_ch*4), CGR(base_ch*4, base_ch*4))
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = nn.Sequential(CGR(base_ch*4, base_ch*8), CGR(base_ch*8, base_ch*8))
        self.pool4 = nn.MaxPool2d(2)
        # Bottleneck
        self.bottleneck = DoubleCGR(base_ch*8, base_ch*16)
        # Decoder
        self.up4 = nn.ConvTranspose2d(base_ch*16, base_ch*8, 2, stride=2)
        self.dec4 = DoubleCGR(base_ch*16, base_ch*8)
        self.up3 = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, stride=2)
        self.dec3 = DoubleCGR(base_ch*8, base_ch*4)
        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
        self.dec2 = DoubleCGR(base_ch*4, base_ch*2)
        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)
        self.dec1 = DoubleCGR(base_ch*2, base_ch)
        # Final conv
        self.final_conv = nn.Conv2d(base_ch, 1, 1)
        self.final_act = nn.Sigmoid()

    def forward(self, x):
        h, w = x.shape[2:]

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))
        # Decoder 
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        #
        out = self.final_act(self.final_conv(d1))
        return out