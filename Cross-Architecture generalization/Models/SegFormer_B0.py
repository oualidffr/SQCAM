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
        if in_H <= out_H and in_W <= out_W:  # For SegFormer 6x8 feature map (/32) → identity.
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
# =========================       SegFormer-B0    PART   ===========================
# ==================================================================================

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class MixFFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x, H, W):
        x = self.fc1(x)
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.fc2(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.proj = nn.Linear(dim, dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.transpose(1, 2).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).transpose(1, 2)
            x_ = self.norm(x_)
            kv = self.kv(x_)
        else:
            kv = self.kv(x)

        kv = kv.reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

# ----------------------------
# Transformer Block
# ----------------------------
class Block(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio, drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads, sr_ratio)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MixFFN(dim, dim * 4)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

# ----------------------------
# Overlapping Patch Embedding
# ----------------------------
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride, padding):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel, stride, padding)
        self.norm = nn.LayerNorm(out_ch, eps=1e-6)

    def forward(self, x):
        x = self.proj(x)
        H, W = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

# ====================================================================================================
# ====================================================================================================

class MiTB0(nn.Module):
    def __init__(self):
        super().__init__()
        embed_dims = [32, 64, 160, 256]
        depths = [2, 2, 2, 2]
        heads = [1, 2, 5, 8]
        sr_ratios = [8, 4, 2, 1]
        self.patch_embeds = nn.ModuleList([
            OverlapPatchEmbed(3, 32, 7, 4, 3),
            OverlapPatchEmbed(32, 64, 3, 2, 1),
            OverlapPatchEmbed(64, 160, 3, 2, 1),
            OverlapPatchEmbed(160, 256, 3, 2, 1),
        ])
        self.blocks = nn.ModuleList([
            nn.ModuleList([Block(embed_dims[i], heads[i], sr_ratios[i]) for _ in range(depths[i])])
            for i in range(4)
        ])

        # ---------------------------- SQCAM Integration ---------------------------- 
        self.use_channel_attn =  False    # ← Toggle here to apply SQCAM  (False = Baseline) 
        if self.use_channel_attn:
            self.channel_attns = nn.ModuleList([
                SoftQuantileChannelAttention(32, 48, 64),
                SoftQuantileChannelAttention(64, 24, 32),
                SoftQuantileChannelAttention(160, 12, 16),
                SoftQuantileChannelAttention(256, 6, 8),
            ])
        # ----------------------------------------------------------------------------

    def forward(self, x):
        features = []
        for i in range(4):
            x, H, W = self.patch_embeds[i](x)
            for blk in self.blocks[i]:
                x = blk(x, H, W)           
            x = x.transpose(1, 2).reshape(x.size(0), -1, H, W)

            # ---------------------------- SQCAM Integration ----------------------------
            if self.use_channel_attn:
                x = self.channel_attns[i](x)
            # ----------------------------------------------------------------------------

            features.append(x)
        return features


class SegFormerHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(c, 256),
                nn.LayerNorm(256, eps=1e-6)
            ) for c in in_channels
        ])
        self.fuse = nn.Conv2d(256 * 4, 256, 1)
        self.cls = nn.Conv2d(256, num_classes, 1)

    def forward(self, features):
        size = features[0].shape[2:]
        outs = []
        for i, feat in enumerate(features):
            B, C, H, W = feat.shape
            x = feat.flatten(2).transpose(1, 2)
            x = self.mlps[i](x)
            x = x.transpose(1, 2).reshape(B, 256, H, W)
            x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
            outs.append(x)
        x = torch.cat(outs, dim=1)
        x = self.fuse(x)
        return self.cls(x)

# ----------------------------
# SegFormer-B0
# ----------------------------
class SegFormerB0(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.encoder = MiTB0()
        self.decode_head = SegFormerHead([32, 64, 160, 256], num_classes)

    def forward(self, x):
        feats = self.encoder(x)
        out = self.decode_head(feats)
        return F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)