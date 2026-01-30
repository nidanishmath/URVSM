import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================= Blocks =========================

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, max(channels // reduction, 4), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // reduction, 4), channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.avg(x))


class DOConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.pw = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.dw = nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False)
        self.dw_to_out = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else None

    def forward(self, x):
        y = self.pw(x) + (self.dw_to_out(self.dw(x)) if self.dw_to_out else self.dw(x))
        return y


class ResDOBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = DOConv2d(in_ch, out_ch)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = DOConv2d(out_ch, out_ch, p=2)  # dilation-like effect
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.proj = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.proj(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


class PFB(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.avg = nn.AvgPool2d(2)
        self.max = nn.MaxPool2d(2)
        self.squeeze = nn.Conv2d(2 * ch, ch, 1)
        self.bn = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = torch.cat([self.avg(x), self.max(x)], dim=1)
        return F.relu(self.bn(self.squeeze(y)))


class AFB(nn.Module):
    def __init__(self, low_ch, high_ch, out_ch):
        super().__init__()
        in_ch = low_ch + high_ch
        mid = max(out_ch // 2, 32)

        def branch(d):
            return nn.Sequential(
                nn.Conv2d(in_ch, mid, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(mid),
                nn.ReLU(inplace=True),
            )

        # Fine-scale emphasis
        self.b1 = branch(1)
        self.b2 = branch(1)
        self.b3 = branch(2)

        self.se = SEBlock(3 * mid)
        self.fuse = nn.Sequential(
            nn.Conv2d(3 * mid, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, low, high):
        if low.shape[-2:] != high.shape[-2:]:
            high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([low, high], dim=1)
        y = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)
        return self.fuse(self.se(y))

# ========================= Network =========================

class ResDO_UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base_ch=16):
        super().__init__()
        c1, c2, c3, c4 = base_ch, base_ch*2, base_ch*4, base_ch*8

        # Encoder
        self.enc1 = ResDOBlock(in_ch, c1)
        self.pfb1 = PFB(c1)

        self.enc2 = ResDOBlock(c1, c2)
        self.pfb2 = PFB(c2)

        self.enc3 = ResDOBlock(c2, c3)

        # Bottleneck (NO extra pooling)
        self.bott = ResDOBlock(c3, c4)

        # Decoder
        self.afb3 = AFB(c3, c4, c3)
        self.dec3 = ResDOBlock(c3, c3)

        self.afb2 = AFB(c2, c3, c2)
        self.dec2 = ResDOBlock(c2, c2)

        self.afb1 = AFB(c1, c2, c1)
        self.dec1 = ResDOBlock(c1, c1)

        self.head = nn.Conv2d(c1, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pfb1(e1))
        e3 = self.enc3(self.pfb2(e2))

        b = self.bott(e3)

        x3 = self.dec3(self.afb3(e3, b))
        x2 = self.dec2(self.afb2(e2, x3))
        x1 = self.dec1(self.afb1(e1, x2))

        return self.head(x1)  # LOGITS (no sigmoid)
