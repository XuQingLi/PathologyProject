# unet_decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Tuple

def _get_norm(norm: Literal["bn", "in", "gn"], num_channels: int) -> nn.Module:
    if norm == "bn":
        return nn.BatchNorm2d(num_channels)
    elif norm == "in":
        return nn.InstanceNorm2d(num_channels, affine=True)
    elif norm == "gn":
        # 8 组较常用；通道数不足时自动降到 1 组
        groups = min(8, num_channels)
        return nn.GroupNorm(groups, num_channels)
    else:
        raise ValueError(f"Unsupported norm: {norm}")

class DoubleConv(nn.Module):
    """(Conv → Norm → ReLU) × 2
    可选深度可分离卷积，显著降算力/显存占用。
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        norm: Literal["bn", "in", "gn"] = "bn",
        separable: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        mid = out_ch

        if separable:
            # 深度可分离：DW(3x3, groups=in_ch) + PW(1x1)
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
                _get_norm(norm, in_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch, mid, 1, bias=False),
                _get_norm(norm, mid),
                nn.ReLU(inplace=True),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=False),
                _get_norm(norm, mid),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid, out_ch, 1, bias=False),
                _get_norm(norm, out_ch),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, mid, 3, padding=1, bias=False),
                _get_norm(norm, mid),
                nn.ReLU(inplace=True),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(mid, out_ch, 3, padding=1, bias=False),
                _get_norm(norm, out_ch),
                nn.ReLU(inplace=True),
            )

        self.do = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.do(x)
        x = self.conv2(x)
        return x

class Down(nn.Module):
    """下采样：MaxPool(2) → DoubleConv"""
    def __init__(self, in_ch: int, out_ch: int, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))

class Up(nn.Module):
    """上采样：上采样(ConvTranspose 或 双线性) → 拼接 skip → DoubleConv"""
    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        bilinear: bool = True,
        **kwargs
    ):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            # 先上采样再用 1x1 降通道
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_ch, in_ch // 2, kernel_size=1, bias=False),
            )
            in_after_cat = in_ch // 2 + skip_ch
        else:
            # 反卷积
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            in_after_cat = in_ch // 2 + skip_ch

        self.conv = DoubleConv(in_after_cat, out_ch, **kwargs)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # 用插值确保和 skip 尺寸一致（避免奇偶尺寸造成的1px偏差）
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class YourUNetDecoder(nn.Module):
    """
    轻量 UNet-style 解码器：
    - 输入:  [B, C_in, H, W]（c2 特征尺度）
    - 输出:  [B, num_classes, H, W]（与输入同尺寸）
    - 通过 in_proj 将高通道特征先压到较小通道（显存友好）
    - 内部自带 U 形结构（不依赖外部 skip），depth 可配

    推荐设置（大图/显存敏感）：
        base_ch=32~64, depth=2, separable=True, bilinear=True, dropout=0.0~0.1
    """
    def __init__(
        self,
        C_in: int,
        num_classes: int = 1,
        base_ch: int = 64,
        depth: int = 2,
        bilinear: bool = True,
        norm: Literal["bn", "in", "gn"] = "bn",
        separable: bool = True,
        dropout: float = 0.0,
        reduce_first: int = 4,   # 先用 1x1 把 C_in 压到 C_in//reduce_first
    ):
        super().__init__()
        assert depth >= 0, "depth >= 0"
        self.depth = depth
        self.num_classes = num_classes

        inter_ch = max(base_ch, C_in // max(1, reduce_first))
        self.in_proj = nn.Sequential(
            nn.Conv2d(C_in, inter_ch, kernel_size=1, bias=False),
            _get_norm(norm, inter_ch),
            nn.ReLU(inplace=True),
        )

        # Encoder（内部自建 skip）
        enc_chs = [inter_ch]
        for i in range(depth):
            enc_chs.append(min(base_ch * (2 ** i), 256))  # 限制上限，避免爆通道

        self.inc = DoubleConv(enc_chs[0], enc_chs[1], norm=norm, separable=separable, dropout=dropout)

        self.downs = nn.ModuleList()
        for i in range(1, depth):
            self.downs.append(Down(enc_chs[i], enc_chs[i+1], norm=norm, separable=separable, dropout=dropout))

        # Bottleneck
        bottleneck_ch = enc_chs[-1] if depth >= 1 else enc_chs[1]
        self.bottleneck = DoubleConv(bottleneck_ch, bottleneck_ch, norm=norm, separable=separable, dropout=dropout)

        # Decoder
        self.ups = nn.ModuleList()
        dec_in = bottleneck_ch
        for i in reversed(range(1, depth+1)):
            skip_ch = enc_chs[i]  # 对应的 encoder 层通道
            out_ch = enc_chs[i-1]
            self.ups.append(Up(dec_in, skip_ch, out_ch, bilinear=bilinear, norm=norm, separable=separable, dropout=dropout))
            dec_in = out_ch

        # 输出头
        self.outc = nn.Conv2d(enc_chs[0], num_classes, kernel_size=1)

    def _pad_to_multiple(self, x: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
        """将 H,W pad 到 multiple 的倍数，返回 pad 后张量与 pad 参数（用于还原裁剪）"""
        if multiple <= 1:
            return x, (0, 0, 0, 0)
        h, w = x.shape[-2:]
        pad_h = (multiple - (h % multiple)) % multiple
        pad_w = (multiple - (w % multiple)) % multiple
        # F.pad 顺序: (left, right, top, bottom)
        x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x_pad, (0, pad_w, 0, pad_h)

    def _unpad(self, x: torch.Tensor, pad: Tuple[int, int, int, int]) -> torch.Tensor:
        l, r, t, b = pad
        if r > 0:
            x = x[..., :, :-r]
        if b > 0:
            x = x[..., :-b, :]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # 1) 压通道，降低显存 & 算力
        x = self.in_proj(x)

        # 2) 保证尺寸可被 2**depth 整除（U 形结构会多次池化/上采样）
        div = 2 ** max(1, self.depth) if self.depth > 0 else 1
        x, pad = self._pad_to_multiple(x, div)

        # 3) Encoder
        skips = []
        x1 = self.inc(x)       # enc level 1
        skips.append(x1)
        x_enc = x1
        for down in self.downs:
            x_enc = down(x_enc)
            skips.append(x_enc)

        # 4) Bottleneck
        x_bot = self.bottleneck(x_enc)

        # 5) Decoder（对称拼接 skip）
        x_dec = x_bot
        # 对齐 skip 顺序：最后一个 skip 是 bottleneck 前的特征
        for up, skip in zip(self.ups, reversed(skips)):
            x_dec = up(x_dec, skip)

        # 6) 输出 & 去除 padding
        out = self.outc(x_dec)
        out = self._unpad(out, pad)
        # 保证与输入相同大小（极端情况下做一次插值兜底）
        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return out