"""Merlin I3ResNet 编码器 + SegResNetDS 式解码器（E2 / E3 用）。

- 编码器：Merlin release checkpoint 的 `encode_image.i3_resnet.*`（ResNet-152 膨胀成 3D）
- 输入轴序 `(B, 1, X, Y, Z)`：I3ResNet 内部把最后一维当深度，正好是 Merlin 的 `(H, W, D)` 约定，
  返回的 skip 也换回同样的 `(B, C, X, Y, Z)`，不用在外部再 permute
- 解码器照 MONAI SegResNetDS 的设计（E0 用的就是它）：反卷积上采样、skip 相加而不是拼接、
  残差块。第一版用的是 trilinear + concat + 非残差块，val MRE 卡在 ~20 mm 不动，
  E0 同期已经到 6.4 mm，所以换成本版
- 编码器 Z 方向只降 4 次（160→10），X/Y 降 5 次；解码器最后两级只在 X/Y 上 2×
- 编码器 BN 是否 eval 由调用方决定：默认 eval + affine 冻结；E2 的 AdaBN 变体（`train()`
 保持 BN train）用 CTA 数据重新估计 running stats，实测比冻结 BN 好很多
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torchvision

_REPO = Path(__file__).resolve().parents[1]


def _load_i3res():
    """只加载 Merlin 的 i3res.py / inflate.py。

    `merlin` 包的 __init__ 会连带导入 transformers / nltk（文本编码器），训练用不到。
    这里给 'merlin.models' 造一个空壳父包，直接把两个文件按路径加载进 sys.modules。
    """
    root = _REPO / "Merlin" / "merlin" / "models"
    for n in ("merlin", "merlin.models"):
        m = sys.modules.setdefault(n, types.ModuleType(n))
        m.__path__ = []

    def load(name, fn):
        spec = importlib.util.spec_from_file_location(name, root / fn)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    sys.modules["merlin.models"].inflate = load("merlin.models.inflate", "inflate.py")
    return load("merlin.models.i3res", "i3res.py").I3ResNet


I3ResNet = _load_i3res()

DEFAULT_CKPT = _REPO / "Merlin/merlin/models/checkpoints/i3_resnet_clinical_longformer_best_clip_04-02-2024_23-21-36_epoch_99.pt"


def build_encoder(ckpt=None) -> I3ResNet:
    # 只用 resnet152 的结构；权重随后被 Merlin checkpoint 全部覆盖，不下载 ImageNet 权重
    enc = I3ResNet(torchvision.models.resnet152(weights=None), class_nb=1692,
                   conv_class=True, return_skips=True)
    sd = torch.load(Path(ckpt) if ckpt else DEFAULT_CKPT, map_location="cpu")
    p = "encode_image.i3_resnet."
    enc.load_state_dict({k[len(p):]: v for k, v in sd.items() if k.startswith(p)},
                        strict=True)
    for m in enc.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()
            for q in m.parameters():
                q.requires_grad_(False)
    return enc


class ResBlock(nn.Module):
    """SegResBlock：norm→act→conv ×2 + identity。"""

    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv3d(ch, ch, 3, padding=1, bias=False)
        self.n1 = nn.InstanceNorm3d(ch, affine=True)
        self.conv2 = nn.Conv3d(ch, ch, 3, padding=1, bias=False)
        self.n2 = nn.InstanceNorm3d(ch, affine=True)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        y = self.conv1(self.act(self.n1(x)))
        y = self.conv2(self.act(self.n2(y)))
        return y + x


def _deconv(cin, cout, xy_only):
    if xy_only:   # Z 已经满分辨率，只放大 X/Y（decoder 里张量是 (X,Y,Z)，卷积按 dim2,3,4 看）
        return nn.ConvTranspose3d(cin, cout, (3, 3, 1), (2, 2, 1),
                                  padding=(1, 1, 0), output_padding=(1, 1, 0), bias=False)
    return nn.ConvTranspose3d(cin, cout, 3, 2, padding=1, output_padding=1, bias=False)


class MerlinUNet(nn.Module):
    SKIP_CH = (1024, 512, 256, 64, 3)          # layer3, layer2, layer1, conv1, 原始输入

    def __init__(self, out_channels=5, widths=(256, 128, 64, 32, 16), ckpt=None,
                 adabn=False):
        super().__init__()
        assert len(widths) == 5
        self.adabn = adabn
        self.encoder = build_encoder(ckpt)
        self.proj = nn.Conv3d(2048, widths[0], 1)
        dec = tuple(widths[1:]) + (widths[-1],)      # 5 级输出通道：128/64/32/16/16
        # Z 只降了 4 次（160→10），前 4 级要 2×，只有最后一级只在 X/Y 上 2×
        self.up = nn.ModuleList([_deconv(widths[i], dec[i], xy_only=False)
                                 for i in range(4)]
                                + [_deconv(dec[-1], dec[-1], xy_only=True)])
        self.skip_proj = nn.ModuleList(nn.Conv3d(c, w, 1)
                                       for c, w in zip(self.SKIP_CH, dec))
        self.blocks = nn.ModuleList(ResBlock(w) for w in dec)
        self.head = nn.Conv3d(dec[-1], out_channels, 1)
        # 热图极稀疏：从"背景≈0"起步。否则 sigmoid(0)=0.5 的初始背景要靠 Adam 慢慢压，
        # 实测 5 个 epoch 都压不下来（E0 的 SegResNetDS 自己很快压下去了，这个解码器不会）。
        # 权重零初始化：5 个通道起点完全一致，由各自 GT 的梯度分化。随机权重会让某个通道
        # 一开始就偏向错误的 blob，sigmoid 饱和后整条通道死掉（实测 LCA 通道 max prob 0.04）。
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, -4.0)

    def train(self, mode=True):
        super().train(mode)
        # 编码器 BN 默认 eval（用 Merlin 的 running stats）；--adabn 保持 train，
        # 用 CTA 数据重估统计量（实测比冻结 BN 好很多）
        if mode and not self.adabn:
            for m in self.encoder.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
        return self

    def forward(self, x):
        # 下游数据是 LPS，Merlin 预处理是 RAS：翻转 X/Y 两轴对齐，输出再翻回来。
        x = torch.flip(x, dims=(2, 3))
        _, _, skips = self.encoder(x)           # skips: [输入, conv1, layer1..4]
        y = self.proj(skips[5])
        for i, skip in enumerate(skips[4::-1]):
            y = self.up[i](y) + self.skip_proj[i](skip)
            y = self.blocks[i](y)
        return torch.flip(self.head(y), dims=(2, 3))
