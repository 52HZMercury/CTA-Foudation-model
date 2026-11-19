"""valve_train.data — 读 valve_prep 产物，GPU 上合成 GT 热图并做几何增强。

三条约定，后面所有代码都按它写：

1. 数组轴序是 **(X, Y, Z)**。nibabel 读 `image_crop.nii.gz` 得到的就是这个顺序，
   `landmarks_voxel.json` 的 `voxel_xyz` 也是这个顺序。张量形状 `(B, C, X, Y, Z)`。
2. 热图**不读盘**，由点坐标按 `prepare._heatmap` 的同一公式现算（σ、截断半径一致）。
3. 几何增强**不插值热图**：先 warp 图像，再把点坐标做同一变换，在输出栅格上重新合成热图。
   旋转 0.15 rad 时两种做法的差别在亚体素量级，而插值会把峰值糊掉。

`_theta()` 里有一处跨轴置换，是 grid_sample 的坑：它的 grid 末维顺序是 (x, y, z)
对应输入张量的 (W, H, D)，和数组轴序相反。`demo()` 用旋转验证这一点。
"""
from __future__ import annotations

import json
import math

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from valve_prep.common import LANDMARKS


# ------------------------------------------------------------------ 数据集
class LandmarkDataset(torch.utils.data.Dataset):
    """每个样本返回 (image, points, existence)。

    image: (1, X, Y, Z) float32，已归一化到 [0, 1]
    points: (5, 3) float32，体素坐标 (X, Y, Z)；缺失点为 0
    existence: (5,) float32，1 表示该点有标注
    """

    def __init__(self, entries, p_lo: float = 4.0, p_hi: float = 1000.0):
        self.entries = entries
        self.p_lo = float(p_lo)
        self.p_hi = float(p_hi)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        e = self.entries[i]
        img = np.asanyarray(nib.load(e["image"]).dataobj, dtype=np.float32)
        img = (np.clip(img, self.p_lo, self.p_hi) - self.p_lo) / (self.p_hi - self.p_lo)

        meta = json.load(open(e["landmarks"]))
        pts = np.zeros((len(LANDMARKS), 3), dtype=np.float32)
        exist = np.zeros(len(LANDMARKS), dtype=np.float32)
        for j, name in enumerate(LANDMARKS):
            v = meta["voxel_xyz"].get(name)
            if v is not None and meta["existence"][j]:
                pts[j] = v
                exist[j] = 1.0

        return torch.from_numpy(img)[None], torch.from_numpy(pts), torch.from_numpy(exist)


# ------------------------------------------------------------------ 热图
def gaussian_heatmaps(pts, exist, shape, spacing, sigma_mm: float = 3.0):
    """合成 (B, N, X, Y, Z) 热图，每通道峰值归一化到 1；缺失点通道全 0。

    pts: (B, N, 3) 体素坐标 (X, Y, Z)
    shape: (X, Y, Z)；spacing: 三个方向的 mm 间距
    """
    dev = pts.device
    gs = []
    for d, (S, sp) in enumerate(zip(shape, spacing)):
        idx = torch.arange(S, device=dev, dtype=torch.float32)
        dist = (idx.view(1, 1, S) - pts[..., d].unsqueeze(-1)) * sp        # (B, N, S) mm
        g = torch.exp(-dist ** 2 / (2.0 * sigma_mm ** 2))
        r = math.ceil(3 * sigma_mm / sp) * sp
        gs.append(torch.where(dist.abs() <= r, g, torch.zeros_like(g)))

    hm = gs[0][..., :, None, None] * gs[1][..., None, :, None] * gs[2][..., None, None, :]
    hm = hm * exist[:, :, None, None, None]
    return hm / hm.amax(dim=(2, 3, 4), keepdim=True).clamp_min(1e-8)


# ------------------------------------------------------------------ 几何增强
# grid_sample 的 grid 末维是 (x, y, z)，对应输入张量的 (W, H, D)，
# 与数组轴序 (X, Y, Z) -> (D, H, W) 相反，所以 theta 两侧各乘一次置换阵。
_P = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])


def random_affine(batch: int, device, max_rot=0.15, max_trans=5.0, max_scale=0.10):
    """返回 (R, t)：(B,3,3) 旋转×缩放、(B,3) 平移，都在体素索引空间。"""
    a = (torch.rand(batch, 3, device=device) * 2 - 1) * max_rot
    s = 1.0 + (torch.rand(batch, 3, device=device) * 2 - 1) * max_scale
    t = (torch.rand(batch, 3, device=device) * 2 - 1) * max_trans

    ca, sa = torch.cos(a), torch.sin(a)
    c0, c1, c2 = ca[:, 0:1], ca[:, 1:2], ca[:, 2:3]
    s0, s1, s2 = sa[:, 0:1], sa[:, 1:2], sa[:, 2:3]
    one, zero = torch.ones_like(c0), torch.zeros_like(c0)
    rx = torch.stack([one, zero, zero, zero, c0, -s0, zero, s0, c0], -1)
    ry = torch.stack([c1, zero, s1, zero, one, zero, -s1, zero, c1], -1)
    rz = torch.stack([c2, -s2, zero, s2, c2, zero, zero, zero, one], -1)
    R = rx.view(-1, 3, 3) @ ry.view(-1, 3, 3) @ rz.view(-1, 3, 3) @ torch.diag_embed(s)
    return R, t


def _theta(shape, R, c, t):
    """把"索引空间"的仿射 x_in = R (x_out - c) + c + t 翻成归一化空间的 theta。"""
    dev = R.device
    S = torch.diag(torch.tensor([2.0 / (s - 1) for s in shape], device=dev))
    s0 = torch.full((3,), -1.0, device=dev)
    M = S @ R @ torch.linalg.inv(S)                                   # (B,3,3)
    b = (-M @ s0 - torch.einsum("bij,bj->bi", S @ R, c) + c @ S.T + t @ S.T + s0)
    P = _P.to(dev)
    return torch.cat([P @ M @ P, (b @ P.T)[:, :, None]], dim=2)       # (B,3,4)


def warp(img, R, t, shape, center=None):
    """img: (B, 1, X, Y, Z)；按 (R, t) 做一次三线性重采样。"""
    B = img.shape[0]
    c = center if center is not None else torch.tensor(
        [(s - 1) / 2.0 for s in shape], device=img.device).expand(B, 3)
    c = c.expand(B, 3)
    grid = F.affine_grid(_theta(shape, R, c, t), (B, 1, *shape), align_corners=True)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


def warp_points(pts, R, t, shape, center=None):
    """pts: (B, N, 3) 输入栅格坐标 -> 输出栅格坐标。"""
    B = pts.shape[0]
    c = center if center is not None else torch.tensor(
        [(s - 1) / 2.0 for s in shape], device=pts.device).expand(B, 3)
    c = c.expand(B, 3)
    d = pts - c[:, None, :] - t[:, None, :]
    return torch.einsum("bij,bnj->bni", torch.linalg.inv(R), d) + c[:, None, :]


# ------------------------------------------------------------------ 峰值
def subvoxel_peak(hm, radius: int = 2):
    """hm: (B, N, X, Y, Z) -> (B, N, 3) 亚体素质心坐标 (X, Y, Z)。"""
    B, N = hm.shape[:2]
    flat = hm.reshape(B * N, -1)
    k = flat.argmax(dim=1)
    shape = hm.shape[2:]
    coords = torch.stack(torch.unravel_index(k, shape), dim=1).float()     # (B*N, 3)

    offsets = torch.tensor(
        [(i, j, l) for i in range(-radius, radius + 1)
         for j in range(-radius, radius + 1)
         for l in range(-radius, radius + 1)], device=hm.device, dtype=torch.float32)

    num = torch.zeros_like(coords)
    den = torch.zeros(B * N, device=hm.device)
    for off in offsets:
        p = coords + off
        inb = torch.stack([((p[:, d] >= 0) & (p[:, d] < shape[d])) for d in range(3)], 1).all(1)
        p = p.clamp(min=torch.zeros(3, device=hm.device),
                    max=torch.tensor([s - 1 for s in shape], device=hm.device, dtype=torch.float32))
        b = torch.arange(B * N, device=hm.device)
        w = flat[b, p[:, 0].long() * shape[1] * shape[2]
                 + p[:, 1].long() * shape[2] + p[:, 2].long()] ** 4
        w = torch.where(inb, w, torch.zeros_like(w))
        num += w[:, None] * p
        den += w
    return (num / den.clamp_min(1e-12)[:, None]).view(B, N, 3)


# ------------------------------------------------------------------ 自检
def demo():
    """一次几何自检：造一个各向尺度的亮斑，做随机仿射，看图像亮斑和点是否一致。"""
    torch.manual_seed(0)
    shape = (40, 48, 24)
    p = torch.tensor([9.0, 31.0, 5.0])
    idx = torch.arange(40)[:, None, None], torch.arange(48)[None, :, None], torch.arange(24)[None, None, :]
    vol = torch.exp(-((idx[0] - p[0]) ** 2 + (idx[1] - p[1]) ** 2 + (idx[2] - p[2]) ** 2) / 8.0)[None, None]

    for R, t in [(torch.eye(3)[None], torch.tensor([[3.0, -2.0, 1.0]])),
                 (random_affine(1, "cpu", 0.15, 0.0, 0.0)[0], torch.zeros(1, 3)),
                 (random_affine(1, "cpu", 0.15, 5.0, 0.10)[0], random_affine(1, "cpu", 0.0, 5.0, 0.0)[1])]:
        out = warp(vol, R, t, shape)
        q = warp_points(p.view(1, 1, 3), R, t, shape)[0, 0]
        peak = torch.stack(torch.unravel_index(out[0, 0].argmax(), shape)).float()
        err = (peak - q).norm().item()
        assert err < 1.0, f"图像峰值 {peak.tolist()} 与变换后点 {q.tolist()} 不符 (err={err:.2f})"
        print(f"ok  peak={peak.tolist()}  point={[round(v, 2) for v in q.tolist()]}")

    hm = gaussian_heatmaps(p.view(1, 1, 3), torch.ones(1, 1), shape, (0.5, 0.5, 0.5))
    got = subvoxel_peak(hm)[0, 0]
    assert (got - p).norm() < 0.05, f"热图峰 {got.tolist()} != {p.tolist()}"
    print(f"ok  heatmap peak={[round(v, 3) for v in got.tolist()]}")


if __name__ == "__main__":
    demo()
