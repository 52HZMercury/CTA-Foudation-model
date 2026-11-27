"""E3：Merlin 编码器的域内自监督，两种目标：

    simsiam 双视图余弦相似度（默认，E3c；无负样本，projection + predictor + stop-grad）
    mask    3D 掩码重建（SimMIM 式，只在被遮体素算 L1；历史对照）

    # 默认 = E3c（SimSiam 100 ep / enc lr 1e-5）：
    bash run_ssl.sh
    # SimMIM（历史对照）：
    bash run_ssl.sh --objective mask --epochs 30
    # 冒烟（直接调模块时自定 --out）
    python -m valve_train.ssl --out /tmp/ssl_smoke --epochs 1 --max-iters 2

输出 logs/<TS>/encoder.pth：键带 `encode_image.i3_resnet.` 前缀，直接给下游当初始化：
    python -m valve_train.train --grid merlin --fold 0 --arch merlin --adabn \
        --merlin-ckpt logs/<TS>/encoder.pth

数据：cases.json 里 split ∈ {pretrain, pretrain_ssl} 的裁剪图（零标签），先跑
convert-pretrain --include-unlabeled → localize --fast → crop --center totalseg。
解码器/投影头只在这个阶段用，结束后丢弃。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from valve_prep.common import GRIDS, WORK
from valve_train.data import random_affine, warp
from valve_train.merlin_net import MerlinUNet, build_encoder


# ------------------------------------------------------------------ 数据
class SSLDataset(Dataset):
    """预训练池裁剪图 → (1, X, Y, Z) float32，归一化与下游完全一致。"""

    def __init__(self, paths, p_lo, p_hi):
        self.paths = paths
        self.p_lo, self.p_hi = float(p_lo), float(p_hi)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        a = np.asanyarray(nib.load(self.paths[i]).dataobj, dtype=np.float32)
        a = (np.clip(a, self.p_lo, self.p_hi) - self.p_lo) / (self.p_hi - self.p_lo)
        return torch.from_numpy(a)[None]


def pretrain_paths(grid="merlin"):
    cases = json.load(open(f"{WORK}/cases.json"))
    paths = [f"{WORK}/{grid}/{k}/image_crop.nii.gz" for k, c in cases.items()
             if c["split"] in ("pretrain", "pretrain_ssl")]
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        raise SystemExit("没有预训练裁剪图：先跑 convert-pretrain --include-unlabeled → "
                         "localize --fast → crop --grid merlin --center totalseg")
    return paths


# ------------------------------------------------------------------ 掩码
def block_mask(shape, block=16, ratio=0.6, device=None):
    """非重叠 block³ 掩码，True=被遮；按块抽 ratio，块内全遮。

    shape = (X, Y, Z)。返回 (X, Y, Z) bool；单样本的掩码，batch 里 stack 后加通道维。
    """
    nb = [(int(s) + block - 1) // block for s in shape]
    n = nb[0] * nb[1] * nb[2]
    idx = torch.randperm(n, device=device)[: int(round(ratio * n))]
    m = torch.zeros(n, dtype=torch.bool, device=device)
    m[idx] = True
    m = m.view(nb).repeat_interleave(block, 0).repeat_interleave(block, 1) \
        .repeat_interleave(block, 2)
    return m[: shape[0], : shape[1], : shape[2]]


def batch_mask(b, shape, block=16, ratio=0.6, device=None):
    """每样本独立掩码 → (B, 1, X, Y, Z) bool。"""
    m = torch.stack([block_mask(shape, block, ratio, device) for _ in range(b)])
    return m[:, None]


# ------------------------------------------------------------------ SimSiam
class SimSiamHeads(nn.Module):
    """projection（3 层 MLP）+ predictor（2 层 MLP），同 SimSiam 原论文。"""

    def __init__(self, dim=2048, hidden=512, pred_hidden=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden))
        self.pred = nn.Sequential(
            nn.Linear(hidden, pred_hidden), nn.BatchNorm1d(pred_hidden),
            nn.ReLU(inplace=True), nn.Linear(pred_hidden, hidden))

    def forward(self, f):
        z = self.proj(f)
        return z, self.pred(z)


def encode_feat(encoder, x):
    """(B,1,X,Y,Z) → layer4 全局平均池化 (B,2048)；X/Y 翻转同 MerlinUNet。"""
    _, _, skips = encoder(torch.flip(x, dims=(2, 3)))
    return skips[5].mean(dim=(2, 3, 4))


def demo():
    torch.manual_seed(0)
    m = block_mask((224, 224, 160), ratio=0.0)
    assert m.shape == (224, 224, 160) and not m.any()
    assert block_mask((224, 224, 160), ratio=1.0).all()
    m = block_mask((224, 224, 160), ratio=0.6)
    assert abs(m.float().mean().item() - 0.6) < 1e-6          # 1960 块，1176 块
    for i0 in (0, 16, 208):
        for j0 in (0, 16, 208):
            b = m[i0:i0 + 16, j0:j0 + 16, 0:16]
            assert b.all() or not b.any(), "块内有洞"
    assert batch_mask(4, (224, 224, 160)).shape == (4, 1, 224, 224, 160)
    print("ssl.demo ok")


# ------------------------------------------------------------------ 训练
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="输出目录（encoder.pth / log.csv）")
    ap.add_argument("--demo", action="store_true", help="只跑掩码自检")
    ap.add_argument("--objective", default="simsiam", choices=["mask", "simsiam"],
                    help="simsiam=双视图余弦相似度（默认，E3c）；mask=SimMIM 掩码重建")
    ap.add_argument("--grid", default="merlin", choices=list(GRIDS))
    ap.add_argument("--merlin-ckpt", default=None, help="默认 release 权重")
    ap.add_argument("--epochs", type=int, default=0,
                    help="0 = simsiam 100 / mask 30（E3c / 最初 E3 的预算）")
    ap.add_argument("--iters", type=int, default=100, help="每 epoch 的 iter 数")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4, help="解码器 / 投影头 lr")
    ap.add_argument("--enc-lr", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=100, help="warmup iter 数")
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--mask-ratio", type=float, default=0.6)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-iters", type=int, default=0,
                    help=">0 时每 epoch 只跑这么多 iter（冒烟用）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.demo:
        demo()
        return
    if not args.out:
        ap.error("--out 必填（或用 --demo）")
    if args.epochs <= 0:
        args.epochs = 100 if args.objective == "simsiam" else 30

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = GRIDS[args.grid]
    shape = spec["size"]                                     # (X, Y, Z)
    p_lo, p_hi = spec["hu"]

    paths = pretrain_paths(args.grid)
    loader = DataLoader(SSLDataset(paths, p_lo, p_hi), batch_size=args.batch,
                        shuffle=True, num_workers=args.workers, pin_memory=True,
                        drop_last=True, persistent_workers=args.workers > 0)
    print(f"ssl pool={len(paths)} shape={shape} batch={args.batch} "
          f"objective={args.objective}")

    if args.objective == "mask":
        # adabn=True：编码器 BN 用预训练池数据重估统计量（同 E2 的结论）
        model = MerlinUNet(1, ckpt=args.merlin_ckpt, adabn=True).to(device)
        # 重建头不是热图头：重新初始化（MerlinUNet 的零权重 + bias −4 是给稀疏热图用的，
        # 重建目标满图都是信号）
        torch.nn.init.kaiming_normal_(model.head.weight)
        torch.nn.init.zeros_(model.head.bias)
        enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
        head_params = [p for n, p in model.named_parameters()
                       if not n.startswith("encoder.") and p.requires_grad]
        clip_params = list(model.parameters())

        def save_sd():
            return {f"encode_image.i3_resnet.{k}": v
                    for k, v in model.encoder.state_dict().items()}
    else:
        model = build_encoder(args.merlin_ckpt).to(device)
        model.train()                     # AdaBN：BN 用预训练池重估
        heads = SimSiamHeads().to(device)
        enc_params = [p for p in model.parameters() if p.requires_grad]
        head_params = list(heads.parameters())
        clip_params = list(model.parameters()) + list(head_params)

        def save_sd():
            return {f"encode_image.i3_resnet.{k}": v
                    for k, v in model.state_dict().items()}

    opt = torch.optim.AdamW([{"params": head_params, "lr": args.lr},
                             {"params": enc_params, "lr": args.enc_lr}],
                            weight_decay=args.wd)
    print(f"param groups: head {sum(p.numel() for p in head_params)/1e6:.1f}M @ {args.lr}, "
          f"encoder {sum(p.numel() for p in enc_params)/1e6:.1f}M @ {args.enc_lr}")

    per_epoch = args.max_iters or args.iters
    total = args.epochs * per_epoch

    def lr_at(it):
        if it < args.warmup:
            return (it + 1) / args.warmup
        p = (it - args.warmup) / max(1, total - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    os.makedirs(args.out, exist_ok=True)
    log_p = f"{args.out}/log.csv"
    new = not os.path.exists(log_p)
    f = open(log_p, "a", newline="")
    w = csv.DictWriter(f, fieldnames=["epoch", "iter", "loss", "sec", "lr"])
    if new:
        w.writeheader()

    it_global = 0
    for ep in range(args.epochs):
        t0, acc, n = time.time(), 0.0, 0
        for x in loader:
            x = x.to(device, non_blocking=True)
            if args.objective == "mask":
                m = batch_mask(x.shape[0], shape, args.block, args.mask_ratio, device)
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                    pred = model(x * ~m)
                loss = ((pred.float() - x).abs() * m).sum() / m.sum()
            else:
                r1, t1 = random_affine(x.shape[0], device)
                r2, t2 = random_affine(x.shape[0], device)
                v1, v2 = warp(x, r1, t1, shape), warp(x, r2, t2, shape)
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                    f1 = encode_feat(model, v1)
                    f2 = encode_feat(model, v2)
                z1, p1 = heads(f1.float())
                z2, p2 = heads(f2.float())
                loss = -(F.cosine_similarity(p1, z2.detach(), dim=-1).mean()
                         + F.cosine_similarity(p2, z1.detach(), dim=-1).mean()) / 2

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(clip_params, 12.0)
            opt.step()
            sched.step()

            it_global += 1
            acc += loss.item(); n += 1
            if n >= per_epoch:
                break
        dt = time.time() - t0
        row = {"epoch": ep, "iter": it_global, "loss": acc / max(n, 1),
               "sec": dt, "lr": opt.param_groups[0]["lr"]}
        w.writerow(row); f.flush()
        # 每个 epoch 覆盖存一次编码器（~0.5 GB，写一次秒级）
        torch.save(save_sd(), f"{args.out}/encoder.pth")
        print(f"ep {ep:3d} loss {row['loss']:.4f} {dt:5.1f}s "
              f"({dt/max(n,1):.2f}s/iter, {n} iter) [{it_global}/{total}]", flush=True)
    f.close()
    print(f"done. encoder -> {args.out}/encoder.pth")


if __name__ == "__main__":
    main()
