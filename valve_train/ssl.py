"""E3：3D 掩码重建自监督（SimMIM 式），给 Merlin 编码器做域内适配。

    # 正式：100 iter/epoch × 30 epoch
    python -m valve_train.ssl --out work/ssl/e3
    # 冒烟：2 iter，验证管线和显存
    python -m valve_train.ssl --out /tmp/ssl_smoke --epochs 1 --max-iters 2
    # 掩码逻辑自检（CPU，秒级）
    python -m valve_train.ssl --demo

输出 <out>/encoder.pth：键带 `encode_image.i3_resnet.` 前缀，直接给下游当初始化：
    python -m valve_train.train --grid merlin --fold 0 --arch merlin --adabn \
        --merlin-ckpt work/ssl/e3/encoder.pth

数据：cases.json 里 split ∈ {pretrain, pretrain_ssl} 的裁剪图（零标签），先跑
convert-pretrain --include-unlabeled → localize → crop --center totalseg。
只有被遮的体素算 L1；解码器只在这个阶段用，结束后丢弃。
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
from torch.utils.data import DataLoader, Dataset

from valve_prep.common import GRIDS, WORK
from valve_train.merlin_net import MerlinUNet


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
                         "localize → crop --grid merlin --center totalseg")
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
    ap.add_argument("--grid", default="merlin", choices=list(GRIDS))
    ap.add_argument("--merlin-ckpt", default=None, help="默认 release 权重")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--iters", type=int, default=100, help="每 epoch 的 iter 数")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4, help="解码器 lr")
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
    print(f"ssl pool={len(paths)} shape={shape} batch={args.batch}")

    # adabn=True：编码器 BN 用预训练池数据重估统计量（同 E2 的结论）
    model = MerlinUNet(1, ckpt=args.merlin_ckpt, adabn=True).to(device)
    # 重建头不是热图头：重新初始化（MerlinUNet 的零权重 + bias −4 是给稀疏热图用的，
    # 重建目标满图都是信号）
    torch.nn.init.kaiming_normal_(model.head.weight)
    torch.nn.init.zeros_(model.head.bias)

    enc = [p for p in model.encoder.parameters() if p.requires_grad]
    dec = [p for n, p in model.named_parameters()
           if not n.startswith("encoder.") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": dec, "lr": args.lr},
                             {"params": enc, "lr": args.enc_lr}],
                            weight_decay=args.wd)
    print(f"param groups: decoder {sum(p.numel() for p in dec)/1e6:.1f}M @ {args.lr}, "
          f"encoder {sum(p.numel() for p in enc)/1e6:.1f}M @ {args.enc_lr}")

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
            m = batch_mask(x.shape[0], shape, args.block, args.mask_ratio, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                pred = model(x * ~m)
            loss = ((pred.float() - x).abs() * m).sum() / m.sum()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
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
        sd = {f"encode_image.i3_resnet.{k}": v for k, v in model.encoder.state_dict().items()}
        torch.save(sd, f"{args.out}/encoder.pth")
        print(f"ep {ep:3d} loss {row['loss']:.4f} {dt:5.1f}s "
              f"({dt/max(n,1):.2f}s/iter, {n} iter) [{it_global}/{total}]", flush=True)
    f.close()
    print(f"done. encoder -> {args.out}/encoder.pth")


if __name__ == "__main__":
    main()
