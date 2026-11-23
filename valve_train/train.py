"""标志点热图回归训练。

    # E0：SegResNetDS 从零
    python -m valve_train.train --grid merlin --fold 0 --out work/runs/e0_fold0
    # E2：Merlin 编码器 + U-Net 解码器（编码器 lr 1e-5，解码器 2e-4）
    python -m valve_train.train --grid merlin --fold 0 --arch merlin --out work/runs/e2_fold0
    python -m valve_train.train --grid merlin --fold 0 --epochs 1 --max-iters 3   # 冒烟

只做标志点任务。plane / calcification 等定稿后再加。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from valve_prep.common import GRIDS, LANDMARKS, WORK
from valve_train.data import (LandmarkDataset, gaussian_heatmaps, random_affine,
                              subvoxel_peak, warp, warp_points)
from valve_train.losses import LandmarkHeatmapLoss


# ------------------------------------------------------------------ 模型
def build_model(arch: str, out_channels: int, init_filters: int, blocks_down,
                merlin_ckpt=None, adabn=False):
    if arch == "segresnet":
        from monai.networks.nets import SegResNetDS
        return SegResNetDS(spatial_dims=3, in_channels=1, out_channels=out_channels,
                           init_filters=init_filters, blocks_down=list(blocks_down),
                           dsdepth=1, norm="INSTANCE")
    if arch == "merlin":
        from valve_train.merlin_net import MerlinUNet
        return MerlinUNet(out_channels, ckpt=merlin_ckpt, adabn=adabn)
    raise SystemExit(f"arch={arch} 还没实现（可选 segresnet / merlin）")


# ------------------------------------------------------------------ 评估
@torch.no_grad()
def evaluate(model, loader, spacing, sigma, device, shape):
    """返回 {"mre": mm, "sdr2/sdr5/sdr10": 比例, "per_lm": {...}}。"""
    model.eval()
    sp = torch.tensor(spacing, device=device)
    errs = {n: [] for n in LANDMARKS}
    for img, pts, exist in loader:
        img, pts, exist = img.to(device), pts.to(device), exist.to(device)
        pred = subvoxel_peak(torch.sigmoid(model(img)))
        d = ((pred - pts) * sp).norm(dim=-1)                        # (B, N) mm
        for j, n in enumerate(LANDMARKS):
            sel = exist[:, j] > 0
            if sel.any():
                errs[n].append(d[sel, j])
    allv = torch.cat([e for v in errs.values() for e in v]) if any(errs.values()) \
        else torch.zeros(1, device=device)
    out = {"mre": allv.mean().item(),
           "sdr2": (allv < 2).float().mean().item(),
           "sdr5": (allv < 5).float().mean().item(),
           "sdr10": (allv < 10).float().mean().item(),
           "per_lm": {n: (torch.cat(v).mean().item() if v else None) for n, v in errs.items()}}
    model.train()
    return out


# ------------------------------------------------------------------ 训练
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="merlin", choices=list(GRIDS))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--split", default=None, help="外部集名字，如 carry_external；给了就不折")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default="segresnet", choices=["segresnet", "merlin"])
    ap.add_argument("--adabn", action="store_true",
                    help="merlin：编码器 BN 保持 train 模式，用 CTA 数据重估 running stats")
    ap.add_argument("--merlin-ckpt", default=None, help="默认 Merlin/merlin/models/checkpoints/ 下的 release 权重")
    ap.add_argument("--enc-lr", type=float, default=1e-5, help="merlin 编码器 lr（解码器用 --lr）")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val-interval", type=int, default=5)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--w-mse", type=float, default=0.5)
    ap.add_argument("--w-dice", type=float, default=1.0)
    ap.add_argument("--w-wing", type=float, default=2.0)
    ap.add_argument("--init-filters", type=int, default=32)
    ap.add_argument("--blocks-down", default="1,2,2,4,4")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-iters", type=int, default=0, help=">0 时每个 epoch 只跑这么多 batch（冒烟用）")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prep = f"{WORK}/{args.grid}"
    spacing = GRIDS[args.grid]["spacing"]
    shape = GRIDS[args.grid]["size"]                       # (X, Y, Z)，与 nibabel 轴序一致
    p_lo, p_hi = GRIDS[args.grid]["hu"]          # 固定 [−1000, 1000]，同 Merlin

    dl = json.load(open(f"{prep}/datalist.json"))
    if args.split:
        train_e, val_e = [], dl["external"]
    else:
        f = dl["folds"][f"fold_{args.fold}"]
        train_e, val_e = f["train"], f["val"]

    ds = lambda e: LandmarkDataset(e, p_lo, p_hi)
    train_loader = DataLoader(ds(train_e), batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True,
                              persistent_workers=args.workers > 0)
    val_loader = DataLoader(ds(val_e), batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True,
                            persistent_workers=args.workers > 0)

    model = build_model(args.arch, len(LANDMARKS), args.init_filters,
                        [int(v) for v in args.blocks_down.split(",")],
                        args.merlin_ckpt, args.adabn).to(device)
    criterion = LandmarkHeatmapLoss(args.w_mse, args.w_dice, args.w_wing).to(device)
    if args.arch == "merlin":
        enc = [p for p in model.encoder.parameters() if p.requires_grad]
        dec = [p for n, p in model.named_parameters()
               if not n.startswith("encoder.") and p.requires_grad]
        opt = torch.optim.AdamW([{"params": dec, "lr": args.lr},
                                 {"params": enc, "lr": args.enc_lr}],
                                weight_decay=args.wd)
        print(f"param groups: decoder {sum(p.numel() for p in dec)/1e6:.1f}M @ {args.lr}, "
              f"encoder {sum(p.numel() for p in enc)/1e6:.1f}M @ {args.enc_lr} "
              f"adabn={args.adabn}")
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    def lr_at(ep):
        if ep < args.warmup:
            return (ep + 1) / args.warmup
        p = (ep - args.warmup) / max(1, args.epochs - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    os.makedirs(args.out, exist_ok=True)
    if args.split:                                    # 只评估外部集：--split external
        print(json.dumps(evaluate(model, val_loader, spacing, args.sigma, device, shape), indent=1))
        return

    writer = SummaryWriter(args.out)
    fields = ["epoch", "lr", "loss", "sec", "mre", "sdr2", "sdr5", "sdr10"] + \
             [f"mre_{n}" for n in LANDMARKS]
    log_path = f"{args.out}/log.csv"
    start_ep, best, best_ep = 0, float("inf"), -1
    if args.resume and os.path.exists(f"{args.out}/last.pth"):
        ck = torch.load(f"{args.out}/last.pth", map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_ep, best, best_ep = ck["epoch"] + 1, ck["best"], ck["best_epoch"]
        print(f"resume from epoch {start_ep}, best={best:.3f}")

    print(f"train={len(train_e)} val={len(val_e)} shape={shape} spacing={spacing} "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    for ep in range(start_ep, args.epochs):
        model.train()
        t0, n, acc = time.time(), 0, 0.0
        for img, pts, exist in train_loader:
            img, pts, exist = img.to(device, non_blocking=True), pts.to(device), exist.to(device)
            R, t = random_affine(img.shape[0], device)
            img_a = warp(img, R, t, shape)
            pts_a = warp_points(pts, R, t, shape)
            hm = gaussian_heatmaps(pts_a, exist, shape, spacing, args.sigma)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(img_a)
            loss = criterion(torch.sigmoid(logits.float()), hm, exist)["total"]

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
            opt.step()

            acc += loss.item(); n += 1
            if args.max_iters and n >= args.max_iters:
                break
        sched.step()
        dt = time.time() - t0

        row = {"epoch": ep, "lr": opt.param_groups[0]["lr"], "loss": acc / max(n, 1), "sec": dt}
        msg = f"ep {ep:3d} loss {row['loss']:.4f} {dt:5.1f}s"

        if (ep + 1) % args.val_interval == 0 or ep == args.epochs - 1:
            m = evaluate(model, val_loader, spacing, args.sigma, device, shape)
            row.update({k: m[k] for k in ("mre", "sdr2", "sdr5", "sdr10")})
            row.update({f"mre_{k}": v for k, v in m["per_lm"].items() if v is not None})
            msg += (f" | val MRE {m['mre']:.2f}mm  SDR@2/5/10 "
                    f"{m['sdr2']:.2f}/{m['sdr5']:.2f}/{m['sdr10']:.2f}")
            for k, v in m.items():
                if isinstance(v, float):
                    writer.add_scalar(f"val/{k}", v, ep)
            improved = m["mre"] < best - 1e-4
            if improved:
                best, best_ep = m["mre"], ep
                torch.save({"model": model.state_dict(), "epoch": ep, "val": m, "args": vars(args)},
                           f"{args.out}/best.pth")
            if ep - best_ep >= args.patience:
                print(f"early stop at {ep} (best {best:.3f} @ {best_ep})")
                writer.close()
                break

        writer.add_scalar("train/loss", row["loss"], ep)
        writer.add_scalar("train/lr", row["lr"], ep)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": ep, "best": best, "best_epoch": best_ep}, f"{args.out}/last.pth")

        new = not os.path.exists(log_path)
        with open(log_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if new:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in fields})
        print(msg, flush=True)

    writer.close()
    print(f"done. best MRE {best:.3f} mm @ ep {best_ep}")


if __name__ == "__main__":
    main()
