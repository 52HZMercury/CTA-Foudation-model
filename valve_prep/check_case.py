"""单例抽查：热图峰值反解回世界坐标，并与 image_crop 叠图。

    python -m valve_prep.check_case --grid merlin --case huaxi_0000034090
    python -m valve_prep.check_case --grid merlin --case-random 5

产物: $VALVE_WORK/<grid>/_check/<case>.png
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

from . import common as C


def peak_world(hm_path: str, img: sitk.Image):
    hm = sitk.ReadImage(hm_path)
    arr = sitk.GetArrayFromImage(hm)                 # (z,y,x)
    if arr.max() <= 0:
        return None, None
    z, y, x = np.unravel_index(int(arr.argmax()), arr.shape)
    world = img.TransformContinuousIndexToPhysicalPoint([float(x), float(y), float(z)])
    return (x, y, z), np.asarray(world)


def draw(cid: str, grid: str, out_png: str) -> dict:
    d = f"{C.out_dir(grid)}/{cid}"
    img = sitk.ReadImage(f"{d}/image_crop.nii.gz")
    vol = sitk.GetArrayFromImage(img)                # (z,y,x)
    meta = C.read_json(f"{d}/landmarks_voxel.json")

    colors = ["#d1495b", "#00798c", "#edae49", "#66a182", "#8d4e85"]
    report = {"case": cid, "grid": grid, "points": {}}

    peaks = {}
    for name in C.LANDMARKS:
        p = f"{d}/targets/{name}_heatmap.nii.gz"
        if not os.path.exists(p):
            continue
        idx, world = peak_world(p, img)
        if idx is None:
            report["points"][name] = None
            continue
        gt = meta["voxel_xyz"].get(name)
        err = None
        if gt is not None:
            err = float(np.linalg.norm(np.array(idx, dtype=float) - np.array(gt)))
        report["points"][name] = {"peak_voxel": idx, "gt_voxel": gt,
                                  "err_voxel": err, "world": world.tolist()}
        peaks[name] = idx

    # 三个切面各自固定在一个参考层（5 个点的中位层），所有点投影到同一张图上；
    # 每个点的实际层号与参考层的差写进图例，避免误读成"全在同一层"。
    if not peaks:
        return report
    ref = np.median(np.array(list(peaks.values())), axis=0).round().astype(int)  # (x,y,z)
    x0, y0, z0 = ref

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(vol[z0], cmap="gray", vmin=-200, vmax=800)
    axes[1].imshow(vol[:, y0, :], cmap="gray", vmin=-200, vmax=800)
    axes[2].imshow(vol[:, :, x0], cmap="gray", vmin=-200, vmax=800)
    axes[0].set_title(f"axial  z={z0}", fontsize=10)
    axes[1].set_title(f"coronal  y={y0}", fontsize=10)
    axes[2].set_title(f"sagittal  x={x0}", fontsize=10)

    for name, (x, y, z) in peaks.items():
        c = colors[C.LANDMARKS.index(name)]
        axes[0].plot(x, y, "+", color=c, ms=13, mew=2,
                     label=f"{name}  dz={z - z0:+d}")
        axes[1].plot(x, z, "+", color=c, ms=13, mew=2)
        axes[2].plot(y, z, "+", color=c, ms=13, mew=2)

    axes[0].legend(fontsize=8, loc="upper right", framealpha=0.7)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{cid}  ({grid})   + = heatmap peak, projected onto one slice")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=90)
    plt.close(fig)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=list(C.GRIDS), required=True)
    ap.add_argument("--case", nargs="*", default=None)
    ap.add_argument("--case-random", type=int, default=0)
    args = ap.parse_args()

    cases = C.read_json(f"{C.WORK}/cases.json")
    ids = args.case or []
    if args.case_random:
        rng = np.random.RandomState(0)
        ids += list(rng.choice(sorted(cases), args.case_random, replace=False))
    if not ids:
        ids = ["huaxi_0000034090"]

    outdir = f"{C.out_dir(args.grid)}/_check"
    for cid in ids:
        png = f"{outdir}/{cid}.png"
        rep = draw(cid, args.grid, png)
        errs = [v["err_voxel"] for v in rep["points"].values() if v and v["err_voxel"] is not None]
        print(f"{cid}: peaks={len(rep['points'])} max_err={max(errs) if errs else None} voxel -> {png}")


if __name__ == "__main__":
    main()
