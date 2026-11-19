"""valve_prep 主流程。

    python -m valve_prep.prepare index
    python -m valve_prep.prepare convert-pretrain           # 预训练 5 点标注 -> 下游 gt 格式
    python -m valve_prep.prepare localize --jobs 4          # TotalSegmentator 主动脉窦质心
    python -m valve_prep.prepare crop   --grid merlin --center totalseg --jobs 4
    python -m valve_prep.prepare targets --grid merlin
    python -m valve_prep.prepare qa      --grid merlin
    python -m valve_prep.prepare split   --grid merlin

所有产物落在 $VALVE_WORK（默认 ~/program/CTA-Foudation-model/work/prep）。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from multiprocessing import Pool

import numpy as np
import SimpleITK as sitk

from . import common as C

SIGMA_MM = 3.0          # 热图 σ，与旧基线一致
RING_SIGMA_MM = 1.0     # 瓣环环的软边宽度
QA_MARGIN_VOX = 10      # landmark 距边界的最小体素余量


# ================================================================ index
def cmd_index(_args):
    out = {}
    for split in ("huaxi", "carry"):
        for d in C.case_ids(split):
            cdir = C.case_dir(split, d)
            img_path = f"{cdir}/image.nii.gz"
            gt = f"{cdir}/gt"
            im = sitk.ReadImage(img_path)
            out[f"{split}_{C.parse_case_id(d)}"] = dict(
                split=split,
                dirname=d,
                image=img_path,
                shape=[int(v) for v in im.GetSize()],          # (x,y,z)
                spacing=[float(v) for v in im.GetSpacing()],
                direction=[float(v) for v in im.GetDirection()],
                origin=[float(v) for v in im.GetOrigin()],
                existence=C.existence_mask(gt),
                has_annulus=os.path.exists(f"{gt}/annulus.mrk.json"),
                has_root=os.path.exists(f"{gt}/root.nrrd"),
                has_calc=os.path.exists(f"{gt}/calcification.nrrd"),
            )
    C.write_json(f"{C.WORK}/cases.json", out)
    n = {"huaxi": 0, "carry": 0}
    for v in out.values():
        n[v["split"]] += 1
    print(f"indexed {len(out)} cases {n}")


# ================================================================ convert-pretrain
# 预训练集只有 raw_fct_marker：三个窦底 Marker + LCA/RCA 开口 + 瓣环曲线，
# 转成下游 gt 格式。转换前已验证（2026-11-19）：
#   1) Marker_1..3 到瓣环平面距离 p90 0.2 mm，与下游 GT 的 L/R/N 同构（都在瓣环平面上）；
#   2) 窦周径平面在 Marker 平面上方 7.5–11.1 mm，说明 Marker 在窦底而非交界；
#   3) 规则「L=角度上离 LCA 最近，R=离 RCA 最近」在 314 例下游 GT 上命中 99.4%/98.7%。
# 分配取 6 种排列里角度距离和最小者（贪心在 2 例上会让 L/R 互抢）。

def _write_mrk(path, name, pts, curve=False):
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    C.write_json(path, {"markups": [dict(
        name=name, type="Curve" if curve else "Fiducial", coordinateSystem="LPS",
        controlPoints=[dict(id=str(i + 1), label=name,
                            position=[float(v) for v in p], locked=True, visibility=True)
                       for i, p in enumerate(pts)])]})


def _plane_fit(pts):
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c)
    n = vt[-1] / np.linalg.norm(vt[-1])
    e1 = vt[0] / np.linalg.norm(vt[0])
    return c, n, e1, np.cross(n, e1)


def _ang(p, c, e1, e2):
    d = p - c
    return float(np.arctan2(d @ e2, d @ e1))


def _adiff(a, b):
    d = abs(a - b) % (2 * np.pi)
    return float(min(d, 2 * np.pi - d))


def cmd_convert_pretrain(args):
    """把预训练标注转成下游 gt 格式，按通道登记。

    覆盖（2219 例 fct_pid）：
      - 803 例 3 Marker + 双开口 -> 5 通道全有；
      - 641 例只有 LCA/RCA 开口 -> 只登记这 2 个直接标注通道（existence 0 的通道
        在损失里直接跳过，见 losses.py 三态 mask，不会教模型"此处无窦点"）；
      - 775 例无开口标注 -> 不加标注。
    `--include-unlabeled` 时，上面 775 例 + 571 例 dicom 登记为 `split="pretrain_ssl"`
    （existence 全 0、无 gt），只跑 localize/crop 供 E3 自监督，不进任何 datalist。
    只有 2 个 Marker 的 210 例不做推导：p90 分配角 56°/128°，缺的那个常是 N，
    强行分配会灌入错误 L/R 标签。
    """
    import itertools
    include_unlabeled = getattr(args, "include_unlabeled", False)
    base = f"{C.RELEASE}/pretrain/huaxi"
    cases = C.read_json(f"{C.WORK}/cases.json")
    down_pids = {C.parse_case_id(c["dirname"]) for c in cases.values()
                 if not c["split"].startswith("pretrain")}
    dirs = sorted(d for d in os.listdir(base) if d.startswith("fct_pid_"))
    rows, added, skipped, n_partial, added_ssl = [], 0, 0, 0, 0

    def _register_ssl(d):
        cases[f"pretrain_{d}"] = dict(
            split="pretrain_ssl", dirname=d, image=f"{base}/{d}/image.nii.gz",
            existence=[0] * len(C.LANDMARKS), has_annulus=False,
            has_root=False, has_calc=False)

    for d in dirs:
        b = f"{base}/{d}/raw_fct_marker"
        mk_p = [f"{b}/landmarks/Marker_{k}.mrk.json" for k in (1, 2, 3)]
        lca_p = f"{b}/landmarks/LCA_ostium_1.mrk.json"
        rca_p = f"{b}/landmarks/DistancePoint1_RCA_height_1.mrk.json"
        ann_p = f"{b}/curves/Aortic_valve_annulus.mrk.json"
        has_lca = os.path.exists(lca_p)
        has_rca = os.path.exists(rca_p)
        if not (has_lca or has_rca):
            skipped += 1
            if include_unlabeled:
                _register_ssl(d)
                added_ssl += 1
            continue
        gt = f"{C.WORK}/pretrain_gt/{d}/gt"
        ex = [0] * len(C.LANDMARKS)
        ann = C.read_markup(ann_p)

        # 三窦点：必须三个 Marker 齐全 + 双开口（用于定向），平面优先用瓣环曲线
        if all(os.path.exists(p) for p in mk_p) and has_lca and has_rca:
            mk = np.vstack([C.read_markup(p).mean(axis=0) for p in mk_p])
            lca = C.read_markup(lca_p).mean(axis=0)
            rca = C.read_markup(rca_p).mean(axis=0)
            plane_pts = ann if ann is not None else mk
            c, _, e1, e2 = _plane_fit(plane_pts)
            am = [_ang(p, c, e1, e2) for p in mk]
            al, ar = _ang(lca, c, e1, e2), _ang(rca, c, e1, e2)
            costs = sorted((_adiff(am[li], al) + _adiff(am[ri], ar), li, ri)
                           for li, ri in itertools.permutations(range(3), 2))
            cost, li, ri = costs[0]
            ni = ({0, 1, 2} - {li, ri}).pop()
            for name, idx in (("L", li), ("R", ri), ("N", ni)):
                _write_mrk(f"{gt}/{name}.mrk.json", name, mk[idx])
                ex[C.LANDMARKS.index(name)] = 1
            rows.append(dict(case=d, l_deg=float(np.degrees(_adiff(am[li], al))),
                             r_deg=float(np.degrees(_adiff(am[ri], ar))),
                             margin_deg=float(np.degrees(costs[1][0] - cost))))
        elif has_lca and has_rca:
            n_partial += 1

        if has_lca:
            _write_mrk(f"{gt}/LCA.mrk.json", "LCA", C.read_markup(lca_p).mean(axis=0))
            ex[C.LANDMARKS.index("LCA")] = 1
        if has_rca:
            _write_mrk(f"{gt}/RCA.mrk.json", "RCA", C.read_markup(rca_p).mean(axis=0))
            ex[C.LANDMARKS.index("RCA")] = 1
        if ann is not None:
            _write_mrk(f"{gt}/annulus.mrk.json", "annulus", ann, curve=True)
        cases[f"pretrain_{d}"] = dict(
            split="pretrain", dirname=d, image=f"{base}/{d}/image.nii.gz",
            existence=ex, has_annulus=ann is not None,
            has_root=False, has_calc=False, gt_dir=gt)
        added += 1

    if include_unlabeled:
        for d in sorted(x for x in os.listdir(base) if x.startswith("dicom_pid_")):
            if os.path.exists(f"{base}/{d}/image.nii.gz"):
                _register_ssl(d)
                added_ssl += 1

    overlap = {C.parse_case_id(r["case"]) for r in rows} & down_pids
    if overlap:
        raise SystemExit(f"预训练与下游 PID 重叠: {sorted(overlap)[:5]}")
    C.write_json(f"{C.WORK}/cases.json", cases)
    C.write_json(f"{C.WORK}/pretrain_convert.json", rows)
    ld = np.array([r["l_deg"] for r in rows])
    rd = np.array([r["r_deg"] for r in rows])
    mg = np.array([r["margin_deg"] for r in rows])
    print(f"convert-pretrain added={added} (5ch {len(rows)}, 2ch {n_partial}) "
          f"skipped={skipped} ssl_only={added_ssl} overlap=0 | "
          f"L deg p50 {np.median(ld):.0f} p90 {np.percentile(ld, 90):.0f} | "
          f"R deg p50 {np.median(rd):.0f} p90 {np.percentile(rd, 90):.0f} | "
          f"ambiguous(<5deg) {(mg < 5).sum()}")


# ================================================================ localize
# TotalSegmentator 的 heartchambers_highres / aortic_sinuses 是收费许可模型，
# 免费可用的只有 total；取其中的 whole-heart 类。
HEART_CLASS = "heart"
SEG_MIN_MM3 = 50_000.0  # 心脏分割小于 50 ml 就当失败


def _ts_bin() -> str:
    """TotalSegmentator 只装了 console script，没有 __main__。"""
    p = shutil.which("TotalSegmentator") or \
        os.path.join(os.path.dirname(sys.executable), "TotalSegmentator")
    if not os.path.exists(p):
        raise RuntimeError("TotalSegmentator 未安装，先 pip install")
    return p


def _process_localize(job):
    """在原图（release 原始栅格）上跑 TotalSegmentator，取主动脉窦质心。

    输出 <WORK>/seg/<case>/center.json（键 "center"，LPS 世界 mm），
    只喂原始图像，不接触任何 GT。
    """
    case, task, device, fast = job
    odir = f"{C.WORK}/seg/{case['id']}"
    center_p = f"{odir}/center.json"
    if os.path.exists(center_p):
        return (case["id"], "skip", "")

    os.makedirs(odir, exist_ok=True)
    mask_p = f"{odir}/{task}.nii.gz"
    if not os.path.exists(mask_p):
        # --ml 时 -o 是输出文件路径
        cmd = [_ts_bin(), "-i", case["image"],
               "-o", mask_p, "-ta", task, "--ml", "-d", device,
               "--nr_thr_saving", "1", "--quiet"] + (["--fast"] if fast else [])
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if not os.path.exists(mask_p):
            tail = ((r.stderr or "") + (r.stdout or "")).strip().splitlines()[-2:]
            return (case["id"], "fail", " | ".join(tail) or f"exit {r.returncode}")

    from totalsegmentator.map_to_binary import class_map   # 类号随版本变，按名字查
    heart_id = {v: k for k, v in class_map[task].items()}[HEART_CLASS]
    m = sitk.ReadImage(mask_p)
    seg = sitk.GetArrayFromImage(m)
    idx = np.argwhere(seg == heart_id)
    if len(idx) == 0:
        return (case["id"], "fail", "empty heart mask")
    vol = float(len(idx) * np.prod(m.GetSpacing()))
    if vol < SEG_MIN_MM3:
        return (case["id"], "fail", f"heart {vol/1000:.0f} ml < {SEG_MIN_MM3/1000:.0f} ml")
    cidx = idx.mean(axis=0)[::-1]                     # (z,y,x) -> (x,y,z)
    heart = np.asarray(m.TransformContinuousIndexToPhysicalPoint([float(v) for v in cidx]))
    center = heart + C.HEART_TO_ROOT_MM
    C.write_json(center_p, dict(center=center.tolist(), heart_center=heart.tolist(),
                                volume_mm3=vol, source=f"totalseg:{task}",
                                image=case["image"]))
    return (case["id"], "ok", "")


def cmd_localize(args):
    cases = C.read_json(f"{C.WORK}/cases.json")
    jobs = [(dict(c, id=k), args.task, args.device, args.fast)
            for k, c in cases.items()]
    if args.limit:
        jobs = jobs[:args.limit]
    stats = {"ok": 0, "skip": 0, "fail": 0}
    fails = []
    with Pool(args.jobs) as pool:
        for cid, status, msg in pool.imap_unordered(_process_localize, jobs):
            stats[status] += 1
            if status == "fail":
                fails.append({"case": cid, "reason": msg})
                print(f"  FAIL {cid}: {msg}", flush=True)
            elif status == "ok":
                print(f"  ok {cid}", flush=True)
    C.write_json(f"{C.WORK}/seg/localize_failures.json", fails)
    print(f"localize task={args.task} {stats}")


# ================================================================ crop
def _crop_center(case: dict, source: str) -> np.ndarray | None:
    gt = C.case_gt(case)
    if source == "totalseg":
        p = f"{C.WORK}/seg/{case['id']}/center.json"
        if not os.path.exists(p):
            return None
        c = C.read_json(p)
        # 用原始心脏质心 + 群体偏移，改常数即可重算，不用重跑分割
        if "heart_center" in c:
            return np.asarray(c["heart_center"], dtype=float) + C.HEART_TO_ROOT_MM
        return np.asarray(c["center"], dtype=float)

    # source == "gt"：优先瓣环质心，退而用可用标志点均值
    pts = C.read_markup(f"{gt}/annulus.mrk.json")
    if pts is None:
        avail = [C.read_markup(f"{gt}/{n}.mrk.json") for n in C.LANDMARKS]
        avail = [p for p in avail if p is not None]
        if not avail:
            return None
        pts = np.vstack(avail)
    return pts.mean(axis=0)


def _process_case(job):
    case, grid, center_source = job
    spec = C.GRIDS[grid]
    odir = f"{C.out_dir(grid)}/{case['id']}"
    meta_p = f"{odir}/crop_meta.json"
    if os.path.exists(f"{odir}/image_crop.nii.gz") and os.path.exists(meta_p) and \
            C.read_json(meta_p).get("center_source") == center_source:
        return (case["id"], "skip", "")

    center = _crop_center(case, center_source)
    if center is None:
        return (case["id"], "fail", f"no center (source={center_source})")
    center = center + C.CENTER_OFFSET_MM

    img = C.read_image(case["image"])
    img_r = C.resample_to_spacing(img, spec["spacing"], sitk.sitkLinear, -1000.0)
    start, lo, hi = C.crop_params(img_r, center, spec["size"])
    img_c = C.apply_crop(img_r, start, lo, hi, spec["size"], fill=-1000.0)

    # 强度裁剪（只裁不归一化，归一化在训练时按 norm.json 做）
    geo = img_c
    arr = sitk.GetArrayFromImage(img_c).astype(np.float32)
    arr = np.clip(arr, spec["hu"][0], spec["hu"][1])
    img_c = sitk.GetImageFromArray(arr)
    img_c.CopyInformation(geo)

    os.makedirs(odir, exist_ok=True)
    sitk.WriteImage(img_c, f"{odir}/image_crop.nii.gz")

    gt = C.case_gt(case)
    for key in ("root", "calcification"):
        p = f"{gt}/{key}.nrrd"
        if os.path.exists(p):
            m = C.resample_like(C.read_image(p), img_r, sitk.sitkNearestNeighbor, 0.0)
            m_c = C.apply_crop(m, start, lo, hi, spec["size"], fill=0.0)
            sitk.WriteImage(sitk.Cast(m_c, sitk.sitkUInt8), f"{odir}/{key}_crop.nii.gz")

    C.write_json(meta_p, dict(
        grid=grid, center_source=center_source, center_world=center.tolist(),
        spacing=spec["spacing"], size=spec["size"], start_index=start.tolist(),
        pad_lo=lo.tolist(),
    ))
    return (case["id"], "ok", "")




def cmd_crop(args):
    cases = C.read_json(f"{C.WORK}/cases.json")
    jobs = [(dict(c, id=k), args.grid, args.center) for k, c in cases.items()]
    os.makedirs(C.out_dir(args.grid), exist_ok=True)
    stats = {"ok": 0, "skip": 0, "fail": 0}
    fails = []
    with Pool(args.jobs) as pool:
        for cid, status, msg in pool.imap_unordered(_process_case, jobs):
            stats[status] += 1
            if status == "fail":
                fails.append({"case": cid, "reason": msg})
                print(f"  FAIL {cid}: {msg}")
    C.write_json(f"{C.out_dir(args.grid)}/crop_failures.json", fails)
    print(f"crop grid={args.grid} {stats}")


# ================================================================ targets
def _heatmap(shape, spacing_zyx, idx_zyx, sigma_mm):
    spacing_zyx = np.asarray(spacing_zyx, dtype=float)
    r = np.ceil(3 * sigma_mm / spacing_zyx).astype(int)          # 每个轴各自算半径
    lo = np.maximum(0, np.floor(idx_zyx).astype(int) - r)
    hi = np.minimum(np.array(shape), np.ceil(idx_zyx).astype(int) + r + 1)
    if np.any(hi <= lo):
        return np.zeros(shape, dtype=np.float32)
    grid = np.meshgrid(*[np.arange(lo[i], hi[i]) for i in range(3)], indexing="ij")
    d2 = sum(((g - idx_zyx[i]) * spacing_zyx[i]) ** 2 for i, g in enumerate(grid))
    out = np.zeros(shape, dtype=np.float32)
    out[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = np.exp(-d2 / (2 * sigma_mm ** 2))
    mx = out.max()
    return out / mx if mx > 0 else out


def _plane_from_points(pts: np.ndarray):
    center = pts.mean(axis=0)
    u, s, vt = np.linalg.svd(pts - center)
    normal = vt[-1]
    normal = normal / np.linalg.norm(normal)
    radial = np.linalg.norm((pts - center) - np.outer((pts - center) @ normal, normal), axis=1)
    return center, normal, float(radial.mean())


def _ring(shape, spacing_zyx, img, center, normal, radius, sigma_mm):
    """在 crop 栅格上渲染软环：到平面的距离 × 到轴的距离，都做高斯衰减。"""
    # 只在半径+3σ 的世界包围盒内计算
    rr = radius + 3 * sigma_mm
    half = np.ceil(rr / np.array(spacing_zyx)).astype(int)
    cidx = np.asarray(img.TransformPhysicalPointToContinuousIndex(
        [float(v) for v in center]))[::-1]
    lo = np.maximum(0, np.floor(cidx).astype(int) - half)
    hi = np.minimum(np.array(shape), np.ceil(cidx).astype(int) + half + 1)
    if np.any(hi <= lo):
        return np.zeros(shape, dtype=np.float32)
    zz, yy, xx = np.meshgrid(*[np.arange(lo[i], hi[i]) for i in range(3)], indexing="ij")
    # 体素 -> 世界（向量化：p = origin + A @ (idx * spacing)，逐体素调 API 会慢两个数量级）
    idx_xyz = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(float)
    A = np.asarray(img.GetDirection(), dtype=float).reshape(3, 3)
    s = np.asarray(img.GetSpacing(), dtype=float)
    origin = np.asarray(img.GetOrigin(), dtype=float)
    world = origin + (idx_xyz * s) @ A.T
    d = world - center
    dist_plane = d @ normal
    in_plane = d - np.outer(dist_plane, normal)
    r = np.linalg.norm(in_plane, axis=1)
    val = np.exp(-(r - radius) ** 2 / (2 * sigma_mm ** 2)) * \
        np.exp(-dist_plane ** 2 / (2 * sigma_mm ** 2))
    out = np.zeros(shape, dtype=np.float32)
    out[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = val.reshape(hi - lo)
    return out


def _process_targets(job):
    case, grid = job
    odir = f"{C.out_dir(grid)}/{case['id']}"
    img_path = f"{odir}/image_crop.nii.gz"
    if not os.path.exists(img_path):
        return (case["id"], "fail", "no crop")
    if os.path.exists(f"{odir}/targets_done.json"):
        return (case["id"], "skip", "")

    img = sitk.ReadImage(img_path)
    arr = sitk.GetArrayFromImage(img)          # (z, y, x)
    shape = arr.shape
    sp_zyx = np.asarray(img.GetSpacing())[::-1]

    gt = C.case_gt(case)
    hms = np.zeros((len(C.LANDMARKS),) + shape, dtype=np.float32)
    ex = []
    landmarks_vox = {}
    for i, name in enumerate(C.LANDMARKS):
        pts = C.read_markup(f"{gt}/{name}.mrk.json")
        if pts is None:
            ex.append(0)
            continue
        p = pts.mean(axis=0)
        idx_xyz = C.world_to_voxel(img, p)[0]
        idx_zyx = idx_xyz[::-1]
        hms[i] = _heatmap(shape, sp_zyx, idx_zyx, SIGMA_MM)
        landmarks_vox[name] = idx_xyz.tolist()
        ex.append(1)

    hdir = f"{odir}/targets"
    os.makedirs(hdir, exist_ok=True)
    for i, name in enumerate(C.LANDMARKS):
        hm = sitk.GetImageFromArray(hms[i])
        hm.CopyInformation(img)
        sitk.WriteImage(hm, f"{hdir}/{name}_heatmap.nii.gz")

    plane = None
    ann = C.read_markup(f"{gt}/annulus.mrk.json")
    if ann is not None and len(ann) >= 3:
        center, normal, radius = _plane_from_points(ann)
        ring = _ring(shape, sp_zyx, img, center, normal, radius, RING_SIGMA_MM)
        ring_img = sitk.GetImageFromArray(ring)
        ring_img.CopyInformation(img)
        sitk.WriteImage(ring_img, f"{hdir}/annulus_ring.nii.gz")
        plane = dict(center=center.tolist(), normal=normal.tolist(), radius=radius)

    C.write_json(f"{odir}/landmarks_voxel.json", dict(
        existence=ex, voxel_xyz=landmarks_vox, grid=grid,
        spacing=list(C.GRIDS[grid]["spacing"]), size=list(shape[::-1]),
    ))
    C.write_json(f"{odir}/plane.json", plane)
    C.write_json(f"{odir}/targets_done.json", True)
    return (case["id"], "ok", "")


def cmd_targets(args):
    cases = C.read_json(f"{C.WORK}/cases.json")
    jobs = [(dict(c, id=k), args.grid) for k, c in cases.items()
            if c["split"] != "pretrain_ssl"]          # 无标注，只裁剪，不做热图
    stats = {"ok": 0, "skip": 0, "fail": 0}
    fails = []
    with Pool(args.jobs) as pool:
        for cid, status, msg in pool.imap_unordered(_process_targets, jobs):
            stats[status] += 1
            if status == "fail":
                fails.append({"case": cid, "reason": msg})
                print(f"  FAIL {cid}: {msg}")
    C.write_json(f"{C.out_dir(args.grid)}/targets_failures.json", fails)
    print(f"targets grid={args.grid} {stats}")


# ================================================================ qa
def cmd_qa(args):
    cases = C.read_json(f"{C.WORK}/cases.json")
    rows, excluded = [], []
    for cid, case in cases.items():
        if case["split"] == "pretrain_ssl":
            continue
        odir = f"{C.out_dir(args.grid)}/{cid}"
        meta_p = f"{odir}/landmarks_voxel.json"
        if not os.path.exists(meta_p):
            excluded.append({"case": cid, "reason": "missing targets"})
            continue
        meta = C.read_json(meta_p)
        size = np.array(C.GRIDS[args.grid]["size"])          # x,y,z
        ex = np.array(meta["existence"])
        bad = []
        for name, p in meta["voxel_xyz"].items():
            i = C.LANDMARKS.index(name)
            q = np.array(p)
            if np.any(q < QA_MARGIN_VOX) or np.any(q > size - QA_MARGIN_VOX):
                bad.append(name)
                ex[i] = 0
        if bad:
            meta["existence"] = ex.tolist()
            C.write_json(meta_p, meta)
        rows.append(dict(case=cid, existence="".join(str(v) for v in ex),
                         out_of_bounds=",".join(bad)))
    os.makedirs(C.out_dir(args.grid), exist_ok=True)
    with open(f"{C.out_dir(args.grid)}/qa_report.csv", "w") as f:
        f.write("case,existence,out_of_bounds\n")
        for r in rows:
            f.write(f"{r['case']},{r['existence']},{r['out_of_bounds']}\n")
    C.write_json(f"{C.out_dir(args.grid)}/exclude.json", excluded)
    print(f"qa grid={args.grid} checked={len(rows)} excluded={len(excluded)}")
    print("  注意：out_of_bounds 的点已被置 0，请抽查若干例确认不是裁剪中心错了")


# ================================================================ split
def cmd_split(args):
    cases = C.read_json(f"{C.WORK}/cases.json")
    grid_dir = C.out_dir(args.grid)

    def sample(cid, case):
        odir = f"{grid_dir}/{cid}"
        meta = C.read_json(f"{odir}/landmarks_voxel.json")
        root = f"{odir}/root_crop.nii.gz"
        calc = f"{odir}/calcification_crop.nii.gz"
        return dict(
            case_id=cid,
            image=f"{odir}/image_crop.nii.gz",
            landmarks=f"{odir}/landmarks_voxel.json",
            plane=f"{odir}/plane.json",
            root=root if os.path.exists(root) else None,
            calcification=calc if os.path.exists(calc) else None,
            heatmaps={n: f"{odir}/targets/{n}_heatmap.nii.gz" for n in C.LANDMARKS},
            annulus_ring=(f"{odir}/targets/annulus_ring.nii.gz"
                          if os.path.exists(f"{odir}/targets/annulus_ring.nii.gz") else None),
            existence=meta["existence"],
        )

    huaxi = [cid for cid, c in cases.items() if c["split"] == "huaxi"]
    carry = [cid for cid, c in cases.items() if c["split"] == "carry"]
    pretrain = [cid for cid, c in cases.items() if c["split"] == "pretrain"]
    huaxi = [c for c in huaxi if os.path.exists(f"{grid_dir}/{c}/landmarks_voxel.json")]
    carry = [c for c in carry if os.path.exists(f"{grid_dir}/{c}/landmarks_voxel.json")]
    # 预训练例只进 train，不进任何一折的验证集
    pretrain = [c for c in pretrain if os.path.exists(f"{grid_dir}/{c}/landmarks_voxel.json")]

    # 按 existence 模式分层，避免某折全缺 N
    groups: dict[str, list[str]] = {}
    for cid in huaxi:
        key = "".join(str(v) for v in cases[cid]["existence"])
        groups.setdefault(key, []).append(cid)
    rng = np.random.RandomState(42)
    folds = [{"train": [], "val": []} for _ in range(3)]
    for key in sorted(groups):
        ids = sorted(groups[key])
        rng.shuffle(ids)
        for i, cid in enumerate(ids):
            # 轮转分配：第 (i%3) 折把它当验证，其余两折当训练
            folds[i % 3]["val"].append(cid)
    for f in folds:
        val = set(f["val"])
        f["train"] = [c for c in huaxi if c not in val] + pretrain

    out = {"grid": args.grid,
           "folds": {f"fold_{i}": {"train": [sample(c, cases[c]) for c in f["train"]],
                                   "val": [sample(c, cases[c]) for c in f["val"]]}
                     for i, f in enumerate(folds)},
           "external": [sample(c, cases[c]) for c in carry]}
    C.write_json(f"{grid_dir}/datalist.json", out)

    # HU 归一化统计：随机 30 例、只取软组织及以上（HU>0）。
    # 阈值取 -500 时肺/空气会主导低分位，软组织和造影剂的动态范围被压扁。
    pick = list(rng.choice(huaxi, min(30, len(huaxi)), replace=False))
    vals = []
    for cid in pick:
        a = sitk.GetArrayFromImage(sitk.ReadImage(f"{grid_dir}/{cid}/image_crop.nii.gz"))
        vals.append(a[a > 0].ravel()[::37])
    v = np.concatenate(vals)
    norm = dict(p_lo=float(np.percentile(v, 0.5)), p_hi=float(np.percentile(v, 99.5)),
                n_samples=len(pick), n_voxels=int(v.size))
    C.write_json(f"{grid_dir}/norm.json", norm)
    print(f"split grid={args.grid} pretrain_per_fold={len(pretrain)} "
          f"folds={[ (len(f['train']), len(f['val'])) for f in folds ]} "
          f"external={len(carry)} norm=[{norm['p_lo']:.1f}, {norm['p_hi']:.1f}]")


# ================================================================ cli
def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("index")
    pcp = sub.add_parser("convert-pretrain")
    pcp.add_argument("--include-unlabeled", action="store_true",
                     help="无开口标注的 775 例 fct + 571 例 dicom 登记为 pretrain_ssl（E3 自监督用）")

    pl = sub.add_parser("localize")
    pl.add_argument("--task", default="total")
    pl.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    pl.add_argument("--fast", action="store_true", help="用 fast 模型（3mm，质心够用）")
    pl.add_argument("--limit", type=int, default=0, help="只跑前 N 例（测试用）")
    pl.add_argument("--jobs", type=int, default=4)

    pc = sub.add_parser("crop")
    pc.add_argument("--grid", choices=list(C.GRIDS), required=True)
    pc.add_argument("--center", choices=["gt", "totalseg"], default="gt")
    pc.add_argument("--jobs", type=int, default=4)

    pt = sub.add_parser("targets")
    pt.add_argument("--grid", choices=list(C.GRIDS), required=True)
    pt.add_argument("--jobs", type=int, default=4)

    pq = sub.add_parser("qa")
    pq.add_argument("--grid", choices=list(C.GRIDS), required=True)

    ps = sub.add_parser("split")
    ps.add_argument("--grid", choices=list(C.GRIDS), required=True)

    args = p.parse_args()
    os.makedirs(C.WORK, exist_ok=True)
    {"index": cmd_index, "convert-pretrain": cmd_convert_pretrain,
     "localize": cmd_localize, "crop": cmd_crop,
     "targets": cmd_targets, "qa": cmd_qa, "split": cmd_split}[args.cmd](args)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
