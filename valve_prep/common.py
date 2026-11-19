"""valve_prep 公共部分：路径、Slicer markup 解析、病例枚举、栅格裁剪。

数据源永远是 release 包，不读 /data/disk_c/valve_pipeline。
"""
from __future__ import annotations

import json
import os

import numpy as np
import SimpleITK as sitk

RELEASE = os.environ.get(
    "VALVE_RELEASE",
    "/media/dell/895d06b8-ef17-44d2-b48e-3cbba7183b51/valve_release_v1_20261113",
)
WORK = os.environ.get(
    "VALVE_WORK",
    os.path.expanduser("~/program/CTA-Foudation-model/work/prep"),
)

LANDMARKS = ["L", "R", "N", "LCA", "RCA"]

# 只有一种栅格：Merlin 原生 spacing
GRIDS = {
    "merlin": dict(spacing=(1.5, 1.5, 3.0), size=(224, 224, 160), hu=(-1000.0, 1000.0)),
}

# 裁剪中心相对主动脉根质心的偏移（世界坐标，LPS：+x 左、+y 后、+z 上）
# "右移 16mm、上移 16mm" -> x 取负、z 取正。单例闭环时必须在 Slicer 里复核。
CENTER_OFFSET_MM = np.array([-16.0, 0.0, 16.0])

# TotalSegmentator 心脏质心 -> 主动脉根质心的群体偏移（常数）。
# huaxi 188 例校准：中位 (-5.1, -1.1, +5.6) mm，逐例 |偏移| 中位 11.2 mm。
# 只有一个全局常数，逐例中心仍完全由图像分割决定。
HEART_TO_ROOT_MM = np.array([-5.1, -1.1, 5.6])


# ---------------------------------------------------------------- 病例枚举
def split_dir(split: str) -> str:
    return f"{RELEASE}/downstream/{split}"


def case_ids(split: str) -> list[str]:
    base = split_dir(split)
    return sorted(d for d in os.listdir(base) if os.path.isdir(f"{base}/{d}"))


def case_dir(split: str, dirname: str) -> str:
    return f"{split_dir(split)}/{dirname}"


def case_gt(case: dict) -> str:
    """病例的 gt 目录；预训练病例用 WORK 下转换出来的合成 gt。"""
    return case.get("gt_dir") or f"{case_dir(case['split'], case['dirname'])}/gt"


def parse_case_id(dirname: str) -> str:
    """dicom_pid_0000034090_20201215 -> huaxi_0000034090（split 由调用方补）"""
    parts = dirname.split("_")
    return parts[2]


def out_dir(grid: str) -> str:
    return f"{WORK}/{grid}"


# ---------------------------------------------------------------- 标注读取
def read_markup(path: str) -> np.ndarray | None:
    """读 Slicer .mrk.json，返回 (N,3) LPS 世界坐标；文件不存在或没有点返回 None。"""
    if not path or not os.path.exists(path):
        return None
    doc = json.load(open(path))
    for m in doc.get("markups", []):
        cps = m.get("controlPoints") or []
        if cps:
            return np.asarray([cp["position"] for cp in cps], dtype=float)
    return None


def landmark_paths(gt_dir: str) -> dict[str, str | None]:
    return {
        name: (f"{gt_dir}/{name}.mrk.json"
               if os.path.exists(f"{gt_dir}/{name}.mrk.json") else None)
        for name in LANDMARKS
    }


def gt_paths(gt_dir: str) -> dict[str, str]:
    return {
        "annulus": f"{gt_dir}/annulus.mrk.json",
        "root": f"{gt_dir}/root.nrrd",
        "calcification": f"{gt_dir}/calcification.nrrd",
        "gt_dir": gt_dir,
    }


def existence_mask(gt_dir: str) -> list[int]:
    return [1 if os.path.exists(f"{gt_dir}/{n}.mrk.json") else 0 for n in LANDMARKS]


# ---------------------------------------------------------------- 重采样与裁剪
def read_image(path: str) -> sitk.Image:
    return sitk.ReadImage(path)


def resample_to_spacing(img: sitk.Image, spacing, interp=sitk.sitkLinear,
                        default: float = -1000.0) -> sitk.Image:
    """保持 origin/direction/FOV 起点，改 spacing；size 按物理长度换算。"""
    old_sp = np.asarray(img.GetSpacing(), dtype=float)
    old_sz = np.asarray(img.GetSize(), dtype=float)
    new_sz = np.maximum(1, np.round(old_sz * old_sp / np.asarray(spacing, dtype=float))).astype(int)
    return sitk.Resample(
        img,
        [int(v) for v in new_sz],
        sitk.Transform(),
        interp,
        img.GetOrigin(),
        [float(s) for s in spacing],
        img.GetDirection(),
        float(default),
    )


def resample_like(img: sitk.Image, ref: sitk.Image, interp=sitk.sitkNearestNeighbor,
                  default: float = 0.0) -> sitk.Image:
    return sitk.Resample(img, ref, sitk.Transform(), interp, float(default), sitk.sitkFloat32)


def crop_params(img: sitk.Image, center_world, size) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """算出裁剪起点（可为负）与需要的 padding。"""
    ci = np.asarray(img.TransformPhysicalPointToContinuousIndex(
        [float(v) for v in center_world]), dtype=float)
    size = np.asarray(size, dtype=int)
    start = np.round(ci - (size - 1) / 2.0).astype(int)
    pad_lo = np.maximum(0, -start)
    pad_hi = np.maximum(0, start + size - np.asarray(img.GetSize()))
    return start, pad_lo, pad_hi


def apply_crop(img: sitk.Image, start, pad_lo, pad_hi, size, fill: float) -> sitk.Image:
    start = np.asarray(start, dtype=int)
    pad_lo = np.asarray(pad_lo, dtype=int)
    pad_hi = np.asarray(pad_hi, dtype=int)
    if pad_lo.any() or pad_hi.any():
        img = sitk.ConstantPad(img, [int(v) for v in pad_lo], [int(v) for v in pad_hi], float(fill))
    return sitk.RegionOfInterest(
        img,
        [int(v) for v in np.asarray(size, dtype=int)],
        [int(v) for v in (start + pad_lo)],
    )


def world_to_voxel(img: sitk.Image, points: np.ndarray) -> np.ndarray:
    """(N,3) 世界坐标 -> (N,3) 连续体素索引"""
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    return np.asarray([img.TransformPhysicalPointToContinuousIndex([float(v) for v in p])
                       for p in pts], dtype=float)


# ---------------------------------------------------------------- 顺序读写
def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)


def read_json(path: str):
    with open(path) as f:
        return json.load(f)
