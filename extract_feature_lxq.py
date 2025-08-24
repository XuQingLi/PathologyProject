# -*- coding: utf-8 -*-
import os
import sys
import json
import math
import glob
import argparse
from typing import List, Tuple

import uuid
import shutil
import errno
import time
import stat
import datetime

import numpy as np
from PIL import Image
from tqdm import tqdm

import h5py
import zarr
from numcodecs import Blosc
import tifffile as tiff  # 保持一致性，哪怕当前未直接使用

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

from openslide import OpenSlide, OpenSlideError
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# =========================
# 实用工具（清理/收尾/兼容）
# =========================
def _rm_path_force(p: str):
    """无论文件/目录，尽力删干净（含权限兜底）。"""
    try:
        if os.path.isdir(p) and not os.path.islink(p):
            shutil.rmtree(p, ignore_errors=True)
        else:
            os.unlink(p)
    except FileNotFoundError:
        return
    except Exception:
        try:
            os.chmod(p, stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR)
            if os.path.isdir(p) and not os.path.islink(p):
                shutil.rmtree(p, ignore_errors=True)
            else:
                os.unlink(p)
        except Exception:
            pass

def _safe_finalize_dir(tmp_dir: str, final_dir: str, overwrite: bool) -> bool:
    """
    把 tmp_dir -> final_dir：
      1) 优先 os.replace（原子）
      2) 失败则强制清理 final 后重试
      3) 再失败则 copytree
      4) 仍失败：保留失败件，改名为 *.failed-<ts>
    返回 True 表示产物已就位或被跳过（不再遗留 tmp）
    """
    parent = os.path.dirname(final_dir)
    os.makedirs(parent, exist_ok=True)

    # 目标已存在
    if os.path.exists(final_dir):
        if overwrite:
            _rm_path_force(final_dir)
        else:
            # 允许跳过：删除 tmp，直接成功返回
            _rm_path_force(tmp_dir)
            print(f"[SKIP] Zarr exists: {final_dir}")
            return True

    # 1) 原子替换
    try:
        os.replace(tmp_dir, final_dir)
        return True
    except Exception as e1:
        # 2) 清理再试
        try:
            _rm_path_force(final_dir)
            os.replace(tmp_dir, final_dir)
            return True
        except Exception as e2:
            # 3) fallback: copytree
            try:
                shutil.copytree(tmp_dir, final_dir)
                shutil.rmtree(tmp_dir, ignore_errors=True)
                print(f"[WARN] os.replace failed ({e1} / {e2}); used copytree fallback.")
                return True
            except Exception as e3:
                # 4) 保留失败件（仅 1 份）
                ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                failed_dir = f"{final_dir}.failed-{ts}"
                try:
                    os.rename(tmp_dir, failed_dir)
                    print(f"[FAIL] finalize: kept partial at {failed_dir}  ({e1} / {e2} / {e3})")
                except Exception:
                    print(f"[FAIL] finalize: kept partial at {tmp_dir}  ({e1} / {e2} / {e3})")
                return False

def _safe_finalize_file(tmp_path: str, final_path: str, overwrite: bool) -> bool:
    """H5 收尾：不再丢 .trash；目标存在且跳过则删除 tmp。"""
    base_dir = os.path.dirname(final_path)
    os.makedirs(base_dir, exist_ok=True)

    if os.path.exists(final_path):
        if overwrite:
            _rm_path_force(final_path)
        else:
            _rm_path_force(tmp_path)
            print(f"[SKIP] H5 exists: {final_path}")
            return True

    try:
        os.replace(tmp_path, final_path)
        return True
    except Exception as e1:
        try:
            _rm_path_force(final_path)
            os.replace(tmp_path, final_path)
            return True
        except Exception as e2:
            try:
                shutil.copy2(tmp_path, final_path)
                _rm_path_force(tmp_path)
                print(f"[WARN] H5 os.replace failed ({e1}/{e2}); used copy fallback.")
                return True
            except Exception as e3:
                _rm_path_force(tmp_path)  # 最终失败也删除 tmp，避免遗留
                print(f"[FAIL] finalize H5 failed ({e1}/{e2}/{e3}); tmp removed.")
                return False

def _quarantine(path: str, root: str):
    """兼容旧调用，但不再制造 .trash-*；直接清理。"""
    _rm_path_force(path)
    print(f"[CLEAN] removed stale tmp: {path}")

def _json_load_maybe(x):
    if isinstance(x, (bytes, str)):
        try:
            return json.loads(x)
        except Exception:
            return x
    return x

def is_wsi_file(name: str) -> bool:
    low = name.lower()
    return low.endswith((".tif", ".tiff", ".svs"))

def tissue_ratio(rgb: np.ndarray, thr: int = 220) -> float:
    """粗略组织检测：统计像素是否非亮背景的比例。"""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return 0.0
    mask = (rgb < thr).any(axis=2)
    return float(mask.mean())

# =========================
# 编码器（返回 c2..c5）
# =========================
class ResNet50C245(nn.Module):
    """
    ResNet50 提取 c2..c5：
      c2: /4   输出通道 256
      c3: /8   输出通道 512
      c4: /16  输出通道 1024
      c5: /32  输出通道 2048
    """
    def __init__(self, pretrained=True):
        super().__init__()
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1 = m.layer1  # c2
        self.layer2 = m.layer2  # c3
        self.layer3 = m.layer3  # c4
        self.layer4 = m.layer4  # c5

    def forward(self, x):
        x = self.stem(x)       # /4
        c2 = self.layer1(x)    # /4
        c3 = self.layer2(c2)   # /8
        c4 = self.layer3(c3)   # /16
        c5 = self.layer4(c4)   # /32
        return c2, c3, c4, c5

# =========================
# H5 Writer（稳健 finalize）
# =========================
class H5FeatureWriter:
    def __init__(self, out_path: str, input_size: Tuple[int,int], downsample: dict):
        self.final_path = out_path
        base_dir = os.path.dirname(out_path)
        os.makedirs(base_dir, exist_ok=True)
        self.tmp_path = out_path + f".tmp-{os.getpid()}-{uuid.uuid4().hex[:6]}"

        # 禁用 HDF5 文件锁，适配 NFS/并发环境
        os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

        self.f = h5py.File(self.tmp_path, "w", libver="latest")
        self.f.create_group("meta")
        self.f["meta"].attrs["input_size"] = json.dumps([int(input_size[0]), int(input_size[1])])
        self.f["meta"].attrs["downsample"] = json.dumps({k: list(map(int, v)) for k, v in downsample.items()})
        self.coords_ds = None
        self._n = 0
        self._layers = {}
        # 默认不覆盖；由调用方在需要时设置 self._overwrite_ = True
        self._overwrite_ = False

    def _ensure_layer(self, name: str, C: int, Hf: int, Wf: int, chunk_N: int = 256):
        if name not in self._layers:
            grp = self.f.create_group(name)
            dset = grp.create_dataset(
                "features",
                shape=(0, C, Hf, Wf),
                maxshape=(None, C, Hf, Wf),
                chunks=(chunk_N, C, Hf, Wf),
                dtype="float16",
                compression="gzip", compression_opts=4
            )
            self._layers[name] = dset

    def _ensure_coords(self, chunk_N: int = 1024):
        if self.coords_ds is None:
            self.coords_ds = self.f.create_dataset(
                "coords",
                shape=(0, 2),
                maxshape=(None, 2),
                chunks=(chunk_N, 2),
                dtype="int32",
                compression="gzip", compression_opts=4
            )

    def append_batch(self, coords_xy: np.ndarray, feats: dict):
        B = coords_xy.shape[0]
        self._ensure_coords()
        n0 = self.coords_ds.shape[0]
        self.coords_ds.resize((n0 + B, 2))
        self.coords_ds[n0:n0+B] = coords_xy.astype("int32")

        for name, arr in feats.items():
            B_, C, Hf, Wf = arr.shape
            assert B_ == B
            self._ensure_layer(name, C, Hf, Wf)
            dset = self._layers[name]
            n0 = dset.shape[0]
            dset.resize((n0 + B, C, Hf, Wf))
            dset[n0:n0+B] = arr.astype(np.float16)

        self._n += B

    def close(self):
        try:
            self.f.flush()
            self.f.close()
        finally:
            overwrite = bool(getattr(self, "_overwrite_", False))
            _safe_finalize_file(self.tmp_path, self.final_path, overwrite)

# =========================
# Zarr 导出（密集网格 + 覆盖掩码 + MIL 向量）
# =========================
def export_dense_feature_grids_to_zarr(
    h5_path: str,
    slide_path: str,
    out_zarr_dir: str,
    layers=("c2","c3","c4"),
    dtype="float16",
    chunk_hw=256,
    save_cover_mask=True,
    save_cls_head=True,
    max_chunk_bytes=64 * 1024 * 1024
):
    """
    累加阶段使用内存 numpy 数组，最终再一次性写入 Zarr；
    .zarr 写入到临时目录，完成后 os.replace 原子替换。
    """
    final_dir = out_zarr_dir
    tmp_dir   = out_zarr_dir.rstrip("/\\") + f".tmp-{os.getpid()}"

    # 若目录存在且允许跳过：直接返回（兼容原语义）
    if os.path.isdir(final_dir) and getattr(export_dense_feature_grids_to_zarr, "_skip_if_done", True) \
       and not getattr(export_dense_feature_grids_to_zarr, "_overwrite", False):
        print(f"[SKIP] Zarr exists (ignore tmp): {final_dir}")
        return

    # 清理历史 tmp（不再 quarantine）
    if os.path.exists(tmp_dir):
        _rm_path_force(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)

    def _choose_chunks(C, H, W, itemsize, prefer_max_side=chunk_hw, cap=max_chunk_bytes):
        c = min(C, 32) if C >= 64 else min(C, 16)
        max_area = max(1, cap // max(1, (c * itemsize)))
        side = int(max(1, min(prefer_max_side, int(math.sqrt(max_area)))))
        h = min(H, side if side > 0 else 1)
        w = min(W, side if side > 0 else 1)
        while c * h * w * itemsize > cap and c > 1:
            c = max(1, c // 2)
        return (max(1, c), max(1, h), max(1, w))

    # 读取 WSI 尺寸
    slide = OpenSlide(slide_path)
    W_wsi, H_wsi = slide.level_dimensions[0]
    slide.close()

    with h5py.File(h5_path, "r") as f:
        coords = f["coords"][:].astype(np.int64)
        meta   = dict(f["meta"].attrs.items()) if "meta" in f else {}
        meta   = {k: (_json_load_maybe(v)) for k,v in meta.items()}
        H_in, W_in = meta.get("input_size", [256,256])
        down = meta.get("downsample", {"c2":[4,4],"c3":[8,8],"c4":[16,16],"c5":[32,32]})

        store = zarr.DirectoryStore(tmp_dir)
        root = zarr.group(store=store, overwrite=True)
        root.attrs["wsi_size"]   = (int(W_wsi), int(H_wsi))
        root.attrs["input_size"] = (int(H_in), int(W_in))
        root.attrs["downsample"] = down
        root.attrs["source_h5"]  = os.path.abspath(h5_path)

        # ---- slide-level 分类向量 + MIL ----
        if save_cls_head and ("c5" in f) and ("features" in f["c5"]):
            feats_c5 = f["c5/features"]  # (N, C5, Hf5, Wf5)
            N_c5, C5, Hf5, Wf5 = feats_c5.shape

            # GAP 后求 slide 级向量
            sum_vec = np.zeros((C5,), dtype=np.float64); cnt = 0
            bs = 1024
            for s in range(0, N_c5, bs):
                arr = feats_c5[s:s+bs].astype(np.float32)
                arr = arr.mean(axis=(2,3))     # (B, C5)
                sum_vec += arr.sum(axis=0); cnt += arr.shape[0]
            cls_vec = (sum_vec / max(1, cnt)).astype(np.float32)
            root.create_dataset("cls_c5_gap", data=cls_vec, shape=cls_vec.shape,
                                chunks=cls_vec.shape, dtype="float32", compressor=compressor)

            mil_grp = root.require_group("mil")
            emb_chunks = (min(4096, N_c5), C5)
            emb_z = mil_grp.create_dataset(
                "emb", shape=(N_c5, C5),
                chunks=emb_chunks, dtype="float16", compressor=compressor
            )
            bs = 1024
            for s in range(0, N_c5, bs):
                e = min(N_c5, s+bs)
                arr = feats_c5[s:e].astype(np.float32).mean(axis=(2,3))  # (B,C5)
                emb_z[s:e] = arr.astype(np.float16)

            coords_all = f["coords"][:].astype("int32")
            mil_grp.create_dataset("coords", data=coords_all, shape=coords_all.shape,
                                   chunks=(min(4096, coords_all.shape[0]), 2),
                                   dtype="int32", compressor=compressor)
            v = down.get("c5", [32,32])
            ds_c5 = (int(v[0]), int(v[1])) if isinstance(v, (list,tuple)) else (int(v), int(v))
            mil_grp.attrs["downsample"] = ds_c5

        # ---- 稠密网格（内存累加 → 一次写回）----
        for lname in layers:
            assert lname in f and "features" in f[lname], f"layer '{lname}/features' missing"
            feats_ds = f[f"{lname}/features"]  # (N, C, Hf, Wf)
            v = down.get(lname, [4,4])
            ds_h, ds_w = (int(v[0]), int(v[1])) if isinstance(v, (list,tuple)) else (int(v), int(v))
            N, C, Hf, Wf = feats_ds.shape
            Hf_glb = (H_wsi + ds_h - 1)//ds_h
            Wf_glb = (W_wsi + ds_w - 1)//ds_w

            sum_arr_np = np.zeros((C, Hf_glb, Wf_glb), dtype=np.float32)
            cnt_arr_np = np.zeros((1, Hf_glb, Wf_glb), dtype=np.uint32)

            bs = 1024
            xs_patch = (coords[:,0] // ds_w).astype(np.int64)
            ys_patch = (coords[:,1] // ds_h).astype(np.int64)
            for s in tqdm(range(0, N, bs), desc=f"[{os.path.basename(h5_path)}] accumulate {lname}"):
                e = min(N, s+bs)
                arr = feats_ds[s:e].astype(np.float32, copy=False)  # (B,C,Hf,Wf)
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                xs = xs_patch[s:e]; ys = ys_patch[s:e]

                for i in range(e - s):
                    x0 = int(xs[i]); y0 = int(ys[i])
                    y1 = y0 + Hf; x1 = x0 + Wf
                    yy0, yy1 = max(0,y0), min(Hf_glb,y1)
                    xx0, xx1 = max(0,x0), min(Wf_glb,x1)
                    ph0, ph1 = yy0 - y0, y1 - yy1
                    pw0, pw1 = xx0 - x0, x1 - xx1
                    patch = arr[i,:,ph0:Hf-ph1, pw0:Wf-pw1]
                    if patch.size == 0:
                        continue
                    sum_arr_np[:, yy0:yy1, xx0:xx1] += patch
                    cnt_arr_np[:, yy0:yy1, xx0:xx1] += 1

            den_c, den_h, den_w = _choose_chunks(C, Hf_glb, Wf_glb, itemsize=(2 if str(dtype)=="float16" else 4))
            dense = root.create_dataset(
                f"{lname}", shape=(C, Hf_glb, Wf_glb),
                chunks=(den_c, den_h, den_w),
                dtype=dtype, compressor=compressor
            )
            step_h, step_w = den_h, den_w
            for y in range(0, Hf_glb, step_h):
                y1 = min(Hf_glb, y+step_h)
                for x in range(0, Wf_glb, step_w):
                    x1 = min(Wf_glb, x+step_w)
                    sblk = sum_arr_np[:, y:y1, x:x1]
                    cblk = cnt_arr_np[:, y:y1, x:x1].astype(np.float32)
                    cblk = np.maximum(cblk, 1.0)
                    dense[:, y:y1, x:x1] = (sblk / cblk).astype(dtype)

            if save_cover_mask:
                m_c, m_h, m_w = _choose_chunks(1, Hf_glb, Wf_glb, itemsize=1)
                cover = root.create_dataset(
                    f"{lname}_mask", shape=(1, Hf_glb, Wf_glb),
                    chunks=(m_c, m_h, m_w),
                    dtype="uint8", compressor=compressor
                )
                cover[...] = (cnt_arr_np[...] > 0).astype(np.uint8)

    # —— 稳健 finalize（不再 quarantine；目标存在且 skip 则删 tmp）——
    overwrite = bool(getattr(export_dense_feature_grids_to_zarr, "_overwrite", False))
    ok = _safe_finalize_dir(tmp_dir, final_dir, overwrite)
    if ok:
        print(f"[OK] Dense feature grids (and MIL view) exported to: {final_dir}")
    else:
        print(f"[FAIL] Exported store kept as *.failed-* for manual recovery: {final_dir}")

# =========================
# 主流程：提特征 → 写 H5 → 导出 Zarr
# =========================
def extract_one_slide(
    slide_path: str,
    out_dir: str,
    model: nn.Module,
    device: str = "cuda",
    patch: int = 224,
    stride: int = 224,
    batch_size: int = 64,
    tissue_thr: float = 0.05,
    save_h5: bool = True,
    export_zarr: bool = True
):
    os.makedirs(out_dir, exist_ok=True)
    slide_id = os.path.splitext(os.path.basename(slide_path))[0]
    h5_path = os.path.join(out_dir, f"{slide_id}.h5")
    zarr_dir = os.path.join(out_dir, f"{slide_id}.zarr")

    # 若成品已存在而且允许跳过：直接跳过该 slide
    if os.path.isfile(h5_path) and os.path.isdir(zarr_dir) and getattr(extract_one_slide, "_skip_if_done", True) \
       and not getattr(extract_one_slide, "_overwrite", False):
        print(f"[SKIP] H5 & Zarr exist: {os.path.basename(slide_path)}")
        return

    # 清理所有历史 tmp（跨 pid），避免冲突与膨胀
    for p in glob.glob(h5_path + ".tmp-*") + glob.glob(zarr_dir + ".tmp-*"):
        _rm_path_force(p)

    # 跳过策略（分情形）
    if save_h5 and not export_zarr:
        if os.path.isfile(h5_path) and not getattr(extract_one_slide, "_overwrite", False) and getattr(extract_one_slide, "_skip_if_done", True):
            print(f"[SKIP] H5 已存在：{h5_path}")
            return
    if (not save_h5) and export_zarr:
        if os.path.isdir(zarr_dir) and not getattr(extract_one_slide, "_overwrite", False) and getattr(extract_one_slide, "_skip_if_done", True):
            print(f"[SKIP] Zarr 已存在：{zarr_dir}")
            return
    if save_h5 and export_zarr:
        if os.path.isfile(h5_path) and os.path.isdir(zarr_dir) and not getattr(extract_one_slide, "_overwrite", False) and getattr(extract_one_slide, "_skip_if_done", True):
            print(f"[SKIP] H5 与 Zarr 均已存在：{slide_id}")
            return

    try:
        slide = OpenSlide(slide_path)
    except OpenSlideError as e:
        print(f"[WARN] skip (cannot open): {slide_path}  ({e})")
        return

    W, H = slide.level_dimensions[0]
    print(f"[INFO] slide={slide_id} size=({W},{H})")

    # 缩略掩码
    thumb_down = 32
    tw, th = max(1, W // thumb_down), max(1, H // thumb_down)
    thumb = slide.get_thumbnail((tw, th)).convert("RGB")
    thumb_np = np.array(thumb)
    mask_small = (thumb_np < 220).any(axis=2)  # True=组织
    mask_big = np.array(Image.fromarray(mask_small.astype(np.uint8) * 255)
                        .resize((max(1, W // stride), max(1, H // stride)), Image.NEAREST)) > 0

    # 覆盖率兜底
    cover = float(mask_big.mean())
    if cover < 1e-3:
        mask_small2 = (thumb_np < 240).any(axis=2)
        mask_big2 = np.array(Image.fromarray(mask_small2.astype(np.uint8)*255)
                     .resize((max(1, W // stride), max(1, H // stride)), Image.NEAREST)) > 0
        if mask_big2.mean() > cover:
            mask_big = mask_big2
            cover = float(mask_big.mean())
    if cover < 1e-4:
        mask_big[:] = True

    xs = list(range(0, W - patch + 1, stride))
    ys = list(range(0, H - patch + 1, stride))
    coords = [(x, y) for j, y in enumerate(ys) for i, x in enumerate(xs) if mask_big[j, i]]
    print(f"[INFO] valid patches: {len(coords)}")

    # 预处理与模型
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    model.eval().to(device)
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    h5w_box = {"obj": None}
    downsample = {"c2": [4,4], "c3": [8,8], "c4": [16,16], "c5": [32,32]}

    batch_imgs: List[torch.Tensor] = []
    batch_coords: List[Tuple[int,int]] = []

    with torch.no_grad():
        for (x, y) in tqdm(coords, desc=f"Patches({slide_id})"):
            rgba = np.array(slide.read_region((x, y), 0, (patch, patch)))  # RGBA
            rgb = rgba[..., :3].astype(np.uint8)

            if tissue_ratio(rgb) < tissue_thr:
                continue

            batch_imgs.append(rgb)
            batch_coords.append((x, y))

            if len(batch_imgs) >= batch_size:
                _flush_batch(batch_imgs, batch_coords, model, device, save_h5,
                             h5w_box, h5_path, (patch, patch), downsample)

        if len(batch_imgs) > 0:
            _flush_batch(batch_imgs, batch_coords, model, device, save_h5,
                         h5w_box, h5_path, (patch, patch), downsample)

    slide.close()
    if save_h5 and h5w_box["obj"] is not None:
        h5w_box["obj"].close()

    # 导出 Zarr（密集网格 + MIL）
    if export_zarr:
        if not (save_h5 and os.path.isfile(h5_path)):
            print(f"[ERR] cannot export Zarr without H5: set save_h5=True for {slide_id}")
            return

        # —— 导出前：确保 H5 真有数据 ——
        with h5py.File(h5_path, "r") as _fchk:
            ok_layers = []
            for _k in ("c2","c3","c4","c5"):
                if _k in _fchk and "features" in _fchk[_k]:
                    if _fchk[_k]["features"].shape[0] > 0:
                        ok_layers.append(_k)
            if len(ok_layers) == 0:
                print(f"[SKIP] no features in H5 (empty after filtering): {os.path.basename(h5_path)}")
                try:
                    os.remove(h5_path)
                    print(f"[CLEAN] removed empty H5: {h5_path}")
                except Exception as _e:
                    print(f"[WARN] failed to remove empty H5: {h5_path} ({_e})")
                return

        export_dense_feature_grids_to_zarr._overwrite    = getattr(extract_one_slide, "_overwrite", False)
        export_dense_feature_grids_to_zarr._skip_if_done = getattr(extract_one_slide, "_skip_if_done", True)
        export_dense_feature_grids_to_zarr(
            h5_path=h5_path,
            slide_path=slide_path,
            out_zarr_dir=zarr_dir,
            layers=("c2","c3","c4"),
            dtype="float16",
            chunk_hw=256,
            save_cover_mask=True,
            save_cls_head=True
        )

def _flush_batch(batch_imgs, batch_coords, model, device, save_h5, h5w_box, h5_path, input_size, downsample):
    # CPU: list -> tensor
    imgs_cpu = [torch.from_numpy(x).permute(2,0,1).contiguous() for x in batch_imgs]
    imgs = torch.stack(imgs_cpu, dim=0)  # uint8

    # GPU 归一化
    imgs = imgs.to(device, non_blocking=True, memory_format=torch.channels_last).float().div_(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
    imgs.sub_(mean).div_(std)

    use_cuda = torch.cuda.is_available() and device.startswith("cuda")
    with torch.amp.autocast("cuda", enabled=use_cuda):
        c2, c3, c4, c5 = model(imgs)

    def to_np(t: torch.Tensor) -> np.ndarray:
        return t.detach().float().cpu().numpy()

    feats = {
        "c2": to_np(c2),
        "c3": to_np(c3),
        "c4": to_np(c4),
        "c5": to_np(c5),
    }
    coords_xy = np.array(batch_coords, dtype="int32")

    # —— 首次真正写入时再创建 H5Writer ——
    if save_h5:
        if h5w_box["obj"] is None:
            h5w_box["obj"] = H5FeatureWriter(h5_path, input_size, downsample)
            # 让 H5 finalize 行为跟随 extract_one_slide 的覆盖策略
            h5w_box["obj"]._overwrite_ = getattr(extract_one_slide, "_overwrite", False)
        h5w_box["obj"].append_batch(coords_xy, feats)

    batch_imgs.clear()
    batch_coords.clear()

    del imgs, c2, c3, c4, c5
    if use_cuda:
        torch.cuda.empty_cache()

# =========================
# 并行辅助
# =========================
def _parse_gpus(gpus_str: str):
    if not gpus_str:
        return [f"{i}" for i in range(torch.cuda.device_count())]
    ids = []
    for tok in gpus_str.split(","):
        tok = tok.strip()
        if tok == "":
            continue
        assert tok.isdigit(), f"非法 GPU id: {tok}"
        ids.append(tok)
    if not ids:
        ids = [f"{i}" for i in range(torch.cuda.device_count())]
    return ids

def _worker_one_slide(
    slide_path: str,
    out_dir: str,
    device: str,
    args_dict: dict
):
    """独立进程内执行：为该进程设置 CUDA 设备，实例化模型，运行 extract_one_slide。"""
    try:
        os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

        if torch.cuda.is_available() and device.startswith("cuda"):
            gid = int(device.split(":")[-1])
            torch.cuda.set_device(gid)
            torch.backends.cudnn.benchmark = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        model = ResNet50C245(pretrained=not args_dict["no_pretrain"])

        extract_one_slide._overwrite     = args_dict["overwrite"]
        extract_one_slide._skip_if_done  = args_dict["skip_if_done"]

        export_dense_feature_grids_to_zarr._overwrite    = args_dict["overwrite"]
        export_dense_feature_grids_to_zarr._skip_if_done = args_dict["skip_if_done"]

        extract_one_slide(
            slide_path=slide_path,
            out_dir=out_dir,
            model=model,
            device=device,
            patch=args_dict["patch"],
            stride=args_dict["stride"],
            batch_size=args_dict["batch_size"],
            tissue_thr=args_dict["tissue_thr"],
            save_h5=not args_dict["no_h5"],
            export_zarr=not args_dict["no_zarr"],
        )
        return (slide_path, "OK", "")
    except Exception as e:
        return (slide_path, "FAIL", str(e))

# =========================
# CLI / main
# =========================
def main():
    try:
        mp.set_start_method("spawn", force=True)
    except Exception:
        pass

    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

    ap = argparse.ArgumentParser()
    ap.add_argument("--slide_dir", type=str, required=True, help="目录下的所有 .tif/.tiff/.svs 将被处理")
    ap.add_argument("--out_dir",   type=str, required=True)
    ap.add_argument("--device",    type=str, default="cuda")
    ap.add_argument("--patch",     type=int, default=224)
    ap.add_argument("--stride",    type=int, default=224)
    ap.add_argument("--batch_size",type=int, default=64)
    ap.add_argument("--tissue_thr",type=float, default=0.05, help="跳过空白阈值（0~1），越大越严格")
    ap.add_argument("--no_h5",     action="store_true", help="不保存 H5（只导出 Zarr）")
    ap.add_argument("--no_zarr",   action="store_true", help="不导出 Zarr（仅 H5）")
    ap.add_argument("--no_pretrain", action="store_true", help="ResNet50 不加载 ImageNet 预训练")

    ap.add_argument("--gpus", type=str, default="", help="逗号分隔的 GPU id 列表，如 '0,1,2'；留空则使用全部可用 GPU")
    ap.add_argument("--num_workers", type=int, default=0, help="并发进程数；默认=GPU 数；CPU 或仅导出 H5 时可手动加大")

    ap.add_argument("--overwrite", action="store_true", help="强制重算并覆盖已存在的产物")
    ap.add_argument("--skip_if_done", action="store_true", default=True, help="若目标产物已存在则跳过")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    slides = sorted([p for p in glob.glob(os.path.join(args.slide_dir, "*")) if is_wsi_file(p)])
    if not slides:
        print(f"[ERR] no WSI files found in: {args.slide_dir}")
        sys.exit(1)

    gpu_ids = _parse_gpus(args.gpus) if torch.cuda.is_available() else []
    if (args.device.startswith("cuda") and not gpu_ids):
        print("[WARN] 未检测到可用 GPU，回退到 CPU。")
    if args.num_workers <= 0:
        args.num_workers = max(1, len(gpu_ids) if gpu_ids else max(1, (os.cpu_count() or 2) // 2))

    args_dict = dict(
        patch=args.patch,
        stride=args.stride,
        batch_size=args.batch_size,
        tissue_thr=args.tissue_thr,
        no_h5=args.no_h5,
        no_zarr=args.no_zarr,
        no_pretrain=args.no_pretrain,
        overwrite=args.overwrite,
        skip_if_done=args.skip_if_done,
    )

    print(f"[INFO] slides={len(slides)}, gpus={gpu_ids if gpu_ids else 'CPU'}, workers={args.num_workers}")
    ok, fail = 0, 0
    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        futures = []
        for idx, sp in enumerate(slides):
            if gpu_ids and args.device.startswith("cuda"):
                gid = gpu_ids[idx % len(gpu_ids)]
                device = f"cuda:{gid}"
            else:
                device = "cpu"
            futures.append(ex.submit(_worker_one_slide, sp, args.out_dir, device, args_dict))

        for fut in as_completed(futures):
            sp, status, msg = fut.result()
            if status == "OK":
                ok += 1
                print(f"[OK] {os.path.basename(sp)}")
            else:
                fail += 1
                print(f"[FAIL] {os.path.basename(sp)} :: {msg}")

    print(f"[DONE] success={ok}, fail={fail}")

if __name__ == "__main__":
    main()
