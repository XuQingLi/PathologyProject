# wsi_stream_eval_png.py  —— ROI + 覆盖区评估 + 一键自检开关
import os, json, math, time
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

from openslide import OpenSlide, OpenSlideError
import tifffile as tiff

# ---------- 工具 ----------
def _json_load_maybe(x):
    if isinstance(x, (bytes, str)):
        try: return json.loads(x)
        except Exception: return x
    return x

def _get_meta(h5f, layer):
    assert layer in h5f and "features" in h5f[layer], f"'{layer}/features' not found"
    feats = h5f[f"{layer}/features"]   # (N, C, Hf, Wf)
    coords = h5f["coords"][:]          # (N, 2)  (x, y)
    meta = dict(h5f["meta"].attrs.items()) if "meta" in h5f else {}
    meta = {k: _json_load_maybe(v) for k, v in meta.items()}

    if "input_size" in meta:
        H_in, W_in = meta["input_size"]
    else:
        H_in = W_in = 256

    if "downsample" in meta and isinstance(meta["downsample"], dict) and layer in meta["downsample"]:
        v = meta["downsample"][layer]
        if isinstance(v, (list, tuple)) and len(v) == 2:
            ds_h, ds_w = int(v[0]), int(v[1])
        else:
            ds_h = ds_w = int(v)
    else:
        ds_h = ds_w = max(1, H_in // feats.shape[2])

    N, C, Hf, Wf = feats.shape
    assert H_in // ds_h == Hf and W_in // ds_w == Wf, \
        f"meta/input_size 与 {layer} 尺度不匹配: input={(H_in,W_in)}, feat={(Hf,Wf)}, down={(ds_h,ds_w)}"
    return feats, coords, (H_in, W_in), (ds_h, ds_w)

def _tile_to_binary_mask(arr: np.ndarray) -> np.ndarray:
    """
    简单二值化：像素值 > 0 视为前景(1)，否则背景(0)。
    - 兼容 L / LA / RGB / RGBA / palette 等；对彩色掩码按“任一通道非零”为前景。
    - 输入是当前 tile 的 ndarray（来自 OpenSlide.read_region 或 tifffile 切片）。
    """
    if arr.ndim == 2:  # 灰度/索引图
        return (arr > 0).astype(np.uint8)
    if arr.ndim == 3:
        # 若 RGBA，忽略透明度；对彩色掩码，任一通道非零即视为前景
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        return (arr.max(axis=2) > 0).astype(np.uint8)
    # 其他形状：当作全背景
    return np.zeros(arr.shape[:2], dtype=np.uint8)


def _clamp(v, lo, hi): return max(lo, min(int(v), hi))

# ---------- 主函数 ----------
@torch.inference_mode()
def stream_decode_eval_png(
    h5_path: str,
    slide_path: str,
    ground_truth_path: str,
    decoder,                      # UNet-style 解码器（.eval().to(device)）
    layer: str = "c2",
    coords_order: str = "xy",     # 'xy' 或 'yx'
    tile_px: int = 2048,          # 合理的 tile 尺寸；卡的话再降到 2048
    tile_overlap_px: int = 0,     # 评估建议设 0
    upsample_mode: str = "bilinear",
    device: str = "cuda",

    out_mask_png: str = "/mnt/gemlab_data_3/LXQ/pred_mask_vis.png",
    out_overlay_png: str = "/mnt/gemlab_data_3/LXQ/pred_overlay_vis.png",
    vis_down: int = 16,
    overlay_alpha: float = 0.35,
    gt_threshold: int = 128,

    # —— 调试开关：'none' | 'coverage' | 'constant1'
    # 'coverage' 只画覆盖区域；'constant1' 在覆盖区恒预测1
    debug_mode: str = "none",
):
    assert os.path.isfile(h5_path)
    assert os.path.isfile(slide_path)
    assert os.path.isfile(ground_truth_path)
    assert coords_order in ("xy", "yx")
    assert upsample_mode in ("nearest", "bilinear")
    assert debug_mode in ("none", "coverage", "constant1")

    torch.backends.cudnn.benchmark = True

    # --- 打开 WSI & GT 一次 ---
    slide = OpenSlide(slide_path)
    W_wsi, H_wsi = slide.level_dimensions[0]

    gt_slide = None
    gt_img = None
    try:
        gt_slide = OpenSlide(ground_truth_path)
        W_gt, H_gt = gt_slide.level_dimensions[0]
    except OpenSlideError:
        gt_img = tiff.imread(ground_truth_path)
        if gt_img.ndim == 3 and gt_img.shape[2] in (3,4):
            H_gt, W_gt = gt_img.shape[:2]
        elif gt_img.ndim == 2:
            H_gt, W_gt = gt_img.shape
        else:
            raise RuntimeError("Unsupported GT tiff shape.")
    assert (W_gt, H_gt) == (W_wsi, H_wsi), f"GT size {W_gt,H_gt} != WSI size {W_wsi,H_wsi}"

    # --- 打开 H5 一次 ---
    h5f = h5py.File(h5_path, "r")
    feats_ds, coords_all, (H_in, W_in), (ds_h, ds_w) = _get_meta(h5f, layer)
    if coords_order == "yx":
        coords_all = coords_all[:, [1, 0]]
    xs = coords_all[:, 0].astype(np.int64)
    ys = coords_all[:, 1].astype(np.int64)
    N, C, Hf, Wf = feats_ds.shape

    # --- 仅在 coords 外接框（ROI）内滑窗，显著提速 ---
    if xs.size == 0:
        raise RuntimeError("No coords found in H5.")
    x_min = _clamp(xs.min(), 0, W_wsi-1)
    y_min = _clamp(ys.min(), 0, H_wsi-1)
    x_max = _clamp(xs.max() + W_in, 1, W_wsi)  # 右边界取到 patch 右侧
    y_max = _clamp(ys.max() + H_in, 1, H_wsi)

    # row/col 范围（按 tile 步进）
    def _range_with_step(start, end, step):
        start = (start // step) * step
        for v in range(start, end, step):
            yield v

    # --- 可视化画布（缩略图尺度） ---
    vis_w = (W_wsi + vis_down - 1) // vis_down
    vis_h = (H_wsi + vis_down - 1) // vis_down
    vis_mask = np.zeros((vis_h, vis_w), dtype=np.uint8)   # 累计票
    vis_cnt  = np.zeros((vis_h, vis_w), dtype=np.uint16)  # 计数
    bg_thumb = slide.get_thumbnail((vis_w, vis_h)).convert("RGB")

    # --- 指标累计（仅覆盖区域） ---
    tp = np.int64(0); fp = np.int64(0); fn = np.int64(0)

    step = tile_px - tile_overlap_px
    assert step > 0, "tile_px must be > tile_overlap_px"
    half_ov = tile_overlap_px // 2

    print("WSI size:", (W_wsi, H_wsi))
    print("H_in/W_in:", (H_in, W_in), "ds:", (ds_h, ds_w))
    print("coords X[min,max]:", (int(xs.min()), int(xs.max())),
          "Y[min,max]:", (int(ys.min()), int(ys.max())))
    print("ROI x:[%d,%d) y:[%d,%d)" % (x_min, x_max, y_min, y_max))
    tiles_with_patches = 0
    total_patches_used = 0

    rows = list(_range_with_step(y_min, y_max, step))
    for y0 in tqdm(rows, desc="Rows(ROI)"):
        y1 = min(H_wsi, y0 + tile_px)
        tile_h_px = y1 - y0

        cols = list(_range_with_step(x_min, x_max, step))
        for x0 in tqdm(cols, desc=f"Cols(y={y0}-{y1})", leave=False):
            x1 = min(W_wsi, x0 + tile_px)
            tile_w_px = x1 - x0

            tile_h_f = math.ceil(tile_h_px / ds_h)
            tile_w_f = math.ceil(tile_w_px / ds_w)

            # 选出覆盖该 tile 的 patch
            sel_mask = (xs < x1) & (xs + W_in > x0) & (ys < y1) & (ys + H_in > y0)
            idx = np.nonzero(sel_mask)[0]
            tile_has_feats = (idx.size > 0)

            if not tile_has_feats:
                continue
            tiles_with_patches += 1
            total_patches_used += int(idx.size)

            # 局部特征累加
            local_sum = np.zeros((C, tile_h_f, tile_w_f), dtype=np.float32)
            local_cnt = np.zeros((1, tile_h_f, tile_w_f), dtype=np.uint32)

            bs = min(2048, max(256, idx.size))
            for s in range(0, idx.size, bs):
                ids = idx[s:s+bs]
                feats_np = feats_ds[ids].astype(np.float32, copy=False)
                feats_np = np.nan_to_num(feats_np, nan=0.0, posinf=0.0, neginf=0.0)  # 清洗

                xs_sel = xs[ids]; ys_sel = ys[ids]
                gx0 = (xs_sel // ds_w).astype(np.int64) - (x0 // ds_w)
                gy0 = (ys_sel // ds_h).astype(np.int64) - (y0 // ds_h)

                M = feats_np.shape[0]
                for i in range(M):
                    y0f, x0f = int(gy0[i]), int(gx0[i])
                    y1f, x1f = y0f + Hf, x0f + Wf
                    yy0, yy1 = max(0, y0f), min(tile_h_f, y1f)
                    xx0, xx1 = max(0, x0f), min(tile_w_f, x1f)
                    ih, iw = yy1 - yy0, xx1 - xx0
                    if ih <= 0 or iw <= 0: 
                        continue
                    ph0 = yy0 - y0f; ph1 = y1f - yy1
                    pw0 = xx0 - x0f; pw1 = x1f - xx1
                    patch = feats_np[i, :, ph0:Hf - ph1, pw0:Wf - pw1]
                    if patch.size == 0: 
                        continue
                    ih2 = min(ih, patch.shape[1]); iw2 = min(iw, patch.shape[2])
                    if ih2 <= 0 or iw2 <= 0: 
                        continue
                    local_sum[:, yy0:yy0+ih2, xx0:xx0+iw2] += patch[:, :ih2, :iw2]
                    local_cnt[:, yy0:yy0+ih2, xx0:xx0+iw2] += 1

            if local_cnt.max() == 0:
                continue

            # 覆盖区域（特征尺度→像素尺度），并截到当前 tile 的像素大小
            cov_px = np.repeat(np.repeat((local_cnt > 0).astype(np.uint8), ds_h, axis=1), ds_w, axis=2)[0]
            cov_px = cov_px[:tile_h_px, :tile_w_px]

            # 中心裁剪（如有重叠）
            cl = half_ov if x0 > 0 else 0
            ct = half_ov if y0 > 0 else 0
            cr = half_ov if x1 < W_wsi else 0
            cb = half_ov if y1 < H_wsi else 0
            if (cl+cr) >= cov_px.shape[1] or (ct+cb) >= cov_px.shape[0]:
                continue
            cov_c = cov_px[ct:cov_px.shape[0]-cb, cl:cov_px.shape[1]-cr]
            x0w, x1w = x0 + cl, x1 - cr
            y0w, y1w = y0 + ct, y1 - cb
            if cov_c.max() == 0:
                continue

            # --- 三种模式：正常推理 / 仅覆盖可视化 / 覆盖恒为1 --- #
            if debug_mode == "coverage":
                pred_bin = (cov_c > 0).astype(np.uint8)
            elif debug_mode == "constant1":
                pred_bin = np.ones_like(cov_c, dtype=np.uint8)
            else:
                # 正常推理
                local_feat = np.divide(local_sum, np.maximum(local_cnt, 1), dtype=np.float32)  # (C,hf,wf)
                tin = torch.from_numpy(local_feat).unsqueeze(0).to(device, non_blocking=True)  # [1,C,hf,wf]
                with autocast(enabled=(device.startswith("cuda"))):
                    pred_f = decoder(tin)                                        # [1,1,hf',wf']
                    pred_px = F.interpolate(
                        pred_f, size=(cov_px.shape[0]+ct+cb, cov_px.shape[1]+cl+cr),
                        mode=upsample_mode, align_corners=False if upsample_mode=="bilinear" else None
                    )
                pred_px = pred_px.squeeze(0).float().cpu().numpy()[0]  # [Ht,Wt]
                pred_px = pred_px[ct:pred_px.shape[0]-cb, cl:pred_px.shape[1]-cr]  # 与 cov_c 对齐
                logits = np.clip(pred_px, -40.0, 40.0)
                prob = 1.0 / (1.0 + np.exp(-logits))
                pred_bin = (prob >= 0.5).astype(np.uint8)

            # 读取同区域 GT 并二值化
            w_tile = x1w - x0w; h_tile = y1w - y0w
            if gt_slide is not None:
                gt_rgba = np.array(gt_slide.read_region((x0w, y0w), 0, (w_tile, h_tile)))
                gt_rgb = gt_rgba[..., :3]
                gt_bin = _tile_to_binary_mask(gt_rgb)
            else:
                if gt_img.ndim == 2:
                    gt_crop = gt_img[y0w:y1w, x0w:x1w]
                else:
                    gt_crop = gt_img[y0w:y1w, x0w:x1w, ...]
                gt_bin = _tile_to_binary_mask(gt_crop)

            if gt_bin.shape != pred_bin.shape:
                gt_bin = np.array(Image.fromarray(gt_bin).resize(pred_bin.shape[::-1], Image.NEAREST), dtype=np.uint8)

            # —— 仅在覆盖区域统计 —— #
            region = cov_c.astype(bool)
            if region.any():
                p = pred_bin[region] > 0
                g = gt_bin[region] > 0
                tp += np.count_nonzero(p & g)
                fp += np.count_nonzero(p & (~g))
                fn += np.count_nonzero((~p) & g)

            # 可视化：仅覆盖区域写票
            if vis_down > 1:
                vh = max(1, (y1w - y0w) // vis_down)
                vw = max(1, (x1w - x0w) // vis_down)
                vis_pred = np.array(Image.fromarray((pred_bin*255).astype(np.uint8)).resize((vw, vh), Image.NEAREST))
                vis_cov  = np.array(Image.fromarray((cov_c*255).astype(np.uint8)).resize((vw, vh), Image.NEAREST)) > 127
                ys0, xs0 = y0w // vis_down, x0w // vis_down
                ys1, xs1 = ys0 + vh, xs0 + vw
                vis_mask[ys0:ys1, xs0:xs1] += ((vis_pred > 127) & vis_cov).astype(np.uint8)
                vis_cnt [ys0:ys1, xs0:xs1] += vis_cov.astype(np.uint8)

    # 指标
    dice = (2.0 * tp) / max(1, (2*tp + fp + fn))
    iou  = (tp) / max(1, (tp + fp + fn))

    # 可视化 PNG（对投票做阈值）
    with np.errstate(divide='ignore', invalid='ignore'):
        vis_prob = np.true_divide(vis_mask, np.maximum(vis_cnt, 1))
        vis_prob[~np.isfinite(vis_prob)] = 0.0
    vis_bin = (vis_prob >= 0.5).astype(np.uint8) * 255
    Image.fromarray(vis_bin).save(out_mask_png)

    # 叠加显示
    overlay = np.array(bg_thumb).astype(np.float32)
    mask_rgb = np.zeros_like(overlay)
    mask_rgb[..., 0] = vis_bin
    blended = (overlay*(1-overlay_alpha) + mask_rgb*overlay_alpha).clip(0,255).astype(np.uint8)
    Image.fromarray(blended).save(out_overlay_png)

    # 关闭句柄
    h5f.close()
    slide.close()
    if gt_slide is not None: gt_slide.close()

    print(f"[OK] ROI tiles with patches: {tiles_with_patches}, avg patches/tile: {total_patches_used/max(1,tiles_with_patches):.1f}")
    print(f"[OK] Saved mask PNG:    {out_mask_png}")
    print(f"[OK] Saved overlay PNG: {out_overlay_png}")
    print(f"[Metrics] Dice: {dice:.6f}, IoU: {iou:.6f}")
    return float(dice), float(iou)
