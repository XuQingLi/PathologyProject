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


# ========= 新增：按层读取 meta 的轻便封装 =========
def _get_meta_for_layer(h5f, layer):
    assert layer in h5f and "features" in h5f[layer], f"'{layer}/features' not found"
    feats = h5f[f"{layer}/features"]   # (N, C, Hf, Wf)
    meta  = dict(h5f["meta"].attrs.items()) if "meta" in h5f else {}
    meta  = {k: _json_load_maybe(v) for k, v in meta.items()}
    H_in, W_in = (meta.get("input_size") or (256, 256))
    v = meta.get("downsample", {}).get(layer, None)
    if v is None:
        ds_h = ds_w = max(1, H_in // feats.shape[2])
    else:
        if isinstance(v, (list, tuple)) and len(v) == 2:
            ds_h, ds_w = int(v[0]), int(v[1])
        else:
            ds_h = ds_w = int(v)
    N, C, Hf, Wf = feats.shape
    assert H_in // ds_h == Hf and W_in // ds_w == Wf, \
        f"[{layer}] input_size 与特征尺寸不匹配: input={(H_in,W_in)}, feat={(Hf,Wf)}, down={(ds_h,ds_w)}"
    return feats, (H_in, W_in), (ds_h, ds_w), (N, C, Hf, Wf)

# ========= 新增：把若干 patch 粘贴成某层的局部网格（保持与你现有写法一致）=========
def _paste_patches_to_tile(feats_ds, ids, xs, ys, x0, y0, tile_h_px, tile_w_px, Hf, Wf, ds_h, ds_w,
                           paste_bs=2048):
    tile_h_f = math.ceil(tile_h_px / ds_h)
    tile_w_f = math.ceil(tile_w_px / ds_w)
    local_sum = np.zeros((feats_ds.shape[1], tile_h_f, tile_w_f), dtype=np.float32)
    local_cnt = np.zeros((1,               tile_h_f, tile_w_f), dtype=np.uint32)

    for s in range(0, ids.size, min(paste_bs, max(256, ids.size))):
        sel = ids[s:s+min(paste_bs, ids.size)]
        f   = feats_ds[sel].astype(np.float32, copy=False)
        f   = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)

        gx0 = (xs[sel] // ds_w).astype(np.int64) - (x0 // ds_w)
        gy0 = (ys[sel] // ds_h).astype(np.int64) - (y0 // ds_h)
        M   = f.shape[0]
        for i in range(M):
            y0f, x0f = int(gy0[i]), int(gx0[i])
            y1f, x1f = y0f + Hf, x0f + Wf
            yy0, yy1 = max(0, y0f), min(tile_h_f, y1f)
            xx0, xx1 = max(0, x0f), min(tile_w_f, x1f)
            ih, iw   = yy1 - yy0, xx1 - xx0
            if ih <= 0 or iw <= 0: 
                continue
            ph0, ph1 = yy0 - y0f, y1f - yy1
            pw0, pw1 = xx0 - x0f, x1f - xx1
            patch = f[i, :, ph0:Hf-ph1, pw0:Wf-pw1]
            if patch.size == 0: 
                continue
            ih2 = min(ih, patch.shape[1]); iw2 = min(iw, patch.shape[2])
            if ih2 <= 0 or iw2 <= 0:
                continue
            local_sum[:, yy0:yy0+ih2, xx0:xx0+iw2] += patch[:, :ih2, :iw2]
            local_cnt[:, yy0:yy0+ih2, xx0:xx0+iw2] += 1

    return local_sum, local_cnt  # (C, hf, wf), (1, hf, wf)


@torch.inference_mode()
def stream_decode_eval_png(
    h5_path: str,
    slide_path: str,
    ground_truth_path: str,
    decoder,                      # UNet-style 解码器（.eval().to(device)）
    coords_order: str = "xy",     # 'xy' 或 'yx'
    tile_px: int = 4096,          # 合理的 tile 尺寸
    tile_overlap_px: int = 0,     # 评估建议设 0
    upsample_mode: str = "bilinear",
    device: str = "cuda",

    out_mask_png: str = "/mnt/gemlab_data_3/LXQ/pred_mask_vis.png",
    out_overlay_png: str = "/mnt/gemlab_data_3/LXQ/pred_overlay_vis.png",
    vis_down: int = 16,
    overlay_alpha: float = 0.35,
):
    assert os.path.isfile(h5_path)
    assert os.path.isfile(slide_path)
    assert os.path.isfile(ground_truth_path)
    assert coords_order in ("xy", "yx")
    assert upsample_mode in ("nearest", "bilinear")

    torch.backends.cudnn.benchmark = True

    # --- 打开 WSI & GT ---
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

    # --- 打开 H5，一次性读取三层的 meta 与 dataset 句柄 ---
    h5f = h5py.File(h5_path, "r")
    if coords_order == "yx":
        coords_all = h5f["coords"][:, [1, 0]]
    else:
        coords_all = h5f["coords"][:]
    xs = coords_all[:, 0].astype(np.int64)
    ys = coords_all[:, 1].astype(np.int64)
    assert xs.size > 0, "No coords found in H5."

    # 三层 meta（c2 作为基准网格）
    feats_c2, (Hin2,Win2), (ds2h,ds2w), (_N2,C2,Hf2,Wf2) = _get_meta_for_layer(h5f, "c2")
    feats_c3, (Hin3,Win3), (ds3h,ds3w), (_N3,C3,Hf3,Wf3) = _get_meta_for_layer(h5f, "c3")
    feats_c4, (Hin4,Win4), (ds4h,ds4w), (_N4,C4,Hf4,Wf4) = _get_meta_for_layer(h5f, "c4")

    # 用 c2 的 patch 尺寸作粗筛（通常三层 input_size 一致）
    H_in, W_in = Hin2, Win2

    # --- 仅在 coords 的外接 ROI 内滑窗（显著提速） ---
    x_min = max(0, int(xs.min()))
    y_min = max(0, int(ys.min()))
    x_max = min(W_wsi, int(xs.max() + W_in))
    y_max = min(H_wsi, int(ys.max() + H_in))

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

    # --- tile 网格 ---
    step = tile_px - tile_overlap_px
    assert step > 0, "tile_px must be > tile_overlap_px"
    half_ov = tile_overlap_px // 2

    print("WSI size:", (W_wsi, H_wsi))
    print("H_in/W_in:", (H_in, W_in), "ds(c2,c3,c4):", (ds2h,ds2w, ds3h,ds3w, ds4h,ds4w))
    print("coords X[min,max]:", (int(xs.min()), int(xs.max())),
          "Y[min,max]:", (int(ys.min()), int(ys.max())))
    print("ROI x:[%d,%d) y:[%d,%d)" % (x_min, x_max, y_min, y_max))

    rows = list(_range_with_step(y_min, y_max, step))
    for y0 in tqdm(rows, desc="Rows(ROI)"):
        y1 = min(H_wsi, y0 + tile_px)
        tile_h_px = y1 - y0

        cols = list(_range_with_step(x_min, x_max, step))
        for x0 in tqdm(cols, desc=f"Cols(y={y0}-{y1})", leave=False):
            x1 = min(W_wsi, x0 + tile_px)
            tile_w_px = x1 - x0

            # 先用 c2 的 patch 外接框做粗筛
            sel_mask = (xs < x1) & (xs + W_in > x0) & (ys < y1) & (ys + H_in > y0)
            idx = np.nonzero(sel_mask)[0]
            if idx.size == 0:
                continue  # 无覆盖的 tile 直接跳过

            # —— 三层分别贴局部网格 —— #
            sum2, cnt2 = _paste_patches_to_tile(feats_c2, idx, xs, ys, x0, y0,
                                                tile_h_px, tile_w_px, Hf2, Wf2, ds2h, ds2w,
                                                paste_bs=2048)
            sum3, cnt3 = _paste_patches_to_tile(feats_c3, idx, xs, ys, x0, y0,
                                                tile_h_px, tile_w_px, Hf3, Wf3, ds3h, ds3w,
                                                paste_bs=2048)
            sum4, cnt4 = _paste_patches_to_tile(feats_c4, idx, xs, ys, x0, y0,
                                                tile_h_px, tile_w_px, Hf4, Wf4, ds4h, ds4w,
                                                paste_bs=2048)

            if (cnt2.max()==0) and (cnt3.max()==0) and (cnt4.max()==0):
                continue

            # 覆盖区（在 c2 网格上对齐，取 union 更稳）
            # 先把各层计数上采样/对齐到 c2 网格，或分别上采样到像素再 union
            # 这里更直接：都上采样到像素尺度后 union
            cov2_px = np.repeat(np.repeat((cnt2>0).astype(np.uint8), ds2h, axis=1), ds2w, axis=2)[0]
            cov3_px = np.repeat(np.repeat((cnt3>0).astype(np.uint8), ds3h, axis=1), ds3w, axis=2)[0]
            cov4_px = np.repeat(np.repeat((cnt4>0).astype(np.uint8), ds4h, axis=1), ds4w, axis=2)[0]
            cov2_px = cov2_px[:tile_h_px, :tile_w_px]
            cov3_px = cov3_px[:tile_h_px, :tile_w_px]
            cov4_px = cov4_px[:tile_h_px, :tile_w_px]
            cov_px  = (cov2_px | cov3_px | cov4_px).astype(np.uint8)
            if cov_px.max() == 0:
                continue

            # 计算各层平均特征
            feat2 = np.divide(sum2, np.maximum(cnt2, 1), dtype=np.float32)  # (C2,hf2,wf2)
            feat3 = np.divide(sum3, np.maximum(cnt3, 1), dtype=np.float32)  # (C3,hf3,wf3)
            feat4 = np.divide(sum4, np.maximum(cnt4, 1), dtype=np.float32)  # (C4,hf4,wf4)

            # 统一上采样到 c2 网格尺寸，并 concat
            hf2 = math.ceil(tile_h_px / ds2h)
            wf2 = math.ceil(tile_w_px / ds2w)

            def _torch_resize(np_feat, out_hw):
                if np_feat.ndim != 3:
                    return np_feat
                t = torch.from_numpy(np_feat[None])  # [1,C,H,W]
                t = F.interpolate(t, size=out_hw, mode="bilinear", align_corners=False)
                return t.squeeze(0).cpu().numpy()

            feat3_up = _torch_resize(feat3, (hf2, wf2))
            feat4_up = _torch_resize(feat4, (hf2, wf2))
            feat_cat = np.concatenate([feat2, feat3_up, feat4_up], axis=0)  # [Csum,hf2,wf2]

            # 中心裁剪（避免重叠重复统计）
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

            # ===== 分块解码开始 =====

            # 1) 目标：得到 c2 网格尺寸的 logits 全图 [hf2, wf2]
            CHUNK_HF = min(768, hf2)     # 你也可以 512/640/896，越小越省显存
            CHUNK_WF = min(768, wf2)
            OVERLAP  = 32                # 重叠，避免块边界伪影

            logits_grid = np.zeros((hf2, wf2), dtype=np.float32)
            weight_grid = np.zeros((hf2, wf2), dtype=np.float32)  # 用简单的“平均融合”

            # 2) 分块遍历
            y = 0
            while y < hf2:
                y1 = min(hf2, y + CHUNK_HF)
                # 为了有重叠，下一块从 y + CHUNK_HF - OVERLAP 开始
                y_next = y1 - OVERLAP if (y1 < hf2) else y1

                x = 0
                while x < wf2:
                    x1 = min(wf2, x + CHUNK_WF)
                    x_next = x1 - OVERLAP if (x1 < wf2) else x1

                    # 取该块特征
                    feat_patch = feat_cat[:, y:y1, x:x1]                            # [Csum, h, w]
                    tin = torch.from_numpy(feat_patch).unsqueeze(0).to(device)      # [1,Csum,h,w]

                    with torch.amp.autocast("cuda", enabled=(device.startswith("cuda"))):
                        pred_chunk = decoder(tin)    # [1,1,h,w] —— 解码器输出对齐 c2 网格
                    pred_chunk = pred_chunk.squeeze(0).squeeze(0).float().cpu().numpy()  # [h, w]

                    # 累计到全局 logits 网格
                    logits_grid[y:y1, x:x1] += pred_chunk
                    weight_grid[y:y1, x:x1] += 1.0

                    # 移动到下一个块
                    x = x_next
                    # 释放块内临时张量（更稳）
                    del tin; 
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                y = y_next

            # 3) 求平均，得到完整 c2 网格 logits
            weight_grid = np.maximum(weight_grid, 1e-6)
            logits_c2 = (logits_grid / weight_grid).astype(np.float32)  # [hf2, wf2]

            # 4) 上采样到像素尺度，并做中心裁剪
            pred_px_full = torch.from_numpy(logits_c2)[None,None].to(device)
            pred_px_full = F.interpolate(
                pred_px_full, size=(tile_h_px, tile_w_px),
                mode=upsample_mode, align_corners=False if upsample_mode=="bilinear" else None
            ).squeeze(0).squeeze(0).float().cpu().numpy()  # [Ht, Wt]

            pred_px = pred_px_full[ct:pred_px_full.shape[0]-cb, cl:pred_px_full.shape[1]-cr]

            # 5) logits→prob→二值
            logits = np.clip(pred_px, -40.0, 40.0)
            prob   = 1.0 / (1.0 + np.exp(-logits))
            pred_bin = (prob >= 0.5).astype(np.uint8)

            # ===== 分块解码结束 =====


            # logits→prob→二值
            logits = np.clip(pred_px, -40.0, 40.0)
            prob   = 1.0 / (1.0 + np.exp(-logits))
            pred_bin = (prob >= 0.5).astype(np.uint8)

            # 读取同区域 GT，并二值化（>0 为前景）
            w_tile = x1w - x0w; h_tile = y1w - y0w
            if gt_slide is not None:
                gt_rgba = np.array(gt_slide.read_region((x0w, y0w), 0, (w_tile, h_tile)))
                gt_bin  = _tile_to_binary_mask(gt_rgba)  # 你已改成 “>0 为前景” 的实现
            else:
                if gt_img.ndim == 2:
                    gt_crop = gt_img[y0w:y1w, x0w:x1w]
                else:
                    gt_crop = gt_img[y0w:y1w, x0w:x1w, ...]
                gt_bin  = _tile_to_binary_mask(gt_crop)

            if gt_bin.shape != pred_bin.shape:
                gt_bin = np.array(Image.fromarray(gt_bin).resize(pred_bin.shape[::-1], Image.NEAREST), dtype=np.uint8)

            # —— 仅在覆盖区域累计指标 —— #
            region = cov_c.astype(bool)
            if region.any():
                p = pred_bin[region] > 0
                g = gt_bin[region] > 0
                tp += np.count_nonzero(p & g)
                fp += np.count_nonzero(p & (~g))
                fn += np.count_nonzero((~p) & g)

            # —— 可视化：仅覆盖区域写票 —— #
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

    print(f"[OK] Saved mask PNG:    {out_mask_png}")
    print(f"[OK] Saved overlay PNG: {out_overlay_png}")
    print(f"[Metrics] Dice: {dice:.6f}, IoU: {iou:.6f}")
    return float(dice), float(iou)
