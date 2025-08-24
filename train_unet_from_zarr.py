# -*- coding: utf-8 -*-
import os, math, argparse, json
import numpy as np
from PIL import Image
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import zarr
from numcodecs import Blosc
import math
import tifffile as tiff

# ---------- 工具 ----------
def set_seed(seed: int = 2025):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def dice_coef(pred: torch.Tensor, target: torch.Tensor, eps=1e-6) -> torch.Tensor:
    # pred/target: [B,1,H,W] binary(0/1)
    inter = (pred * target).sum(dim=(2,3))
    denom = pred.sum(dim=(2,3)) + target.sum(dim=(2,3)) + eps
    return (2.0 * inter / denom).mean()

def iou_coef(pred: torch.Tensor, target: torch.Tensor, eps=1e-6) -> torch.Tensor:
    inter = (pred * target).sum(dim=(2,3))
    union = (pred + target - pred * target).sum(dim=(2,3)) + eps
    return (inter / union).mean()

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.w = bce_weight
    def forward(self, logits, target):
        # logits: [B,1,H,W], target: [B,1,H,W] (0/1)
        bce = self.bce(logits, target)
        prob = torch.sigmoid(logits)
        inter = (prob * target).sum(dim=(2,3))
        denom = prob.sum(dim=(2,3)) + target.sum(dim=(2,3)) + 1e-6
        dice = 1.0 - (2.0 * inter / denom).mean()
        return self.w * bce + (1.0 - self.w) * dice

def read_mask_resized_to_c2(gt_path: str, target_size: Tuple[int,int]) -> np.ndarray:
    """
    快速读取并下采样到 c2 尺度 (W2,H2)，返回 [H2,W2] 的 {0,1} np.uint8。
    关键：优先选用金字塔低层，避免读取最高分辨率整幅图。
    """
    W2, H2 = target_size
    import math
    try:
        with tiff.TiffFile(gt_path) as tf:
            # 选择和 (H2,W2) 面积最接近的 level（优先同一 series 的金字塔）
            best = None
            best_diff = float("inf")
            best_series_idx, best_level_idx = 0, 0
            for si, s in enumerate(tf.series):
                levels = getattr(s, "levels", [s])
                for li, lv in enumerate(levels):
                    h, w = lv.shape[:2]
                    diff = abs(math.log((h * w + 1e-9) / (H2 * W2 + 1e-9)))
                    if diff < best_diff:
                        best_diff = diff
                        best = lv
                        best_series_idx, best_level_idx = si, li
            # 就地读取该 level（无需先读最大层）
            arr = tf.series[best_series_idx].asarray(level=best_level_idx)
    except Exception:
        # 兜底（非金字塔或读失败）：再用 tifffile.imread
        arr = tiff.imread(gt_path)

    if arr.ndim == 3:
        arr = arr[..., 0]
    # 二值化：全黑为背景，>0 为肿瘤
    arr = (arr > 0).astype(np.uint8) * 255

    # 调整到 (W2,H2)（PIL 的尺寸是 (W,H)）
    if (arr.shape[1], arr.shape[0]) != (W2, H2):
        from PIL import Image
        arr = np.array(Image.fromarray(arr).resize((W2, H2), Image.NEAREST))
    return (arr > 0).astype(np.uint8)



# ---------- 读取 Zarr ----------
def load_zarr_layers(zarr_dir: str):
    """
    返回：
      c2,c3,c4: zarr.Array 视图（懒加载），形状 [C,H,W]，dtype=float16/float32
      meta: dict，包含 downsample, wsi_size, input_size
    """
    g = zarr.open_group(zarr.DirectoryStore(zarr_dir), mode='r')
    for k in ['c2','c3','c4']:
        assert k in g, f"'{k}' missing in {zarr_dir}"
    c2 = g['c2']  # zarr.Array
    c3 = g['c3']
    c4 = g['c4']
    meta = {
        'downsample': g.attrs.get('downsample', {'c2':[4,4],'c3':[8,8],'c4':[16,16]}),
        'wsi_size': g.attrs.get('wsi_size', None),
        'input_size': g.attrs.get('input_size', None),
    }
    return c2, c3, c4, meta

# ---------- 你给的 UNet 解码器（FPN 风格） ----------
class YourUNetDecoder(nn.Module):
    """
    输入: c2 [B,C2,H/4,W/4], c3 [B,C3,H/8,W/8], c4 [B,C4,H/16,W/16]
    输出: logits [B,num_classes,H/4,W/4]
    """
    def __init__(self, C2: int, C3: int, C4: int,
                 num_classes: int = 1, fuse_ch: int = 64,
                 use_gn: bool = True, dropout: float = 0.0):
        super().__init__()
        Norm = (lambda c: nn.GroupNorm(8, c)) if use_gn else (lambda c: nn.BatchNorm2d(c))
        self.lat2 = nn.Sequential(nn.Conv2d(C2, fuse_ch, 1, bias=False), Norm(fuse_ch), nn.ReLU(inplace=True))
        self.lat3 = nn.Sequential(nn.Conv2d(C3, fuse_ch, 1, bias=False), Norm(fuse_ch), nn.ReLU(inplace=True))
        self.lat4 = nn.Sequential(nn.Conv2d(C4, fuse_ch, 1, bias=False), Norm(fuse_ch), nn.ReLU(inplace=True))
        self.smooth4 = nn.Sequential(nn.Conv2d(fuse_ch, fuse_ch, 3, padding=1, bias=False), Norm(fuse_ch), nn.ReLU(inplace=True))
        self.smooth3 = nn.Sequential(nn.Conv2d(fuse_ch, fuse_ch, 3, padding=1, bias=False), Norm(fuse_ch), nn.ReLU(inplace=True))
        self.smooth2 = nn.Sequential(nn.Conv2d(fuse_ch, fuse_ch, 3, padding=1, bias=False), Norm(fuse_ch), nn.ReLU(inplace=True))
        head_ch = max(fuse_ch, 32)
        self.head = nn.Sequential(
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(fuse_ch, head_ch, 3, padding=1, bias=False), Norm(head_ch), nn.ReLU(inplace=True),
            nn.Conv2d(head_ch, num_classes, 1)
        )

    def forward(self, c2: torch.Tensor, c3: torch.Tensor, c4: torch.Tensor) -> torch.Tensor:
        l2 = self.lat2(c2)      # /4
        l3 = self.lat3(c3)      # /8
        l4 = self.lat4(c4)      # /16
        p4 = self.smooth4(l4)   # /16
        p3 = F.interpolate(p4, size=l3.shape[-2:], mode='bilinear', align_corners=False) + l3  # /8
        p3 = self.smooth3(p3)
        p2 = F.interpolate(p3, size=l2.shape[-2:], mode='bilinear', align_corners=False) + l2  # /4
        p2 = self.smooth2(p2)
        logits = self.head(p2)  # /4
        return logits

# ---------- 训练数据集（按 c2 尺度滑窗） ----------
class ZarrTileDataset(Dataset):
    """
    从 dense zarr 中按 c2 尺度切 tile，映射到 c3/c4 对应子块，并返回 /4 尺度的 GT。
    """
    def __init__(self, zarr_dir: str, gt_path: str,
                 tile_px: int = 4096, tile_overlap_px: int = 0):
        self.c2, self.c3, self.c4, self.meta = load_zarr_layers(zarr_dir)
        C2,H2,W2 = self.c2.shape
        C3,H3,W3 = self.c3.shape
        C4,H4,W4 = self.c4.shape
        self.H2, self.W2 = H2, W2
        # 读取并下采样 GT -> c2 尺度
        self.gt = None
        if gt_path is not None and os.path.exists(gt_path):
            # 目标尺寸：c2 尺度
            target_size = (W2, H2)
            self.gt = read_mask_resized_to_c2(gt_path, target_size)  # [H2,W2] uint8 {0,1}

        # 滑窗索引（c2 尺度）
        ds2_h, ds2_w = self.meta['downsample'].get('c2', [4,4])
        self.tile_h2 = max(1, min(H2, tile_px // max(1, ds2_h)))
        self.tile_w2 = max(1, min(W2, tile_px // max(1, ds2_w)))
        self.stride_h2 = max(1, self.tile_h2 - tile_overlap_px // max(1, ds2_h))
        self.stride_w2 = max(1, self.tile_w2 - tile_overlap_px // max(1, ds2_w))

        idxs = []
        for y0 in range(0, H2, self.stride_h2):
            for x0 in range(0, W2, self.stride_w2):
                y1 = min(H2, y0 + self.tile_h2)
                x1 = min(W2, x0 + self.tile_w2)
                idxs.append((y0,x0,y1,x1))
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        y0,x0,y1,x1 = self.idxs[i]
        H2,W2 = self.H2, self.W2
        C2 = self.c2.shape[0]; C3 = self.c3.shape[0]; C4 = self.c4.shape[0]

        # 取 c2 子块
        t2 = self.c2[:, y0:y1, x0:x1]  # numpy [C2,h2,w2]
        # 映射到 c3/c4 子块（按整比映射，避免小数）
        y0_3, y1_3 = (y0 * self.c3.shape[1]) // H2, (y1 * self.c3.shape[1]) // H2
        x0_3, x1_3 = (x0 * self.c3.shape[2]) // W2, (x1 * self.c3.shape[2]) // W2
        y0_4, y1_4 = (y0 * self.c4.shape[1]) // H2, (y1 * self.c4.shape[1]) // H2
        x0_4, x1_4 = (x0 * self.c4.shape[2]) // W2, (x1 * self.c4.shape[2]) // W2
        t3 = self.c3[:, y0_3:y1_3, x0_3:x1_3]
        t4 = self.c4[:, y0_4:y1_4, x0_4:x1_4]

        # to torch
        t2 = torch.from_numpy(np.asarray(t2))   # [C2,h2,w2]
        t3 = torch.from_numpy(np.asarray(t3))   # [C3,h3,w3]
        t4 = torch.from_numpy(np.asarray(t4))   # [C4,h4,w4]

        # NHWC -> NCHW with batch=1 later
        if self.gt is not None:
            gt = torch.from_numpy(self.gt[y0:y1, x0:x1]).unsqueeze(0).float()  # [1,h2,w2]
        else:
            gt = torch.zeros(1, t2.shape[-2], t2.shape[-1], dtype=torch.float32)

        return t2, t3, t4, gt, (y0,x0,y1,x1)

# ---------- 训练 ----------
def train_one_zarr(zarr_path: str, gt_path: str, out_dir: str,
                   epochs=1, batch_tiles=1, device='cuda',
                   precision='fp16', lr=1e-3, fuse_ch=64, use_gn=True):
    os.makedirs(out_dir, exist_ok=True)

    ds = ZarrTileDataset(zarr_path, gt_path, tile_px=4096, tile_overlap_px=0)
    c2, c3, c4, _ = load_zarr_layers(zarr_path)
    C2, C3, C4 = c2.shape[0], c3.shape[0], c4.shape[0]

    # —— 模型保持 FP32，不要 .half() ——
    net = YourUNetDecoder(C2=C2, C3=C3, C4=C4, num_classes=1,
                          fuse_ch=fuse_ch, use_gn=use_gn).to(device)

    use_fp16 = (precision.lower() == 'fp16')
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    loss_fn = BCEDiceLoss(0.5)

    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    scaler = torch.amp.GradScaler('cuda', enabled=use_fp16)
    net.train()
    # 启用 channels_last 提升卷积效率
    try:
        net = net.to(memory_format=torch.channels_last)
    except Exception:
        pass
    torch.backends.cudnn.benchmark = True

    total_steps = len(loader)
    log_every = max(1, total_steps // 50)  # 每 ~2% 打一行

    accum = 0
    for ep in range(epochs):
        ep_loss = 0.0
        for step, (t2, t3, t4, gt, _) in enumerate(loader):
            # 输入也转为 channels_last（保持 dtype 交给 autocast）
            t2 = t2.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            t3 = t3.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            t4 = t4.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            gt = gt.to(device, dtype=torch.float32, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=use_fp16):
                logits = net(t2, t3, t4)
                loss = loss_fn(logits.float(), gt)

            # NaN/Inf 早停保护，避免 scaler 长期跳过 step
            if not torch.isfinite(loss):
                print(f"[warn] non-finite loss at ep{ep} step{step}: {loss.item()}")
                opt.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            ep_loss += float(loss.detach().cpu())

            accum += 1
            if accum >= batch_tiles:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                accum = 0

            # 释放局部引用（不要 empty_cache 以免同步）
            del t2, t3, t4, gt, logits, loss

            if (step % log_every) == 0:
                print(f"[train] ep {ep+1}/{epochs} step {step+1}/{total_steps} "
                      f"avg_loss={ep_loss/(step+1):.4f}")

    ckpt_path = os.path.join(out_dir, "ckpt_last.pt")
    torch.save({'model': net.state_dict(),
                'C2': C2, 'C3': C3, 'C4': C4,
                'fuse_ch': fuse_ch, 'use_gn': use_gn}, ckpt_path)
    print(f"[OK] saved: {ckpt_path}")
    return ckpt_path


# ---------- 推理/评估（整图流式） ----------
def _place_tile(sum_map: torch.Tensor, cnt_map: torch.Tensor,
                tile: torch.Tensor, y0: int, x0: int):
    # sum/count 累加，最后做 sum/count 取平均，避免接缝
    _, _, h, w = tile.shape
    sum_map[:, :, y0:y0+h, x0:x0+w] += tile
    cnt_map[:, :, y0:y0+h, x0:x0+w] += 1

def stream_decode_eval_png(zarr_dir: str, decoder: nn.Module,
                           ground_truth_path: str = None,
                           tile_px: int = 4096, tile_overlap_px: int = 0,
                           device: str = "cuda", precision: str = "fp16",
                           out_mask_png: str = None, out_overlay_png: str = None,
                           overlay_alpha: float = 0.35):
    c2, c3, c4, meta = load_zarr_layers(zarr_dir)
    C2,H2,W2 = c2.shape
    C3,H3,W3 = c3.shape
    C4,H4,W4 = c4.shape

    # 滑窗尺寸（c2 尺度）
    ds2_h, ds2_w = meta['downsample'].get('c2', [4,4])
    tile_h2 = max(1, min(H2, tile_px // max(1, ds2_h)))
    tile_w2 = max(1, min(W2, tile_px // max(1, ds2_w)))
    stride_h2 = max(1, tile_h2 - tile_overlap_px // max(1, ds2_h))
    stride_w2 = max(1, tile_w2 - tile_overlap_px // max(1, ds2_w))

    # 累计器
    sum_prob = torch.zeros(1,1,H2,W2, dtype=torch.float32, device='cpu')
    cnt_prob = torch.zeros(1,1,H2,W2, dtype=torch.float32, device='cpu')

    decoder = decoder.to(device).eval()
    use_fp16 = (precision.lower() == 'fp16')

    # ---- chunk 感知的 tile 尺度（以 c2 尺度为基准）----
    ds2_h, ds2_w = meta['downsample'].get('c2', [4, 4])

    # 从 zarr 里拿到 chunk 大小，避免 None
    c2_chunks = getattr(c2, "chunks", None)
    if c2_chunks and len(c2_chunks) == 3:
        _, hchunk2, wchunk2 = c2_chunks
    else:
        # 无法获取时给个温和默认值
        hchunk2, wchunk2 = 512, 512

    # 让 tile 至少覆盖 1 个 chunk，最多不超过 tile_px 预算
    tile_h2 = min(H2, max(hchunk2, (tile_px // max(1, ds2_h))))
    tile_w2 = min(W2, max(wchunk2, (tile_px // max(1, ds2_w))))
    # 重叠按像素折算到 c2 尺度
    stride_h2 = max(1, tile_h2 - (tile_overlap_px // max(1, ds2_h)))
    stride_w2 = max(1, tile_w2 - (tile_overlap_px // max(1, ds2_w)))

    # 统计总 tile 数，便于进度提示
    total_tiles = ((H2 + stride_h2 - 1) // stride_h2) * ((W2 + stride_w2 - 1) // stride_w2)
    done_tiles = 0

    # decoder 设置为 channels_last，可稍微提升卷积性能
    decoder = decoder.to(device).eval()
    try:
        decoder = decoder.to(memory_format=torch.channels_last)
    except Exception:
        pass

    use_fp16 = (precision.lower() == "fp16")

    # 累计图：sum/count（在 CPU 上），避免显存暴涨
    sum_prob = torch.zeros(1, 1, H2, W2, dtype=torch.float32, device="cpu")
    cnt_prob = torch.zeros(1, 1, H2, W2, dtype=torch.float32, device="cpu")

    # ---- 推理主循环：按 c2 尺度切块，并对齐到 c3/c4 子块 ----
    with torch.inference_mode():
        for y0 in range(0, H2, stride_h2):
            y1 = min(H2, y0 + tile_h2)
            # 对应 c3/c4 的整数映射区间（严格整比）
            y0_3, y1_3 = (y0 * H3) // H2, (y1 * H3) // H2
            y0_4, y1_4 = (y0 * H4) // H2, (y1 * H4) // H2

            for x0 in range(0, W2, stride_w2):
                x1 = min(W2, x0 + tile_w2)
                x0_3, x1_3 = (x0 * W3) // W2, (x1 * W3) // W2
                x0_4, x1_4 = (x0 * W4) // W2, (x1 * W4) // W2

                # ---- 从 zarr 读取连续块（numpy 视图）----
                # 注意：切片与 chunk 对齐时读取速度会明显更好
                t2_np = np.asarray(c2[:, y0:y1, x0:x1], dtype=np.float32)  # 保持 float32 前向更稳
                t3_np = np.asarray(c3[:, y0_3:y1_3, x0_3:x1_3], dtype=np.float32)
                t4_np = np.asarray(c4[:, y0_4:y1_4, x0_4:x1_4], dtype=np.float32)

                # ---- 转 torch 张量（CPU->GPU），channels_last ----
                t2 = torch.from_numpy(t2_np).unsqueeze(0).to(device, non_blocking=True)
                t3 = torch.from_numpy(t3_np).unsqueeze(0).to(device, non_blocking=True)
                t4 = torch.from_numpy(t4_np).unsqueeze(0).to(device, non_blocking=True)
                try:
                    t2 = t2.contiguous(memory_format=torch.channels_last)
                    t3 = t3.contiguous(memory_format=torch.channels_last)
                    t4 = t4.contiguous(memory_format=torch.channels_last)
                except Exception:
                    pass

                # ---- 前向：autocast 做混合精度；logits 回 FP32 做 sigmoid 和累计 ----
                with torch.amp.autocast('cuda', enabled=use_fp16):
                    logits = decoder(t2, t3, t4)               # [1,1,h2,w2]
                prob = torch.sigmoid(logits.float()).to("cpu", non_blocking=True)

                # ---- 无缝融合（sum/count），避免接缝 ----
                _place_tile(sum_prob, cnt_prob, prob, y0, x0)

                # 释放局部变量（不调用 empty_cache，避免隐式同步）
                del t2, t3, t4, logits, prob

                done_tiles += 1
                if (done_tiles % 8) == 0 or done_tiles == total_tiles:
                    print(f"[infer] tiles {done_tiles}/{total_tiles} (c2 scale {H2}x{W2})")

    # ---- 汇总整图概率（c2 尺度）----
    prob_full = sum_prob / torch.clamp_min(cnt_prob, 1.0)
    # pred_bin = (prob_full >= 0.5).float()

    # 仅当 ground_truth_path 不为空时评估
    dice = iou = 0.0
    if ground_truth_path is not None and os.path.exists(ground_truth_path):
        gt2 = read_mask_resized_to_c2(ground_truth_path, (W2, H2))  # {0,1}
        pred2 = (prob_full[0,0].numpy() >= 0.5).astype(np.uint8)
        inter = (gt2 & pred2).sum()
        dice = float((2 * inter) / (gt2.sum() + pred2.sum() + 1e-6))
        union = (gt2 | pred2).sum()
        iou  = float(inter / (union + 1e-6))


    # 保存可视化
    if out_mask_png is not None:
        vis = (prob_full[0,0].clamp(0,1).numpy()*255).astype(np.uint8)
        Image.fromarray(vis).save(out_mask_png)
    if out_overlay_png is not None:
        # 用下采样到 c2 尺度的“原图替代物”叠加（没有原 RGB，就只存热力图）
        base = Image.fromarray(np.zeros((H2,W2), dtype=np.uint8)).convert("RGB")
        heat = Image.fromarray((prob_full[0,0].clamp(0,1).numpy()*255).astype(np.uint8)).convert("RGBA")
        heat.putalpha(int(overlay_alpha*255))
        over = Image.alpha_composite(base.convert("RGBA"), heat)
        over.save(out_overlay_png)

    return dice, iou, prob_full  # prob_full: [1,1,H2,W2] CPU

# ---------- 主函数 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zarr', type=str, required=True, help='path to slide.zarr')
    ap.add_argument('--gt',   type=str, required=True, help='path to pixel GT (png/tiff)')
    ap.add_argument('--out',  type=str, required=True, help='out dir')
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--precision', type=str, default='fp16', choices=['fp16','fp32'])
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--fuse_ch', type=int, default=64)
    ap.add_argument('--use_gn', action='store_true', help='use GroupNorm (recommended)')
    args = ap.parse_args()

    set_seed(2025)
    os.makedirs(args.out, exist_ok=True)

    # 训练
    ckpt = train_one_zarr(args.zarr, args.gt, args.out,
                          epochs=args.epochs, batch_tiles=1,
                          device=args.device, precision=args.precision,
                          lr=args.lr, fuse_ch=args.fuse_ch, use_gn=args.use_gn)

    # 构建解码器并加载
    c2, c3, c4, _ = load_zarr_layers(args.zarr)
    C2,C3,C4 = c2.shape[0], c3.shape[0], c4.shape[0]
    net = YourUNetDecoder(C2=C2, C3=C3, C4=C4, num_classes=1,
                          fuse_ch=args.fuse_ch, use_gn=args.use_gn)
    sd = torch.load(ckpt, map_location='cpu', weights_only=True)
    net.load_state_dict(sd['model'], strict=True)

    # 推理 + 评估 + 保存
    out_mask = os.path.join(args.out, 'pred_mask.png')
    out_ovl  = os.path.join(args.out, 'overlay.png')
    dice, iou, _ = stream_decode_eval_png(
        zarr_dir=args.zarr,
        decoder=net,
        ground_truth_path=args.gt,
        tile_px=4096,
        tile_overlap_px=0,
        device=args.device,
        precision=args.precision,
        out_mask_png=out_mask,
        out_overlay_png=out_ovl,
        overlay_alpha=0.35
    )
    print(f"[EVAL] Dice={dice:.4f} IoU={iou:.4f}")

if __name__ == "__main__":
    main()
