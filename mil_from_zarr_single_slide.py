# mil_from_zarr_c5.py
import os
import zarr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_clam import CLAM_SB  # 若需要多原型可改为 CLAM_MB

# ========= 基本配置 =========
ZARR_PATH   = "/mnt/gemlab_data_3/LXQ/PHASE/featuresZarrLatest/00928370e2dfeb8a507667ef1d4efcbb.zarr"
N_CLASSES   = 2                 # 二分类示例
LABEL_INT   = 1                 # 当前 WSI 的监督标签
SAVE_CKPT   = "./mil_from_c5_clam_only.ckpt"
LR          = 1e-4
WD          = 1e-4
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# ========= 读取 zarr 中的 c5-MIL 视图 =========
def load_mil_from_zarr(zarr_path: str):
    """
    读取：
      - root["mil/emb"]   -> [N, C5]  float16/float32
      - root["mil/coords"]（可选）-> [N, 2] int32
    返回 (tokens[N,C5]: torch.float32, coords[N,2] or None)
    """
    z = zarr.open(zarr_path, mode="r")
    if "mil" not in z or "emb" not in z["mil"]:
        raise FileNotFoundError(f"'mil/emb' not found in {zarr_path}")
    emb = z["mil/emb"][...]                 # numpy, [N,C5]
    tokens = torch.from_numpy(emb).float()  # -> float32

    coords = None
    if "coords" in z["mil"]:
        coords = z["mil/coords"][...]      # numpy [N,2]，仅可解释使用

    return tokens, coords

def main():
    # 1) 读入 MIL tokens（c5 GAP 后的 patch 表征）
    tokens, coords = load_mil_from_zarr(ZARR_PATH)  # tokens: [N,C5]
    N, C5 = tokens.shape
    print(f"[INFO] loaded mil/emb from zarr: tokens shape={tuple(tokens.shape)}  (N={N}, C5={C5})")

    # 2) 构建 CLAM（注意 embed_dim 必须与 tokens 的通道数一致）
    clam = CLAM_SB(
        gate=True, size_arg="small", dropout=0.0,
        k_sample=8, n_classes=N_CLASSES,
        embed_dim=C5
    ).to(DEVICE)

    # 3) 标签与优化器
    label = torch.tensor([LABEL_INT], dtype=torch.long, device=DEVICE)  # [1]
    tokens = tokens.to(DEVICE)  # [N,C5]

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(clam.parameters(), lr=LR, weight_decay=WD)

    # 4) 单步训练示例
    clam.train()
    optimizer.zero_grad()
    logits, Y_prob, Y_hat, A_raw, extra = clam(tokens, label=label, instance_eval=False)  # logits: [1,n_classes]
    loss = criterion(logits, label)
    loss.backward()
    optimizer.step()

    # 5) 打印结果
    with torch.no_grad():
        prob = Y_prob.detach().cpu().numpy()
        pred = Y_hat.detach().cpu().numpy()
    print(f"[INFO] logits={logits.detach().cpu().numpy()}, prob={prob}, pred={pred}, loss={loss.item():.4f}")

    # 6) （可选）可解释性：有 coords 时可将注意力 A_raw 回映射
    if coords is not None:
        print(f"[INFO] coords available: shape={coords.shape}  # 可用于注意力热图回映射")

    # 7) 保存权重
    ckpt = {
        "clam_state": clam.state_dict(),
        "n_classes": N_CLASSES,
        "embed_dim": C5,
        "zarr_path": ZARR_PATH,
    }
    torch.save(ckpt, SAVE_CKPT)
    print(f"[OK] saved ckpt to {SAVE_CKPT}")

if __name__ == "__main__":
    main()
