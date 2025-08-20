import h5py
from wsi_stream_decode import stream_decode_eval_png
from unet_decoder import YourUNetDecoder
h5_path = "/mnt/gemlab_data_3/LXQ/PHASE/features/h5_files/001c62abd11fa4b57bf7a6c603a11bb9.h5"
slide_path = "/mnt/gemlab_data_2/User_database/zhushiwei/PHASE/train_images/001c62abd11fa4b57bf7a6c603a11bb9.tiff"   # 若没有原始 WSI 文件，可设为 None
ground_path="/mnt/gemlab_data_2/User_database/zhushiwei/PHASE/train_label_masks/001c62abd11fa4b57bf7a6c603a11bb9_mask.tiff"  # 若没有 ground truth 掩码，可设为 None
# h5_path = "/mnt/gemlab_data_3/LXQ/PHASE/features/h5_files/00a26aaa82c959624d90dfb69fcf259c.h5"
# slide_path = "/mnt/gemlab_data_2/User_database/zhushiwei/PHASE/train_images/00a26aaa82c959624d90dfb69fcf259c.tiff"   # 若没有原始 WSI 文件，可设为 None
# ground_path="/mnt/gemlab_data_2/User_database/zhushiwei/PHASE/train_label_masks/00a26aaa82c959624d90dfb69fcf259c_mask.tiff"  # 若没有 ground truth 掩码，可设为 None
out_path = "/mnt/gemlab_data_3/LXQ"
coords_order = "xy"  # 'xy' 或 'yx'

# C = 256            # 你的特征通道数
num_classes = 1    # 二分类掩码：1 通道；多类改为 >1
with h5py.File(h5_path, "r") as f:
    C2 = f["c2/features"].shape[1]
    C3 = f["c3/features"].shape[1]
    C4 = f["c4/features"].shape[1]
C_in = C2 + C3 + C4
print(f"[INIT] C2={C2}, C3={C3}, C4={C4}  =>  C_in={C_in}")

decoder = YourUNetDecoder(
    C_in=C_in,        # 输入通道数，来自 HDF5 文件
    num_classes=num_classes,
    base_ch=64,        # 显存吃紧可降到 32
    depth=2,           # 2 层 U 形；显存紧张可设 1，极端设 0（仅局部卷积，不下采样）
    bilinear=True,     # 双线性上采样更省参数
    norm="bn",
    separable=True,    # 打开深度可分离卷积，显著省算力
    dropout=0.0,
    reduce_first=4     # 先把 C_in 压到 C_in/4 再进网络
).eval().to("cuda")

# 2) 运行：生成PNG + 输出Dice/IoU
dice, iou = stream_decode_eval_png(
    h5_path=h5_path,
    slide_path=slide_path,
    ground_truth_path=ground_path,
    decoder=decoder,
    coords_order="xy",
    tile_px=8192,           # 显存紧张可用 4096
    tile_overlap_px=0,      # 评估期建议 0，避免重叠重复计数
    upsample_mode="bilinear",
    device="cuda",
    out_mask_png="/mnt/gemlab_data_3/LXQ/pred_mask_vis.png",
    out_overlay_png="/mnt/gemlab_data_3/LXQ/pred_overlay_vis.png",
    vis_down=16,            # 可视化下采样；看不清可降到 8
    overlay_alpha=0.35,
    # debug_mode="coverage",  # 'coverage' 或 'full'; 'coverage' 只输出覆盖率
)
print("Dice/IoU:", dice, iou)

