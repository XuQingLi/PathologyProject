import h5py, json, torch

path = "/mnt/gemlab_data_3/LXQ/PHASE/features/h5_files/001c62abd11fa4b57bf7a6c603a11bb9.h5"
with h5py.File(path, "r") as f:
    # 1) 关键节点是否存在
    print("keys at root:", list(f.keys()))                  # 期望含 'coords', 'c2', 'c3', ...
    print("c2 subkeys:", list(f['c2'].keys()))              # 期望含 'features'
    print("has compat /features:", 'features' in f)

    # 2) 形状是否合理
    coords = f['coords'][:]
    c2 = f['c2/features'][:]
    print("coords:", coords.shape)                          # (N, 2)
    print("c2/features:", c2.shape)                         # (N, C2, H2, W2)

    # 3) downsample 是否存在且可解析
    down = json.loads(f['meta'].attrs['downsample'])
    print("downsample:", down)                              # 例如 {'c2': [4,4], 'c3':[8,8],...}

# 4) .pt 文件能否用 torch.load 打开
pt_path = "/mnt/gemlab_data_3/LXQ/PHASE/features/pt_files/001c62abd11fa4b57bf7a6c603a11bb9.pt"
t = torch.load(pt_path, map_location="cpu",weights_only=False)
print(type(t), t.shape)  # 期望: torch.Tensor, (N, C2, H2, W2)

import numpy as np

# 以 c2 为例
is_finite = np.isfinite(c2).all()
print("finite:", is_finite)             # True 才行

# 不是全零、不是常数
print("min:", float(c2.min()), "max:", float(c2.max()))
print("std:", float(c2.std()))
