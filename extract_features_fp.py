import time
import os
import argparse
import torch
import h5py
import openslide
from tqdm import tqdm
import numpy as np
import json

from torch.utils.data import DataLoader
from utils.file_utils import save_hdf5  # 如果项目里其他地方用到了，保留导入
from dataset_modules.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from models import get_encoder_lxq

# -------------------------
# 全局设备
# -------------------------
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# -------------------------
# H5 工具
# -------------------------
def _init_or_open_h5(path, meta=None):
    h5 = h5py.File(path, 'a')
    if meta is not None:
        h5.require_group('meta')
        for k, v in meta.items():
            if k == 'downsample':
                h5['meta'].attrs['downsample'] = json.dumps(v)
            elif isinstance(v, (list, tuple)):
                h5['meta'].attrs[k] = json.dumps(list(v))
            elif isinstance(v, dict):
                h5['meta'].attrs[k] = json.dumps(v)
            else:
                h5['meta'].attrs[k] = v
    return h5

def _append_ds(h5, name, arr, compression='gzip', compression_opts=4):
    """
    将 arr 追加到 h5[name] 的第 0 维（N）末尾；不存在则新建可扩展数据集。
    """
    arr = np.asarray(arr)
    if name in h5:
        ds = h5[name]
        n_old = ds.shape[0]
        n_new = n_old + arr.shape[0]
        ds.resize((n_new,) + ds.shape[1:])
        ds[n_old:n_new] = arr
    else:
        maxshape = (None,) + arr.shape[1:]
        h5.create_dataset(
            name, data=arr, maxshape=maxshape, chunks=True,
            compression=compression, compression_opts=compression_opts
        )

def _ensure_coords(h5, coords_np):
    _append_ds(h5, 'coords', coords_np.astype(np.int32))


# -------------------------
# 多尺度空间特征提取与保存
# -------------------------
def compute_w_loader_lxq(
    output_path,
    loader,
    model,
    device='cuda',
    verbose=1,
    wanted_levels=('c2', 'c3', 'c4', 'c5'),
    dtype_np=np.float16,
    write_compat_features=True,
    compat_level='c2'
):
    """
    将模型返回的多尺度空间特征（FPN风格 dict: c2..c5）逐批保存到 HDF5。
    同时（可选）将某一主层写到 '/features'，与旧管线兼容。

    Args:
        output_path: 目标 h5 文件路径
        loader:      DataLoader，data['img']: [B,3,H,W], data['coord']: [B,2]
        model:       前向返回 dict({'c2':T,[B,C,H,W], ...})
        wanted_levels: 要保存的层名子集
        dtype_np:    存储精度（默认 float16 节省容量）
        write_compat_features: 是否另存 '/features'
        compat_level: 另存为 '/features' 的主层名（默认 'c2'）
    """
    model.eval()
    levels_observed = None
    downsample = {}      # {level: (ds_h, ds_w)}
    meta_written = False
    h5 = None

    if verbose:
        print(f'processing a total of {len(loader)} batches')

    with torch.inference_mode():
        for bidx, data in enumerate(tqdm(loader)):
            batch = data['img'].to(device, non_blocking=True)
            coords = data['coord'].numpy().astype(np.int32)

            out = model(batch)
            if not isinstance(out, dict):
                raise TypeError(f"Expected model to return dict of feature maps, got {type(out)}")

            if levels_observed is None:
                levels_observed = list(out.keys())
                if verbose:
                    print(f"Observed levels: {levels_observed}")

            # 初始化 H5 与 meta
            if h5 is None:
                _, _, H_in, W_in = batch.shape
                meta = {
                    'input_size': (int(H_in), int(W_in)),
                    'levels': list(wanted_levels),
                    'compat_level': compat_level if write_compat_features else ''
                }
                h5 = _init_or_open_h5(output_path, meta=meta)

            # 先写 coords
            _ensure_coords(h5, coords)

            # 逐层保存
            for lv in wanted_levels:
                if lv not in out:
                    if bidx == 0 and verbose:
                        print(f"[warn] level '{lv}' not in model output keys {list(out.keys())}, skip.")
                    continue

                feat = out[lv]
                if not torch.is_tensor(feat):
                    raise TypeError(f"Output[{lv}] is not a Tensor, got {type(feat)}")

                if feat.ndim != 4:
                    raise ValueError(f"Output[{lv}] must be 4D (B,C,H,W), got shape {tuple(feat.shape)}")

                # 首次计算下采样倍率
                if lv not in downsample:
                    _, _, Hf, Wf = feat.shape
                    ds_h = batch.shape[-2] // Hf
                    ds_w = batch.shape[-1] // Wf
                    downsample[lv] = (int(ds_h), int(ds_w))
                    if verbose and bidx == 0:
                        print(f"[{lv}] downsample ≈ (h={ds_h}, w={ds_w}) from {batch.shape[-2:]} -> {(Hf, Wf)}")

                feat_np = feat.detach().cpu().numpy().astype(dtype_np)  # [B,C,H,W]
                _append_ds(h5, f'{lv}/features', feat_np)

                # 兼容写到 '/features'
                if write_compat_features and lv == compat_level:
                    _append_ds(h5, 'features', feat_np)

            # 写 meta/downsample（只需一次）
            if not meta_written and len(downsample) > 0:
                h5['meta'].attrs['downsample'] = json.dumps(downsample)
                meta_written = True

    if h5 is not None:
        h5.close()
    if verbose:
        print(f"Features saved to: {output_path}")

    return output_path


# -------------------------
# CLI
# -------------------------
parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default='.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--model_name', type=str, default='resnet50_trunc',
                    choices=['resnet50_trunc', 'uni_v1', 'conch_v1'])
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--target_patch_size', type=int, default=224)
args = parser.parse_args()


# -------------------------
# 主流程
# -------------------------
if __name__ == '__main__':
    print('initializing dataset')
    csv_path = args.csv_path
    if csv_path is None:
        raise NotImplementedError("Please provide --csv_path")

    bags_dataset = Dataset_All_Bags(csv_path)

    os.makedirs(args.feat_dir, exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
    dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

    # 编码器（含 transforms）
    model, img_transforms = get_encoder_lxq(args.model_name, target_img_size=args.target_patch_size)
    print('model:', model)
    model.eval()
    model = model.to(device)
    total = len(bags_dataset)

    loader_kwargs = {'num_workers': 8, 'pin_memory': True} if device.type == "cuda" else {}

    for bag_candidate_idx in tqdm(range(total)):
        slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
        bag_name = slide_id + '.h5'
        h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
        slide_file_path = os.path.join(args.data_slide_dir, slide_id + args.slide_ext)
        print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
        print(slide_id)

        if not args.no_auto_skip and slide_id + '.pt' in dest_files:
            print('skipped {}'.format(slide_id))
            continue

        output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
        time_start = time.time()

        wsi = openslide.open_slide(slide_file_path)
        dataset = Whole_Slide_Bag_FP(
            file_path=h5_file_path,
            wsi=wsi,
            img_transforms=img_transforms
        )

        loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **loader_kwargs)

        # 核心：保存多尺度空间特征；同时写一份 '/features' = c2
        output_file_path = compute_w_loader_lxq(
            output_path,
            loader=loader,
            model=model,
            device=device,
            verbose=1,
            wanted_levels=('c2', 'c3', 'c4', 'c5'),
            dtype_np=np.float16,              # 如需高精度改为 np.float32
            write_compat_features=True,       # 兼容老管线
            compat_level='c2'                 # '/features' 指向 c2
        )

        time_elapsed = time.time() - time_start
        print('\ncomputing features for {} took {:.2f} s'.format(output_file_path, time_elapsed))

        # 兼容读取：此处仍按 '/features' 读取（即 c2）；如需某层，改为 f['c3/features'] 等
        with h5py.File(output_file_path, "r") as f:
            if 'features' in f:
                features = f['features'][:]       # (N,C,H,W) from compat_level
            else:
                # 回退：若未写 compat，则取最高分辨率可用层
                for cand in ('c2', 'c3', 'c4', 'c5'):
                    if f.get(f'{cand}/features') is not None:
                        features = f[f'{cand}/features'][:]
                        break
                else:
                    raise RuntimeError("No features found in H5.")
            coords = f['coords'][:]
            print('features size: ', features.shape)
            print('coordinates size: ', coords.shape)

        # 保存为 .pt（与原流程一致）
        features_t = torch.from_numpy(features)  # dtype: float16/float32
        bag_base, _ = os.path.splitext(bag_name)
        torch.save(features_t, os.path.join(args.feat_dir, 'pt_files', bag_base + '.pt'))