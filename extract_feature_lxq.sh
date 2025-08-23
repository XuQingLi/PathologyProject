export CUDA_VISIBLE_DEVICES=0,2,3

# 执行特征提取脚本
#!/bin/bash
# 运行特征提取脚本

python ../extract_feature_lxq.py \
  --slide_dir /mnt/gemlab_data_2/User_database/zhushiwei/PHASE/train_images \
  --out_dir /mnt/gemlab_data_3/LXQ/PHASE/featuresZarrNew \
  --device cuda \
  --patch 224 \
  --stride 224 \
  --batch_size 64 \
  --tissue_thr 0.05 \
  --gpus 0,1,2 \
  --num_workers 3 \
  --skip_if_done