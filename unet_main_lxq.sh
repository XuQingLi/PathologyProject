python train_unet_from_zarr.py \
  --zarr /mnt/gemlab_data_3/LXQ/PHASE/featuresZarrLatest/0018ae58b01bdadc8e347995b69f99aa.zarr \
  --gt   /mnt/gemlab_data_2/User_database/zhushiwei/PHASE/train_label_masks/0018ae58b01bdadc8e347995b69f99aa_mask.tiff \
  --out  /mnt/gemlab_data_3/LXQ \
  --epochs 2 \
  --device cuda \
  --precision fp16

  # --zarr /mnt/gemlab_data_3/LXQ/PHASE/featuresZarrLatest/00412139e6b04d1e1cee8421f38f6e90.zarr \
  # --gt   /mnt/gemlab_data_2/User_database/zhushiwei/PHASE/train_label_masks/00412139e6b04d1e1cee8421f38f6e90_mask.tiff \