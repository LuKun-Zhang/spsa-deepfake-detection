#!/bin/bash
cd /root/autodl-tmp/DeepfakeBench
echo "=== A_s2048 cross start $(date) ==="
/root/miniconda3/bin/python training/test.py \
  --detector_path /root/autodl-tmp/stat_cfgs/stat_A_s2048.yaml \
  --test_dataset DFDC DFDCP \
  --weights_path /root/autodl-tmp/DeepfakeBench/logs/training/ucf_stat_A_s2048_2026-08-20-16-02-38/test/Celeb-DF-v2/ckpt_best.pth \
  > /root/autodl-tmp/test_cross_A_s2048.log 2>&1
echo "=== A_s4096 cross start $(date) ==="
/root/miniconda3/bin/python training/test.py \
  --detector_path /root/autodl-tmp/stat_cfgs/stat_A_s4096.yaml \
  --test_dataset DFDC DFDCP \
  --weights_path /root/autodl-tmp/DeepfakeBench/logs/training/ucf_stat_A_s4096_2026-08-20-21-34-36/test/Celeb-DF-v2/ckpt_best.pth \
  > /root/autodl-tmp/test_cross_A_s4096.log 2>&1
echo "=== A CROSS ALL DONE $(date) ==="
