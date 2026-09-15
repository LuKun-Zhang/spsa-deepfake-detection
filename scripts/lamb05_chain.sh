#!/bin/bash
B=/root/autodl-tmp/DeepfakeBench
cd $B
echo "=== lamb05 s2048 TRAIN start $(date) ==="
rm -rf training/trainer/__pycache__ training/detectors/__pycache__ 2>/dev/null
/root/miniconda3/bin/python training/train.py --detector_path /root/autodl-tmp/stat_cfgs/stat_E1_lamb05_s2048.yaml --no-save_feat > /root/autodl-tmp/train_lamb05_s2048.log 2>&1
CKPT=$(ls -td $B/logs/training/ucf_stat_E1_lamb05_s2048_*/test/Celeb-DF-v2/ckpt_best.pth 2>/dev/null | head -1)
echo "s2048 CKPT=$CKPT"
/root/miniconda3/bin/python training/test.py --detector_path /root/autodl-tmp/stat_cfgs/stat_E1_lamb05_s2048.yaml --test_dataset DFDC DFDCP --weights_path "$CKPT" > /root/autodl-tmp/test_cross_lamb05_s2048.log 2>&1
echo "=== lamb05 s4096 TRAIN start $(date) ==="
rm -rf training/trainer/__pycache__ training/detectors/__pycache__ 2>/dev/null
/root/miniconda3/bin/python training/train.py --detector_path /root/autodl-tmp/stat_cfgs/stat_E1_lamb05_s4096.yaml --no-save_feat > /root/autodl-tmp/train_lamb05_s4096.log 2>&1
CKPT=$(ls -td $B/logs/training/ucf_stat_E1_lamb05_s4096_*/test/Celeb-DF-v2/ckpt_best.pth 2>/dev/null | head -1)
echo "s4096 CKPT=$CKPT"
/root/miniconda3/bin/python training/test.py --detector_path /root/autodl-tmp/stat_cfgs/stat_E1_lamb05_s4096.yaml --test_dataset DFDC DFDCP --weights_path "$CKPT" > /root/autodl-tmp/test_cross_lamb05_s4096.log 2>&1
echo "=== LAMB05 CHAIN ALL DONE $(date) ==="
