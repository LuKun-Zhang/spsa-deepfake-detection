#!/bin/bash
OUT=/root/autodl-tmp/spsa_intra_out.log
# 等 GPU 空闲 (4096 训练结束后)
while [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) -gt 1000 ]; do sleep 60; done
echo "[intra] $(date +%H:%M:%S) GPU free, start intra-domain tests" >> $OUT
cd /root/autodl-tmp/DeepfakeBench
DET=/root/autodl-tmp/stat_cfgs/stat_SPSA_test.yaml
CK1=/root/autodl-tmp/DeepfakeBench/logs/training/ucf_2026-08-22-14-28-48/test/Celeb-DF-v2/ckpt_best.pth
CK2=/root/autodl-tmp/DeepfakeBench/logs/training/ucf_2026-08-22-19-26-20/test/Celeb-DF-v2/ckpt_best.pth
CK3=/root/autodl-tmp/DeepfakeBench/logs/training/ucf_2026-08-22-21-53-49/test/Celeb-DF-v2/ckpt_best.pth
/root/miniconda3/bin/python training/test.py --detector_path $DET --test_dataset FaceForensics++ --weights_path $CK1 > /root/autodl-tmp/test_intra_spsa_s1024.log 2>&1 && echo "[intra] s1024 done" >> $OUT
/root/miniconda3/bin/python training/test.py --detector_path $DET --test_dataset FaceForensics++ --weights_path $CK2 > /root/autodl-tmp/test_intra_spsa_s2048.log 2>&1 && echo "[intra] s2048 done" >> $OUT
/root/miniconda3/bin/python training/test.py --detector_path $DET --test_dataset FaceForensics++ --weights_path $CK3 > /root/autodl-tmp/test_intra_spsa_s4096.log 2>&1 && echo "[intra] s4096 done" >> $OUT
echo "[intra] $(date +%H:%M:%S) ALL_DONE" >> $OUT
