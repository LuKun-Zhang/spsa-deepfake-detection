#!/bin/bash
OUT=/root/autodl-tmp/spsa2_out.log
cd /root/autodl-tmp/DeepfakeBench
PY=/root/miniconda3/bin/python
echo "[spsa2] $(date +%H:%M:%S) start s1024" >> $OUT
$PY training/train.py --detector_path /root/autodl-tmp/stat_cfgs/stat_SPSA2_s1024.yaml > /root/autodl-tmp/train_spsa2_s1024.log 2>&1
echo "[spsa2] $(date +%H:%M:%S) s1024 done" >> $OUT
echo "[spsa2] $(date +%H:%M:%S) start s2048" >> $OUT
$PY training/train.py --detector_path /root/autodl-tmp/stat_cfgs/stat_SPSA2_s2048.yaml > /root/autodl-tmp/train_spsa2_s2048.log 2>&1
echo "[spsa2] $(date +%H:%M:%S) s2048 done" >> $OUT
echo "[spsa2] $(date +%H:%M:%S) SPSA2_CHAIN_DONE" >> $OUT
