#!/bin/bash
set -e
cd /root/autodl-tmp/DeepfakeBench
PY=/root/miniconda3/bin/python
echo "[spsa1swa] $(date +%H:%M:%S) start s1024"
$PY training/train.py --detector_path /root/autodl-tmp/stat_cfgs/stat_SPSA1SWA_s1024.yaml > /root/autodl-tmp/train_spsa1swa_s1024.log 2>&1
echo "[spsa1swa] $(date +%H:%M:%S) s1024 done"
echo "[spsa1swa] $(date +%H:%M:%S) start s2048"
$PY training/train.py --detector_path /root/autodl-tmp/stat_cfgs/stat_SPSA1SWA_s2048.yaml > /root/autodl-tmp/train_spsa1swa_s2048.log 2>&1
echo "[spsa1swa] $(date +%H:%M:%S) s2048 done"
echo "SPSA1SWA_CHAIN_DONE $(date +%H:%M:%S)"
