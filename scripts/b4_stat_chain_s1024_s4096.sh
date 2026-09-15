#!/bin/bash
# 659 machine - B4 cross-backbone, controlled protocol, 2 extra seeds (s1024, s4096).
# s2048 already done on 652. Order: complete a full seed pair (CTRL then SPSA1SWA) ASAP.
set -e
export PYTHONUNBUFFERED=1
cd /root/autodl-tmp/DeepfakeBench
PY=/root/miniconda3/bin/python
D=/root/autodl-tmp/DeepfakeBench/training/config/detector
L=/root/autodl-tmp/DeepfakeBench/logs
echo "[b4-chain] $(date +%Y-%m-%d_%H:%M:%S) start B4-CTRL s1024"
$PY training/train.py --detector_path $D/b4_stat_ctrl_efficientnetb4_s1024.yaml > $L/b4ctrl_s1024_stdout.log 2>&1
echo "[b4-chain] $(date +%Y-%m-%d_%H:%M:%S) B4-CTRL s1024 done"
$PY training/train.py --detector_path $D/b4_stat_SPSA1SWA_efficientnetb4_s1024.yaml > $L/b4spsa_s1024_stdout.log 2>&1
echo "[b4-chain] $(date +%Y-%m-%d_%H:%M:%S) B4-SPSA1SWA s1024 done"
$PY training/train.py --detector_path $D/b4_stat_ctrl_efficientnetb4_s4096.yaml > $L/b4ctrl_s4096_stdout.log 2>&1
echo "[b4-chain] $(date +%Y-%m-%d_%H:%M:%S) B4-CTRL s4096 done"
$PY training/train.py --detector_path $D/b4_stat_SPSA1SWA_efficientnetb4_s4096.yaml > $L/b4spsa_s4096_stdout.log 2>&1
echo "[b4-chain] $(date +%Y-%m-%d_%H:%M:%S) B4-SPSA1SWA s4096 done"
echo "B4_CHAIN_S1024_S4096_DONE $(date +%Y-%m-%d_%H:%M:%S)"
