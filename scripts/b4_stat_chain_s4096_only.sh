#!/bin/bash
# 601-cutoff recovery: run only the two s4096 legs (CTRL then SPSA1SWA).
set -e
export PYTHONUNBUFFERED=1
cd /root/autodl-tmp/DeepfakeBench
PY=/root/miniconda3/bin/python
D=/root/autodl-tmp/DeepfakeBench/training/config/detector
L=/root/autodl-tmp/DeepfakeBench/logs
echo "[b4-s4096-chain] $(date +%Y-%m-%d_%H:%M:%S) start B4-CTRL s4096"
$PY training/train.py --detector_path $D/b4_stat_ctrl_efficientnetb4_s4096.yaml > $L/b4ctrl_s4096_stdout.log 2>&1
echo "[b4-s4096-chain] $(date +%Y-%m-%d_%H:%M:%S) B4-CTRL s4096 done"
$PY training/train.py --detector_path $D/b4_stat_SPSA1SWA_efficientnetb4_s4096.yaml > $L/b4spsa_s4096_stdout.log 2>&1
echo "[b4-s4096-chain] $(date +%Y-%m-%d_%H:%M:%S) B4-SPSA1SWA s4096 done"
echo "B4_CHAIN_S4096_ONLY_DONE $(date +%Y-%m-%d_%H:%M:%S)"
