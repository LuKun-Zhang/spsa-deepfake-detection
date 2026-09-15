#!/bin/bash
# B4 cross-backbone directionality run (controlled protocol, seed 2048, single seed)
# SPSA1SWA then CTRL. Chain in one GPU reservation.
set -e
cd /root/autodl-tmp/DeepfakeBench
PY=/root/miniconda3/bin/python
echo "[b4-stat-chain] $(date +%Y-%m-%d_%H:%M:%S) start B4-SPSA1SWA s2048 (controlled 4ep, seed2048)"
$PY training/train.py --detector_path /root/autodl-tmp/DeepfakeBench/training/config/detector/b4_stat_SPSA1SWA_efficientnetb4_s2048.yaml > /root/autodl-tmp/DeepfakeBench/logs/b4spsa_s2048_stdout.log 2>&1
echo "[b4-stat-chain] $(date +%Y-%m-%d_%H:%M:%S) B4-SPSA1SWA done"
echo "[b4-stat-chain] $(date +%Y-%m-%d_%H:%M:%S) start B4-CTRL s2048"
$PY training/train.py --detector_path /root/autodl-tmp/DeepfakeBench/training/config/detector/b4_stat_ctrl_efficientnetb4_s2048.yaml > /root/autodl-tmp/DeepfakeBench/logs/b4ctrl_s2048_stdout.log 2>&1
echo "[b4-stat-chain] $(date +%Y-%m-%d_%H:%M:%S) B4-CTRL done"
echo "B4_STAT_CHAIN_DONE $(date +%Y-%m-%d_%H:%M:%S)"
