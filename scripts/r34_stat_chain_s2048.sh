#!/bin/bash
# ResNet34 cross-backbone directionality run (controlled protocol, seed 2048, single seed)
# SPSA1SWA then CTRL. Chain in one GPU reservation.
set -e
cd /root/autodl-tmp/DeepfakeBench
PY=/root/miniconda3/bin/python
echo "[r34-stat-chain] $(date +%Y-%m-%d_%H:%M:%S) start R34-SPSA1SWA s2048 (controlled 4ep, seed2048)"
$PY training/train.py --detector_path /root/autodl-tmp/DeepfakeBench/training/config/detector/r34_stat_SPSA1SWA_resnet34_s2048.yaml > /root/autodl-tmp/DeepfakeBench/logs/r34spsa_s2048_stdout.log 2>&1
echo "[r34-stat-chain] $(date +%Y-%m-%d_%H:%M:%S) R34-SPSA1SWA done"
echo "[r34-stat-chain] $(date +%Y-%m-%d_%H:%M:%S) start R34-CTRL s2048"
$PY training/train.py --detector_path /root/autodl-tmp/DeepfakeBench/training/config/detector/r34_stat_ctrl_resnet34_s2048.yaml > /root/autodl-tmp/DeepfakeBench/logs/r34ctrl_s2048_stdout.log 2>&1
echo "[r34-stat-chain] $(date +%Y-%m-%d_%H:%M:%S) R34-CTRL done"
echo "R34_STAT_CHAIN_DONE $(date +%Y-%m-%d_%H:%M:%S)"
