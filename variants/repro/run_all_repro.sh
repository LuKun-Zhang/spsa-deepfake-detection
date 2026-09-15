#!/bin/bash
# run_all_repro.sh — 串行重训 4 模型（SPSA+SWA 优先），带 CUBLAS 确定性配置
cd /root/autodl-tmp/DeepfakeBench || exit 1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/root/miniconda3/bin/python
run() {
  local name=$1
  local yaml=$2
  echo "===== START $name $(date '+%F %T') ====="
  $PY training/train.py --detector_path training/config/detector/$yaml > /root/autodl-tmp/repro_${name}.log 2>&1
  echo "===== END $name rc=$? $(date '+%F %T') ====="
}
run spsaswa_s1024 ucf_repro_spsaswa_s1024.yaml
run spsaswa_s2048 ucf_repro_spsaswa_s2048.yaml
run ctrl_s1024 ucf_repro_ctrl_s1024.yaml
run ctrl_s2048 ucf_repro_ctrl_s2048.yaml
echo "ALL DONE $(date '+%F %T')"
