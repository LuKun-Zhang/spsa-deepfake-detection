#!/bin/bash
OUT=/root/autodl-tmp/swa_cross_out.log
cd /root/autodl-tmp/DeepfakeBench
SPSA_DET=/root/autodl-tmp/stat_cfgs/stat_SPSA_test.yaml
BASE=/root/autodl-tmp/DeepfakeBench/logs/training
PY=/root/miniconda3/bin/python

run_test() {
  local name=$1 ds=$2 ck=$3
  $PY training/test.py --detector_path $SPSA_DET --test_dataset "$ds" --weights_path $ck > /root/autodl-tmp/test_swa_$name.log 2>&1
  echo "[swa] $(date +%H:%M:%S) $name ($ds) done" >> $OUT
}

S1024=$BASE/ucf_2026-08-23-20-35-40
S2048=$BASE/ucf_2026-08-24-09-59-38

# ===== s1024 (4) =====
run_test s1024_dfdc_best  "DFDC"  $S1024/test/Celeb-DF-v2/ckpt_best.pth
run_test s1024_dfdc_swa   "DFDC"  $S1024/swa.pth
run_test s1024_dfdcp_best "DFDCP" $S1024/test/Celeb-DF-v2/ckpt_best.pth
run_test s1024_dfdcp_swa  "DFDCP" $S1024/swa.pth

# ===== s2048 (4) =====
run_test s2048_dfdc_best  "DFDC"  $S2048/test/Celeb-DF-v2/ckpt_best.pth
run_test s2048_dfdc_swa   "DFDC"  $S2048/swa.pth
run_test s2048_dfdcp_best "DFDCP" $S2048/test/Celeb-DF-v2/ckpt_best.pth
run_test s2048_dfdcp_swa  "DFDCP" $S2048/swa.pth

echo "[swa] $(date +%H:%M:%S) SWA_CROSS_ALL_DONE" >> $OUT
