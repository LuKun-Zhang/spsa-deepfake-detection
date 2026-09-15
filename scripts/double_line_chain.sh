#!/bin/bash
OUT=/root/autodl-tmp/double_line_out.log
# 等 c4096 训练结束
while pgrep -f '[t]raining/train.py' > /dev/null; do
  echo "[dl] $(date +%H:%M:%S) c4096 still training, sleep 120" >> $OUT
  sleep 120
done
echo "[dl] $(date +%H:%M:%S) train done, start double-line tests" >> $OUT
cd /root/autodl-tmp/DeepfakeBench
SPSA_DET=/root/autodl-tmp/stat_cfgs/stat_SPSA_test.yaml
CTRL_DET=/root/autodl-tmp/stat_cfgs/stat_ctrl_test.yaml
BASE=/root/autodl-tmp/DeepfakeBench/logs/training
PY=/root/miniconda3/bin/python

run_test() {
  local name=$1 det=$2 ds=$3 ck=$4
  $PY training/test.py --detector_path $det --test_dataset "$ds" --weights_path $ck > /root/autodl-tmp/test_dl_$name.log 2>&1
  echo "[dl] $(date +%H:%M:%S) $name ($ds) done" >> $OUT
}

# ===== 域内 FF++ (6) =====
run_test spsa_s1024_ffpp  $SPSA_DET "FaceForensics++" $BASE/ucf_2026-08-22-14-28-48/test/Celeb-DF-v2/ckpt_best.pth
run_test spsa_s2048_ffpp  $SPSA_DET "FaceForensics++" $BASE/ucf_2026-08-22-19-26-20/test/Celeb-DF-v2/ckpt_best.pth
run_test spsa_s4096_ffpp  $SPSA_DET "FaceForensics++" $BASE/ucf_2026-08-22-21-53-49/test/Celeb-DF-v2/ckpt_best.pth
run_test ctrl_c1024_ffpp  $CTRL_DET "FaceForensics++" $BASE/ucf_2026-08-22-17-10-08/test/Celeb-DF-v2/ckpt_best.pth
run_test ctrl_c2048_ffpp  $CTRL_DET "FaceForensics++" $BASE/ucf_2026-08-23-09-43-38/test/Celeb-DF-v2/ckpt_best.pth
run_test ctrl_c4096_ffpp  $CTRL_DET "FaceForensics++" $BASE/ucf_2026-08-23-11-56-09/test/Celeb-DF-v2/ckpt_best.pth

# ===== 跨域 DFDC (6) =====
run_test spsa_s1024_dfdc  $SPSA_DET "DFDC" $BASE/ucf_2026-08-22-14-28-48/test/Celeb-DF-v2/ckpt_best.pth
run_test spsa_s2048_dfdc  $SPSA_DET "DFDC" $BASE/ucf_2026-08-22-19-26-20/test/Celeb-DF-v2/ckpt_best.pth
run_test spsa_s4096_dfdc  $SPSA_DET "DFDC" $BASE/ucf_2026-08-22-21-53-49/test/Celeb-DF-v2/ckpt_best.pth
run_test ctrl_c1024_dfdc  $CTRL_DET "DFDC" $BASE/ucf_2026-08-22-17-10-08/test/Celeb-DF-v2/ckpt_best.pth
run_test ctrl_c2048_dfdc  $CTRL_DET "DFDC" $BASE/ucf_2026-08-23-09-43-38/test/Celeb-DF-v2/ckpt_best.pth
run_test ctrl_c4096_dfdc  $CTRL_DET "DFDC" $BASE/ucf_2026-08-23-11-56-09/test/Celeb-DF-v2/ckpt_best.pth

# ===== 跨域 DFDCP (6) =====
run_test spsa_s1024_dfdcp $SPSA_DET "DFDCP" $BASE/ucf_2026-08-22-14-28-48/test/Celeb-DF-v2/ckpt_best.pth
run_test spsa_s2048_dfdcp $SPSA_DET "DFDCP" $BASE/ucf_2026-08-22-19-26-20/test/Celeb-DF-v2/ckpt_best.pth
run_test spsa_s4096_dfdcp $SPSA_DET "DFDCP" $BASE/ucf_2026-08-22-21-53-49/test/Celeb-DF-v2/ckpt_best.pth
run_test ctrl_c1024_dfdcp $CTRL_DET "DFDCP" $BASE/ucf_2026-08-22-17-10-08/test/Celeb-DF-v2/ckpt_best.pth
run_test ctrl_c2048_dfdcp $CTRL_DET "DFDCP" $BASE/ucf_2026-08-23-09-43-38/test/Celeb-DF-v2/ckpt_best.pth
run_test ctrl_c4096_dfdcp $CTRL_DET "DFDCP" $BASE/ucf_2026-08-23-11-56-09/test/Celeb-DF-v2/ckpt_best.pth

echo "[dl] $(date +%H:%M:%S) DOUBLE_LINE_ALL_DONE" >> $OUT
