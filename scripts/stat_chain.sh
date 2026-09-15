#!/bin/bash
cd /root/autodl-tmp/DeepfakeBench
echo "[chain] start $(date)"
while :; do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  echo "[chain] wait gpu... used=${USED}MiB $(date +%H:%M:%S)"
  if [ "$USED" -lt 3000 ]; then break; fi
  sleep 120
done
echo "[chain] gpu free, begin $(date)"
SUMMARY=/root/autodl-tmp/stat_summary.txt
echo "exp cfg best_auc" > $SUMMARY
for exp in A1 A2 A3 E1a E1b; do
  case $exp in
    A*) CFG=stat_A.yaml ;;
    E1*) CFG=stat_E1.yaml ;;
  esac
  LOG=/root/autodl-tmp/stat_${exp}.log
  echo "[chain] RUN $exp ($CFG) $(date +%H:%M:%S)"
  /root/miniconda3/bin/python training/train.py --detector_path training/config/detector/$CFG --no-save_feat > $LOG 2>&1
  BEST=$(grep -E 'Each dataset best metric' -A2 $LOG | grep -oE 'auc=[0-9.]+' | head -1 | cut -d= -f2)
  echo "$exp $CFG $BEST" >> $SUMMARY
  echo "[chain] DONE $exp best_auc=$BEST $(date +%H:%M:%S)"
done
echo "[chain] ALL_DONE $(date)"
