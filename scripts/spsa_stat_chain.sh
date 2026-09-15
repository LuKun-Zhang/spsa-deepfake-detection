#!/bin/bash
cd /root/autodl-tmp/DeepfakeBench
while pgrep -f '[t]raining/train.py' > /dev/null; do
  echo "[wait] $(date +%H:%M:%S) train.py running, sleep 120"
  sleep 120
done
echo "[chain] $(date +%H:%M:%S) GPU free, start s2048 SPSA"
/root/miniconda3/bin/python training/train.py --detector_path /root/autodl-tmp/stat_cfgs/stat_SPSA_s2048.yaml > /root/autodl-tmp/train_spsa_s2048.log 2>&1
echo "[chain] $(date +%H:%M:%S) s2048 SPSA done, start s4096 SPSA"
/root/miniconda3/bin/python training/train.py --detector_path /root/autodl-tmp/stat_cfgs/stat_SPSA_s4096.yaml > /root/autodl-tmp/train_spsa_s4096.log 2>&1
echo "CHAIN_SPSA_DONE $(date +%H:%M:%S)"
