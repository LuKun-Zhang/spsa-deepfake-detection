#!/bin/bash
OUT=/root/autodl-tmp/ctrlswa_out.log
cd /root/autodl-tmp/DeepfakeBench
PY=/root/miniconda3/bin/python

echo "[cs] $(date +%H:%M:%S) waiting for swa_cross tests to finish" >> $OUT
while pgrep -f '[t]esting/test.py' > /dev/null || pgrep -f '[s]wa_cross_chain.sh' > /dev/null; do
  sleep 60
done
echo "[cs] $(date +%H:%M:%S) cross tests done, start CTRL+SWA s1024" >> $OUT
nohup $PY training/train.py --detector_path /root/autodl-tmp/stat_cfgs/stat_CTRLSWA_s1024.yaml > /root/autodl-tmp/train_ctrlswa_s1024.log 2>&1 &
while pgrep -f '[t]raining/train.py' > /dev/null; do sleep 60; done
echo "[cs] $(date +%H:%M:%S) CTRL+SWA s1024 done, start s2048" >> $OUT
nohup $PY training/train.py --detector_path /root/autodl-tmp/stat_cfgs/stat_CTRLSWA_s2048.yaml > /root/autodl-tmp/train_ctrlswa_s2048.log 2>&1 &
while pgrep -f '[t]raining/train.py' > /dev/null; do sleep 60; done
echo "[cs] $(date +%H:%M:%S) CTRLSWA_ALL_DONE" >> $OUT
