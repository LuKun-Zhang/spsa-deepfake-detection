#!/bin/bash
set -e
cd /root/autodl-tmp/DeepfakeBench/training/config/detector
for S in 1024 4096; do
  sed "s/2048/${S}/g" b4_stat_ctrl_efficientnetb4_s2048.yaml   > b4_stat_ctrl_efficientnetb4_s${S}.yaml
  sed "s/2048/${S}/g" b4_stat_SPSA1SWA_efficientnetb4_s2048.yaml > b4_stat_SPSA1SWA_efficientnetb4_s${S}.yaml
done
echo "== generated manualSeed =="
grep -H "^manualSeed:" b4_stat_ctrl_efficientnetb4_s1024.yaml b4_stat_SPSA1SWA_efficientnetb4_s1024.yaml b4_stat_ctrl_efficientnetb4_s4096.yaml b4_stat_SPSA1SWA_efficientnetb4_s4096.yaml
echo "== files =="; ls -1 b4_stat_*_s1024.yaml b4_stat_*_s4096.yaml 2>/dev/null
