#!/bin/bash
# Launch one R34 mechanism leg on A01, detached from the ssh channel.
#
#   setsid bash /root/launch_r34m.sh r34m_spsa1swa_s2048
#
# cwd MUST be the repo root: train.py opens './training/config/train_config.yaml'
# and the preprocessing paths are root-relative too.
# CUBLAS_WORKSPACE_CONFIG is required by the deterministic-algorithms setting
# that train.py turns on together with cudnn: false.
#
# The run is wrapped in `bash -c` so that an exit-status sentinel ($OUT.rc) is
# written when the training process ends, whatever the reason.  Without it
# "finished" and "crashed at hour 3" look identical from the outside -- the log
# marker only appears at the very end, so a leg that dies mid-run leaves a log
# that is neither complete nor obviously broken.  watch_r34m.py keys off this
# file so that a crash is recoverable instead of silently skipped.
NAME="$1"
if [ -z "$NAME" ]; then echo "usage: $0 <yaml-name>"; exit 2; fi
ROOT=/root/autodl-tmp/DeepfakeBench
YAML="$ROOT/training/config/detector/$NAME.yaml"
OUT=/autodl-fs/data/r34m/${NAME}_run.out

[ -f "$YAML" ] || { echo "no such config: $YAML"; exit 3; }
# one leg owns the whole GPU: refuse to stack
if pgrep -f "[t]raining/train.py" > /dev/null; then
    echo "REFUSING: a train.py is already running"
    pgrep -af "[t]raining/train.py"
    exit 4
fi

cd "$ROOT" || exit 1
rm -f "$OUT.rc"
setsid nohup bash -c "CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    /root/miniconda3/bin/python training/train.py \
    --detector_path './training/config/detector/$NAME.yaml' \
    > '$OUT' 2>&1; echo \$? > '$OUT.rc'" \
    > /dev/null 2>&1 < /dev/null &
echo "launched $NAME  log=$OUT  sentinel=$OUT.rc"
sleep 30
if pgrep -f "[t]raining/train.py" > /dev/null; then
    echo "alive after 30s; newest run dir:"
    ls -td /autodl-fs/data/r34m/logs/*/ 2>/dev/null | head -1
else
    echo "DIED within 30s -- tail of $OUT:"
    tail -c 1200 "$OUT"
    exit 5
fi
