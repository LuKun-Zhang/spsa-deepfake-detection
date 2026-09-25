#!/bin/bash
# Launch one mechanism-experiment leg on 871, detached from the ssh channel.
#
#   setsid bash /root/launch_mexp.sh mexp_p_spsa_l0
#
# cwd MUST be the repo root: train.py:250 opens './training/config/train_config.yaml',
# and preprocessing paths are root-relative too.  Running from training/ is the
# mistake that made the first smoke die instantly with FileNotFoundError.
#
# CUBLAS_WORKSPACE_CONFIG is required by the deterministic-algorithms setting
# that train.py:67 turns on.
NAME="$1"
if [ -z "$NAME" ]; then echo "usage: $0 <yaml-name>"; exit 2; fi
ROOT=/root/autodl-tmp/DeepfakeBench
YAML="$ROOT/training/config/detector/$NAME.yaml"
OUT=/autodl-fs/data/swa_exp/${NAME}_run.out

[ -f "$YAML" ] || { echo "no such config: $YAML"; exit 3; }
# a running leg owns the whole GPU (single 4090D, ~16 GB/leg): refuse to stack
if pgrep -f "[t]raining/train.py" > /dev/null; then
    echo "REFUSING: a train.py is already running"
    pgrep -af "[t]raining/train.py"
    exit 4
fi

cd "$ROOT" || exit 1
nohup env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    /root/miniconda3/bin/python training/train.py \
    --detector_path "./training/config/detector/$NAME.yaml" \
    > "$OUT" 2>&1 &
PID=$!
echo "launched $NAME  pid=$PID  log=$OUT"
sleep 20
if kill -0 "$PID" 2>/dev/null; then
    echo "alive after 20s; dir=$(ls -td /autodl-fs/data/swa_exp/*/ucf_* 2>/dev/null | head -1)"
else
    echo "DIED within 20s -- tail of $OUT:"
    tail -c 700 "$OUT"
    exit 5
fi
