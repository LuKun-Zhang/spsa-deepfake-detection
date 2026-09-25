#!/bin/bash
# Launch the four strong-backbone legs on 871, detached from the ssh channel.
#
#   setsid bash /root/launch_cross_bb.sh            # all four legs, chained
#   setsid bash /root/launch_cross_bb.sh cnx_ctrl   # one leg by short name
#
# cwd MUST be the repo root: train.py:250 opens './training/config/train_config.yaml'
# and preprocessing paths are root-relative.  Running from training/ is the mistake
# that made the first mexp smoke die instantly with FileNotFoundError.
#
# CUBLAS_WORKSPACE_CONFIG is required by the deterministic-algorithms setting train.py
# turns on.  HF_ENDPOINT is required because huggingface.co is unreachable from 871;
# the timm weights for both new backbones are already in ~/.cache/huggingface/hub,
# but a cold cache must not silently hang the leg.
set -u

PY=/root/miniconda3/bin/python
ROOT=/root/autodl-tmp/DeepfakeBench
LOGDIR=/root/cross_bb_out          # system disk: data disk is 94% full (7.9 G free)
CHAIN_LOG="$LOGDIR/_chain.log"

export HF_ENDPOINT=https://hf-mirror.com
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p "$LOGDIR"
cd "$ROOT" || exit 1

ALL="cnx_ctrl cnx_spsaswa swv2_ctrl swv2_spsaswa"
yaml_for () {
    case "$1" in
        cnx_ctrl)     echo cnx_stat_ctrl_convnext_tiny_s2048 ;;
        cnx_spsaswa)  echo cnx_stat_SPSA1SWA_convnext_tiny_s2048 ;;
        swv2_ctrl)    echo swv2_stat_ctrl_swinv2_tiny_s2048 ;;
        swv2_spsaswa) echo swv2_stat_SPSA1SWA_swinv2_tiny_s2048 ;;
        *) return 1 ;;
    esac
}

LEGS="${*:-$ALL}"
log () { echo "$(date '+%F %T') $*" | tee -a "$CHAIN_LOG"; }

# --- preconditions ---------------------------------------------------------
for short in $LEGS; do
    NAME=$(yaml_for "$short") || { log "FATAL: unknown leg '$short'"; exit 2; }
    F="$ROOT/training/config/detector/$NAME.yaml"
    [ -f "$F" ] || { log "FATAL: missing $F -- run deploy.py first"; exit 3; }
    grep -q "rng_neutral_build: true" "$F" || {
        log "FATAL: $NAME.yaml lacks 'rng_neutral_build: true'."
        log "       Without it the two arms draw DIFFERENT base seeds, so the paired"
        log "       CTRL-vs-SPSA/SWA difference is confounded with a shuffle-order"
        log "       difference and the leg cannot support the claim it was run for."
        exit 4
    }
done
grep -q "ConvNeXtTiny" "$ROOT/training/networks/__init__.py" || {
    log "FATAL: training/networks/__init__.py does not register the new backbones --"
    log "       run deploy.py first"; exit 5; }
log "code guard ok"

# Only one leg owns the GPU.  A smoke or a leftover job must finish first.
while pgrep -f "[t]raining/train.py" > /dev/null; do
    log "GPU busy; waiting for the running job to finish"
    sleep 120
done

RC=0
for short in $LEGS; do
    NAME=$(yaml_for "$short")
    OUT="$LOGDIR/${short}.out"
    log "--- leg $short ($NAME) ---"
    nohup "$PY" training/train.py \
        --detector_path "./training/config/detector/$NAME.yaml" \
        > "$OUT" 2>&1 &
    PID=$!
    log "launched pid=$PID log=$OUT"
    sleep 60
    if ! kill -0 "$PID" 2>/dev/null; then
        log "DIED within 60s -- tail of $OUT:"
        tail -c 900 "$OUT" | tee -a "$CHAIN_LOG"
        log "stopping the chain"
        exit 6
    fi
    wait "$PID"; RC=$?
    log "leg $short exited rc=$RC"
    if [ $RC -ne 0 ]; then
        log "FATAL: leg $short rc=$RC -- stopping the chain"
        exit $RC
    fi
done
log "=== all legs done: $LEGS ==="
