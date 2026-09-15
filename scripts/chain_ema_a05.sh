#!/bin/bash
# EMA 3-seed chain on A05. seed1 (s1024) already running from 2026-09-09 11:29
# (run dir logs/training/ucf_2026-09-09-11-29-41). Wait its Final-EMA marker,
# then launch s2048, then s4096. Fully detached: survives client disconnect.
PY=/root/miniconda3/bin/python
BASE=/root/autodl-tmp/DeepfakeBench
LOG=$BASE/logs
cd "$BASE" || exit 9
log(){ echo "[chain $(date '+%m-%d %H:%M:%S')] $*"; }

wait_seed(){ # $1=yaml-stem  $2=training.log path
  local stem="$1" done="$2" i
  for i in $(seq 1 1200); do
    if grep -q 'Final EMA testing done' "$done" 2>/dev/null; then
      log "$stem DONE (marker found)"; return 0
    fi
    if [ "$i" -gt 4 ] && ! pgrep -f "$stem.yaml" >/dev/null 2>&1; then
      log "$stem ABORT: process gone before Final-EMA marker"; return 1
    fi
    sleep 30
  done
  log "$stem ABORT: 10h timeout"; return 1
}

# ---- seed1: already running ----
log "waiting seed1 (s1024) ..."
wait_seed s1024 "$LOG/training/ucf_2026-09-09-11-29-41/training.log" || { log "CHAIN_ABORT seed1"; exit 3; }

# ---- seed2: s2048 ----
log "launching seed2 (s2048)"
nohup $PY training/train.py --detector_path training/config/detector/stat_ema_s2048.yaml > "$LOG/ema_s2048_stdout.log" 2>&1 < /dev/null &
sleep 25
D2=$(grep -oE 'logs/training/ucf_[0-9-]+' "$LOG/ema_s2048_stdout.log" 2>/dev/null | tail -1 | sed 's#^\./##')
[ -n "$D2" ] || D2=$(ls -dt logs/training/ucf_* | head -1)
log "s2048 run dir: $D2"
wait_seed s2048 "$D2/training.log" || { log "CHAIN_ABORT seed2"; exit 4; }

# ---- seed3: s4096 ----
log "launching seed3 (s4096)"
nohup $PY training/train.py --detector_path training/config/detector/stat_ema_s4096.yaml > "$LOG/ema_s4096_stdout.log" 2>&1 < /dev/null &
sleep 25
D3=$(grep -oE 'logs/training/ucf_[0-9-]+' "$LOG/ema_s4096_stdout.log" 2>/dev/null | tail -1 | sed 's#^\./##')
[ -n "$D3" ] || D3=$(ls -dt logs/training/ucf_* | head -1)
log "s4096 run dir: $D3"
wait_seed s4096 "$D3/training.log" || { log "CHAIN_ABORT seed3"; exit 5; }

log "CHAIN_ALL_DONE"
