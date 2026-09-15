#!/bin/bash
# 官方完整协议（cosine 6 周期）+ 自适应 SWA 窗 SPSA1SWA vs CTRL 全量链式训练脚本
# 6 个训练：SPSA1SWA(自适应) s1024/s2048/s4096 + CTRL s1024/s2048/s4096
# 用法：在 DeepfakeBench 根目录执行（参考 README 环境说明）
set -e

# 云端实测环境：AutoDL RTX 5090 32G，miniconda Python 3.12
# 按本地环境调整：PY=python（conda env）、DETECTOR_PATH=绝对/相对路径
PY=python
ROOT=$(pwd)
CFG_DIR=$ROOT/configs

run() {
    local tag=$1 cfg=$2
    echo "[official-cos-adapt-chain] $(date +%H:%M:%S) start $tag"
    $PY training/train.py --detector_path "$cfg" > "$ROOT/train_official_cos_adapt_$tag.log" 2>&1
    echo "[official-cos-adapt-chain] $(date +%H:%M:%S) done $tag"
}

echo "[official-cos-adapt-chain] $(date +%H:%M:%S) ========== 官方完整协议 + 自适应SWA窗 6 轮 cosine =========="

run spsa1swa_s1024_adapt "$CFG_DIR/official_cos_SPSA1SWA_s1024_adapt.yaml"
run spsa1swa_s2048_adapt "$CFG_DIR/official_cos_SPSA1SWA_s2048_adapt.yaml"
run spsa1swa_s4096_adapt "$CFG_DIR/official_cos_SPSA1SWA_s4096_adapt.yaml"
run ctrl_s1024          "$CFG_DIR/official_cos_CTRL_s1024.yaml"
run ctrl_s2048          "$CFG_DIR/official_cos_CTRL_s2048.yaml"
run ctrl_s4096          "$CFG_DIR/official_cos_CTRL_s4096.yaml"

echo "[official-cos-adapt-chain] $(date +%H:%M:%S) ========== ALL DONE =========="
