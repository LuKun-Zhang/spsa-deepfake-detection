# 变体 / 复现 / 诊断实验日志（受控协议）

来源：第二台 AutoDL 实例 bjb1（`connect.bjb1.seetacloud.com`，2026-09-03 无卡归档）。
路径：`/root/autodl-tmp/DeepfakeBench/logs/training/`。第一台 bjb2 **无**这批实验。
均为受控协议（nEpochs=3 → E0-E3 4 周期）下的补充实验，training.log 头部含完整 config dump。

## GaussSWA（高斯扰动 SWA 对照）
| 目录 | 说明 |
|---|---|
| ucf_stat_GaussSWA_s1024_2026-08-30-00-07-10 | s1024 首跑（48K） |
| ucf_stat_GaussSWA_s1024_2026-08-30-09-56-01 | s1024 二次（62K，完整 ckpt_ep0-3 + swa） |
| ucf_stat_GaussSWA_s2048_2026-08-30-13-50-54 | s2048（61K） |

## SPSAOFF（SPSA 关闭对照）
| 目录 | 说明 |
|---|---|
| ucf_stat_SPSAOFF_s1024_2026-08-30-05-04-49 | s1024（54K，ckpt_ep0-2） |

## repro（2026-08-31 受控复现，保存 step/epoch 检查点）
| 目录 | 配置 | 说明 |
|---|---|---|
| ucf_2026-08-31-00-06-12 | SPSA+SWA s1024 | rc=137 被打断（5.5K） |
| ucf_2026-08-31-00-17-53 | SPSA+SWA s2048 | 完整：7×ckpt_step + ep0-3 + swa/swa_fixed |
| ucf_2026-08-31-04-41-58 | CTRL s1024 | 完整：7×ckpt_step + ep0-3 |
| ucf_2026-08-31-08-35-12 | CTRL s2048 | 34K，中断（ckpt_step5743/8615/11487） |

顶层 `repro_*.log` / `run_all_repro.sh`：repro 批次启动、kill+resume 监控 stdout。

> ⚠️ 这批 8-31 复现的 UCF 7 点值 **≠** `../docs/authority_7point_benchmark.md` 的"5090 复现"列——
> 那列的数据源是已丢失的 `ucf_2026-08-31-15-35-06`（见 `../logs/ctrl_training_logs/README.md`）。
> 本批为同 GPU/seed 复现噪声验证（GPU 假说旁证），非表 2 数据源。

## diag（logits / swa-fixed 诊断，见 spsa1_ctrl_verified_data.md 第十一节）
- `diag_v1_nosw.log` / `diag_v1b.log` — 无 SWA 基线诊断
- `diag_v2_seedfix.log` / `diag_v3_seedfix.log` — seed 修正重测
- `diag_auth_swafixed_celeb.log` / `diag_repro5743_celeb.log` / `diag_repro_swafixed_celeb.log` — 权威/repro 权重在 Celeb-DF-v2 的 logits 诊断
