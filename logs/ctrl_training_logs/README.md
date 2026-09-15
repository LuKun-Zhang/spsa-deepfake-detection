# 受控协议 12 组训练日志（training.log）

来源：AutoDL `/root/autodl-tmp/DeepfakeBench/logs/training/ucf_*/training.log`（2026-09-03 无卡归档）。
每个 training.log 头部含完整 config dump，正文含每测试点（step）帧级 auc/acc/eer/ap/video_auc 与当前 lr —— **7 点轨迹（图 4、表 1、2×2 归因）的唯一不可再生数据源**（受控 12 组未保存中间权重，仅 ckpt_best/swa，见 `../cross_domain/crossdomain_7ptmean_report_STOP_20260830.md`）。

## 目录 → 配置/seed 映射（受控协议：配置 nEpochs=3 → 实际 E0-E3 4 周期，cosine T_max=3，cons off）

| ucf 目录 | 配置 | seed |
|---|---|---|
| ucf_2026-08-22-17-10-08 | CTRL | 1024 |
| ucf_2026-08-23-09-43-38 | CTRL | 2048 |
| ucf_2026-08-23-11-56-09 | CTRL | 4096 |
| ucf_2026-08-22-14-28-48 | SPSA | 1024 |
| ucf_2026-08-22-19-26-20 | SPSA | 2048 |
| ucf_2026-08-22-21-53-49 | SPSA | 4096 |
| ucf_2026-08-24-13-04-19 | CTRL+SWA | 1024 |
| ucf_2026-08-24-15-22-15 | CTRL+SWA | 2048 |
| ucf_2026-08-29-02-08-19 | CTRL+SWA | 4096 |
| ucf_2026-08-23-20-35-40 | SPSA+SWA | 1024 |
| ucf_2026-08-24-09-59-38 | SPSA+SWA | 2048 |
| ucf_2026-08-28-23-38-12 | SPSA+SWA | 4096 |

## 数据遗失说明（2026-09-03 核对）

复现 s2048（受控 SPSA+SWA s2048 二次运行，目录 `ucf_2026-08-31-15-35-06`，表 2 SPSA 列 7 检查点权重源）**在云端已整体丢失**（2026-08-31 后该目录不存在；`find` 全盘与 `.Trash-0` 均无）。

- 影响：该 run 的 7×step_*.pth（180M）与其自身 training.log 不可再生。
- 补救：① UCF 侧 7 点轨迹可由同配置首跑 `ucf_2026-08-24-09-59-38`（本目录）替代；② 其跨域 eval 数值已固化于 `../cross_domain/cross_domain_results_7points.json` 与 `crossdomain_final_matrix_20260831.md` —— 表 2 数字仍可完整回源。
