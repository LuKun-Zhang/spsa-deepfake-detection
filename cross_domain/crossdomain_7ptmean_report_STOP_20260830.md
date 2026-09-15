# crossdomain_7ptmean_report.md — 跨域均值评估执行报告

> 执行方：云端实例 LLM（AutoDL，2026-08-30 夜）
> 交办方：SPSA 论文主助手
> 状态：**STOP（触发任务书 §2 硬性停止条件）**，未跑全量评估

---

## 0. 一句话执行摘要

12/12 受控模型均**无 7 个中间检查点**（只有 ckpt_best.pth / swa.pth），跨域 7 点均值在现有数据下无法执行，按任务书 §2 硬性停止条件（缺失模型 ≥3 → 停止）**立即停止**，未运行任何全量评估。环境核验（步骤 0）与 12 模型定位（步骤 1）已完成，评估命令已恢复。

## 1. 停止原因（硬性停止条件触发）

任务书 §2：`找不到 7 检查点的模型数 ≥ 3 个 → 停止，回传缺失清单（证据不完整，全量无意义）`

**实际缺失：12/12 个受控模型全部缺少 7 个中间检查点。** 每个模型目录仅有：
- `test/Celeb-DF-v2/ckpt_best.pth`（+ `test/avg/ckpt_best.pth`）——训练中每验证点**覆盖式保存**，最终只剩 best 权重
- SWA 配置另有 `swa.pth`（SWA 平均权重）

## 2. 根本原因分析

1. **训练代码从不按步落盘中间权重**：`trainer.py` L133-147 `save_ckpt()` 固定写 `ckpt_best.pth`（覆盖式，`ckpt_info` 仅日志描述）；L394-395 每半 epoch 验证后调用，即 7 个验证步（5743/8615/11487/14359/17231/20103/22975）的权重全部被覆盖为同一个文件，**从未保存 7 份独立权重**。
2. **按 epoch 存快照是后加的补丁**：`train.py` L345-349 的 `ckpt_ep{epoch}.pth` 保存逻辑属于 `[night_run 补丁 03/03]`（8-30 上线），受控主实验（8-22~8-29 训练）无此逻辑；全盘仅 8-30 的 `ucf_stat_*`（GaussSWA/SPSAOFF 系列）存在 `ckpt_ep{0..3}.pth`，且最多 4 个（按 epoch 非按 7 点步）。
3. **Celeb-DF 上的"7 点轨迹"来自训练时验证日志**（如 CTRL s1024 `ucf_2026-08-22-17-10-08/training.log`：step 5743 → auc=0.692713 与权威真值 0.6927 一致），权重当时未落盘，**事后无法重建**。
4. **结论**：跨域（DFDC/DFDCP/FF++）7 点均值需要 7 个中间权重逐一推理，权重不存在 → 任务不可执行。

## 3. 环境信息（步骤 0）

- 主机：`autodl-container-zzdbp6twc2-9f0ded93`（新实例）
- GPU：NVIDIA GeForce RTX 4080 SUPER，显存 32760 MiB（32GB），执行时 0% util / 1 MiB used（空闲）
- 代码：`/root/autodl-tmp/DeepfakeBench`
- Python：`/root/miniconda3/bin/python`
- 历史评估日志：`/root/autodl-tmp/test_dfdc_dfdcp.log`、`test_cross_A2.log`、`test_cross_B.log`（均为输出日志，无命令头）
- 评估命令（已恢复，来自 `launch_test_dfdc.py`）：
  ```
  cd /root/autodl-tmp/DeepfakeBench && /root/miniconda3/bin/python training/test.py \
    --detector_path training/config/detector/ucf.yaml \
    --test_dataset <DFDC|DFDCP|FaceForensics++|Celeb-DF-v2> --weights_path <ckpt.pth>
  ```

## 4. 数据环境状态

| 测试集 | 路径（datasets/rgb 下） | 状态 |
|---|---|---|
| Celeb-DF-v2 | `DeepfakeBench/datasets/rgb/Celeb-DF-v2` | ✅ 已解压可用 |
| DFDC | `DeepfakeBench/datasets/rgb/DFDC` | ✅ 已解压可用 |
| DFDCP | `DeepfakeBench/datasets/rgb/DFDCP` | ✅ 已解压可用 |
| FF++ c23 | `DeepfakeBench/datasets/rgb/FaceForensics++` | ✅ 已解压可用（test 划分待评估代码确认） |
| FF++ c40 | 未解压（FaceForensics++.zip） | ⚠️ 按任务书闸门判"不可用"，未解压 |

> 注：`/root/autodl-tmp/DFDC`、`/root/autodl-tmp/DFDCP` 亦存在（含 method_A/method_B 结构，供 test.py 直接读取；与 rgb 目录并存）。

## 5. 12 模型分类表（步骤 1）

所有 12 个目录的 `training.log` 头部 Configuration dump 均已确认受控协议（nEpochs=3 起 epoch0 共 4 次、cons off、Adam lr=2e-4、cosine T_max=3、batch 16、256×256、manualSeed 1024/2048/4096）：

| 配置 | seed | 训练目录 | 检查点文件 |
|---|---|---|---|
| CTRL | 1024 | ucf_2026-08-22-17-10-08 | 仅 ckpt_best.pth(×2) |
| CTRL | 2048 | ucf_2026-08-23-09-43-38 | 仅 ckpt_best.pth(×2) |
| CTRL | 4096 | ucf_2026-08-23-11-56-09 | 仅 ckpt_best.pth(×2) |
| SPSA | 1024 | ucf_2026-08-22-14-28-48 | 仅 ckpt_best.pth(×2) |
| SPSA | 2048 | ucf_2026-08-22-19-26-20 | 仅 ckpt_best.pth(×2) |
| SPSA | 4096 | ucf_2026-08-22-21-53-49 | 仅 ckpt_best.pth(×2) |
| CTRL+SWA | 1024 | ucf_2026-08-24-13-04-19 | ckpt_best.pth(×2) + swa.pth |
| CTRL+SWA | 2048 | ucf_2026-08-24-15-22-15 | ckpt_best.pth(×2) + swa.pth |
| CTRL+SWA | 4096 | ucf_2026-08-29-02-08-19 | ckpt_best.pth(×2) + swa.pth |
| SPSA+SWA | 1024 | ucf_2026-08-23-20-35-40 | ckpt_best.pth(×2) + swa.pth + swa_fixed.pth |
| SPSA+SWA | 2048 | ucf_2026-08-24-09-59-38 | ckpt_best.pth(×2) + swa.pth + swa_fixed.pth |
| SPSA+SWA | 4096 | ucf_2026-08-28-23-38-12 | ckpt_best.pth(×2) + swa.pth |

**缺失清单：12/12 模型缺失 7 个中间检查点（E0末/E1中/E1末/E2中/E2末/E3中/E3末 步号 5743/8615/11487/14359/17231/20103/22975）。**

## 6. 未执行的步骤与说明

| 步骤 | 状态 | 说明 |
|---|---|---|
| 步骤 2 热身跑（DFDC 计时） | 未执行 | 因 7 检查点缺失整体停止；未浪费 GPU |
| 步骤 3 锚点核验（Celeb-DF） | 未执行 | 同上；Celeb-DF 真值 [0.6927, 0.7520, 0.6952, 0.7314, 0.6926, 0.7104, 0.7112] 与训练日志 step 5743 auc=0.692713 等逐点可对上，证明 7 点数据来源无误 |
| 步骤 4-6 全量/聚合 | 未执行 | — |
| 表 A/B/C/D | 未产出 | 全部置"未完成"（诚实原则，禁止编造数字） |

## 7. 后续可选方向（供主助手/用户决策）

1. **改训练代码 + 重训 12 模型**：在现有 `ckpt_ep{epoch}.pth` 基础上增加"每 2872 步存 ckpt_step{step}.pth"，重训全部 12 模型。粗估 12 × ~2.5h ≈ 30h，超出本任务 9-10h 预算，需另排时段。
2. **降级交付（推荐过渡）**：用现有 `ckpt_best.pth` + `swa.pth` 在 DFDC/DFDCP/FF++ c23 上做 **best + SWA-Final 双口径**评估（含 FF++ 同域数字补齐，约 2-4h）。与主表 7 点均值口径仍不一致，但可先填充表 2 结构并暴露问题。
3. **最小重训**：仅重训论文表 2 涉及的 CTRL / SPSA / CTRL+SWA / SPSA+SWA 各 1-2 个 seed（带 7 点检查点保存），其余沿用现有数据，表 2 降级为部分 7 点均值。

## 8. 诚实声明

- 未编造任何数字；表 A/B/C/D 全部未产出（无数据可聚合）。
- 现有表 2 的 best 单点值（DFDC：SPSA+SWA 0.7161/SPSA 0.7133/CTRL 0.7094；DFDCP：SPSA+SWA 0.7233/SPSA 0.7136/CTRL 0.7142）为历史 best 口径结果，本任务未复测。
- 若后续决定跑"降级双口径"，可直接复用本报告的 12 模型目录表与恢复的评估命令。
