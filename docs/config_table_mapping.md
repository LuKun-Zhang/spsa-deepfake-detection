# 论文表格 ↔ 配置 ↔ 分支 ↔ verified-data 映射（ICASSP2027 提交口径）

> 目的：让审稿人/复现者从「论文某张表 / 某个数字」出发，直接定位到产生它的 config、所在分支、
> 以及 verified-data。仓库按"协议/实验轴"分层为多个分支；本文件是全局索引。

**指标口径（全仓库统一）**：7-point mean = 7 个测试点的 Celeb-DF-v2 帧级 AUC 均值，测试点步号
5743 / 8615 / 11487 / 14359 / 17231 / 20103 / 22975（共约 22976 iters）。论文全部表格均用该口径
（不是 best-checkpoint），因此与以 best 为口径的已发表 SOTA 不可比——这是论文论据的一部分。

**噪声口径**：±0.95（×1e-2）为受控协议的 seed 噪声 σ_mean 量级；论文凡"增益"均指 7pt-mean 的
同 seed 配对差（SPSA1SWA/SAM/EMA − 配对 CTRL）。跨 3 seed 报告 mean±区间，不作单 seed 显著性声明。

| 论文表 / 实验 | configs | 所在分支 | verified-data |
|---|---|---|---|
| 受控 2×2 消融（SPSA1SWA / CTRLSWA / SPSA / E / 纯 CTRL） | `configs/stat_{SPSA1SWA,CTRLSWA,SPSA,E,ctrl}_s*.yaml` | `main` | `docs/authority_7point_benchmark.md`（main） |
| 官方协议自适应窗（主口径 +1.02） | `configs/official_cos_*_adapt.yaml`（s1024/2048/4096） | `adaptive-window` | 该分支 README + verified 段 |
| 官方协议固定窄窗（对照版） | `configs/official_cos_*.yaml` | `official-cosine` | 该分支 README |
| **SAM 替代方法臂**（受控，纯 detector+SAM，mean +0.28） | `configs/stat_sam_s{1024,2048,4096}.yaml` | **本分支 `controlled-alt-methods`** | `docs/sam_verified_data.md` |
| **EMA 替代方法臂**（受控，纯 detector+EMA，结果见 doc） | `configs/stat_ema_s{1024,2048,4096}.yaml` | **本分支** | `docs/ema_verified_data.md` |
| **纯 CTRL 4-cycle 三 seed** | `configs/stat_ctrl_s{1024,2048,4096}.yaml` | 1024 在本分支；2048/4096 同时存于 `main` | `docs/sam_verified_data.md`（CTRL 列） |
| OOD/跨域 4 域 3-seed（+1.00） | `cross_domain/`（main） | `main` | `cross_domain/crossdomain_final_matrix_20260831.md` |
| 跨骨干方向性（B4 复现 / R34 未复现） | `configs/b4_*/r34_*_s2048.yaml` | `cross-backbone-reproduce`（父分支） | `docs/cross_backbone_verified_data.md` |
| **跨骨干 B4 补 seed → 3-seed 定稿（mean +1.41）** | `configs/b4_stat_*_efficientnetb4_s{1024,2048,4096}.yaml` | **本分支**（s1024/s4096 新增；s2048 继承父分支） | `docs/b4_backbone_3seed.md` |

## 复现代码覆盖（overlay 到 DeepfakeBench）

把本分支 `code/` 下各文件按路径覆盖到 DeepfakeBench `training/` 对应位置（父分支 README 有对照表）。
**本分支新增必须一并放置的两个 optimizer 模块**（`code/train.py` 顶层无条件 import 它们）：

| 仓库文件 | 目标位置 | 说明 |
|---|---|---|
| `code/optimizor/SAM.py` | `training/optimizor/SAM.py` | SAM（Sharpness-Aware Minimization, adam-base, rho=0.05）；此前仓库缺此文件，SAM/基线 import 均依赖它 |
| `code/optimizor/LinearLR.py` | `training/optimizor/LinearLR.py` | `LinearDecayLR`（DFB 上游即需） |
| `code/trainer.py` `code/train.py` | `training/trainer/trainer.py` `training/train.py` | 本分支在父分支之上加入 EMA 支持（`_ema_init`/`_ema_sd`/每步 EMA 更新/测试点与 Final 均测 EMA 权重） |

EMA 臂的推导脚本存于 `scripts/`（`patch_ema.py` 生成逐处补丁、`gen_ema_yamls.py` 从 stat_sam 模板生成三 seed EMA yaml、`chain_ema_a05.sh` 为三 seed 链式启动），供审计：configs/stat_ema_*.yaml 与 code 里的 EMA 改动即由它们产出。

## 受控协议定义（供各表共用）

受控 4-cycle：`nEpochs: 3`（epoch 0–3）、cosine `lr_T_max: 3`、Adam `lr 2e-4 / wd 5e-4`、batch 16、
分辨率 256、cons 关、FF++ c23 → Celeb-DF-v2。官方协议为 6 周期（epoch 0–5），见 `official-*` 分支。
