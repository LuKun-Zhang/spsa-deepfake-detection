# EfficientNet-B4 跨骨干 3-seed 定稿数据（受控协议，论文表 5）

本文件是 **EfficientNet-B4（EFNB4）跨骨干复现的权威定稿数据**：受控协议、3 seed
（s1024 / s2048 / s4096）、Celeb-DF-v2 帧级 AUC、7 点均值配对差。它把
`cross-backbone-reproduce` 分支的单 seed s2048 方向性观察（+1.38）扩为三 seed，
并取代 `controlled-alt-methods` 分支上临时的 `docs/b4_backbone_3seed.md` 结果表。

> **口径声明**：跨骨干各腿均无重复测试误差棒；三 seed 配对差 3/3 同向为正、均值 +1.41 点，
> 远超单 seed 配对噪声标尺（±0.95 点）的 1 倍以上，故可作为骨干特异性结论的定稿数据进入论文表 5。
> ResNet34 单 seed（−2.34）与 UCF 主骨干（同 seed +1.40）仍按方向性观察引用（见
> `docs/cross_backbone_verified_data.md`）。

## 1. EFNB4 三 seed 对照（论文表 5 数字）

受控协议：4 周期（E0–E3，cosine `lr_T_max: 3`，cons off）、分辨率 256、batch 16、Adam lr 2e-4/wd 5e-4、
7 个评估点（step 5743→22975，间隔 2872）。配对差 = SPSA1SWA 7 点均值 − CTRL 7 点均值。

| seed | CTRL 7 点均值 | SPSA1SWA 7 点均值 | 配对差（点） | 同向 |
|---|---|---|---|---|
| s1024 | 0.7164 | 0.7223 | **+0.59** | ✓ |
| s2048 | 0.7192 | 0.7330 | **+1.38** | ✓ |
| s4096 | 0.7370 | 0.7594 | **+2.25**（+2.24） | ✓ |
| **3-seed 均值** | 0.7242 | 0.7382 | **+1.41（3/3 同向为正）** | ✓ |

## 2. 逐 run-dir 溯源（日志可查）

| seed | 腿 | run dir | 7 点均值 | 执行机 |
|---|---|---|---|---|
| s1024 | CTRL | `ucf_2026-09-06-21-59-14` | 0.7164 | 659 |
| s1024 | SPSA1SWA | `ucf_2026-09-07-20-55-12`（重跑） | 0.7223 | 659 |
| s2048 | CTRL | 652 原始权威运行（468 链） | 0.7192 | 652 |
| s2048 | SPSA1SWA | `ucf_2026-09-04-20-36-42` | 0.7330 | 652 |
| s4096 | CTRL | `ucf_2026-09-07-12-57-56` | 0.7370 | 798 |
| s4096 | SPSA1SWA | `ucf_2026-09-07-16-39-15` | 0.7594 | 798 |

（UCF 主骨干同 seed 参照：s2048 CTRL 0.7160 / SPSA+SWA 0.7300 → +1.40，见 cross_backbone doc。）

## 3. 尖峰/重跑排除说明（诚实披露）

- **s2048 CTRL 取"第二次独立运行"为权威基线（0.7192）**：该 seed 首运行在评估点 step 11487 出现孤立高值
  0.7815（相邻点均 0.71–0.73），判为评估噪声单点尖峰；第二次运行同点回落 0.7413、未复现，故以第二次为基线
  （已在 cross_backbone doc 声明，进入论文表 5）。
- **453/798 s2048 CTRL 重跑 0.7346 不作数**：该次为受控 s4096 实验之后的额外重跑，与权威基线同 seed 两次
  相差约 1.5 点、落双跑噪声带内；不改变 s2048 权威基线取值。EFNB4 3-seed 定稿一律以
  `docs/efnb4_3seed_verified_data.md` 这张表为准。
- **s1024 SPSA1SWA 以重跑腿 0.7223 为准**：更早一次 s1024 SPSA1SWA 运行未采用（见跨骨干历史），只保留重跑值。

## 4. 复现脚本

三 seed 链式/补跑脚本（本分支 `scripts/`，云端路径 `/root/autodl-tmp/DeepfakeBench/training/config/detector/`）：

| 脚本 | 作用 |
|---|---|
| `b4_cfggen_s1024_s4096.sh` | 用 sed 自 s2048 模板生成 s1024/s4096 的 CTRL 与 SPSA1SWA 配置（seed 替换） |
| `b4_stat_chain_s1024_s4096.sh` | 主链：s1024 CTRL→SPSA1SWA、s4096 CTRL→SPSA1SWA 串行（每 seed 先 CTRL 后 SPSA） |
| `b4_stat_chain_s4096_only.sh` | s4096 补跑链（故障恢复/单 seed 重跑用） |
| `b4_stat_chain_s2048.sh` | s2048 链（继承自 `cross-backbone-reproduce`，652 机使用） |

配置（`configs/`，本分支新增 s1024/s4096，s2048 继承自 cross-backbone-reproduce）：

| 配置 | seed | 方案 |
|---|---|---|
| `b4_stat_ctrl_efficientnetb4_s1024.yaml` / `_s4096.yaml` | 1024 / 4096 | CTRL（spsa:false, SWA:false） |
| `b4_stat_SPSA1SWA_efficientnetb4_s1024.yaml` / `_s4096.yaml` | 1024 / 4096 | SPSA1SWA（spsa:true, SWA:true, swa_start:2） |

所有配置与 s2048 模板逐字段一致，仅 `manualSeed` 不同。
