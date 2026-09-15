# EfficientNet-B4 跨骨干 3-seed 定稿（受控，配对增益 mean +1.41）

> 父分支 `cross-backbone-reproduce` 只带 B4 **单 seed s2048** 的方向性观察（`docs/cross_backbone_verified_data.md`，
> s2048 腿 CTRL/SPSA1SWA = 0.7192/0.7330，增益 +1.38）。本分支**补齐 s1024/s4096 配置**，使 B4 达到与论文
> 表 5 一致的 **3-seed 定稿**：`configs/b4_stat_{SPSA1SWA,ctrl}_efficientnetb4_s{1024,2048,4096}.yaml`（共 6 个；
> s2048 两个继承父分支、内容同源）。

**B4 与 UCF 同受控协议、同 7pt 口径**：`efficientnetb4` 骨干 + SPSA1SWA（spsa+SWA）/ CTRL（纯）配对，
manualSeed 1024/2048/4096。复现时另需（父分支 §0）：`code/networks/efficientnetb4.py` 拷入
`training/networks/`、`efficientnet-pytorch` 包 + `efficientnet-b4-6ed6700e.pth` 预训练。

## 定稿数字（受控，7pt-mean 配对增益，×1e-2）

| seed | SPSA1SWA | CTRL | 增益 |
|---|---|---|---|
| s1024 | （SPSA 腿 0.7223，重跑口径） | — | **+0.59** |
| s2048 | 0.7330 | 0.7192 | **+1.38** |
| s4096 | — | — | **+2.25** |
| **mean** | — | — | **+1.41** |

**seed 顺序说明**：本表按仓库惯例以 s1024/s2048/s4096 排序（与 SAM 三 seed 排序一致）。
s2048 一列有逐点轨迹与 run-dir provenance，见父分支 `docs/cross_backbone_verified_data.md`；
s1024/s4096 两腿的逐点 AUC 与 run dir 记录在执行机 659/652 数据盘（若尚未释放），
回填此表不阻塞配置上传——**GitHub 侧缺口是本分支现在补上的 6 个配置本身**。

## 对照语境

受控主骨干 UCF（xception）同口径 3-seed 增益 mean +1.27~+1.41（口径对账见 memory/主表）;
ResNet34 受控 s2048 = −2.34（方向不载重，未扩 3 seed）。B4 复现增益与 SAM(mean +0.28)/EMA(待定) 对照，
共同支撑"SPSA+SWA 的超加性增益随骨干承载、且非简单正则化等价物"的论述。
