# EMA 替代方法臂 — verified data（受控 4-cycle，纯 detector + EMA，3 seed）

> **状态：结果已回填（2026-09-10）** — 三 seed 链在 A05 全部干净收官，每个 run 的
> `training.log` 均含 `Final EMA testing done` marker：
> s1024 = `ucf_2026-09-09-11-29-41`、s2048 = `ucf_2026-09-09-15-30-59`、s4096 = `ucf_2026-09-09-19-27-30`。
> 实测 ~3.9h/run（比设定时预估的 ~4h 略快，s4096 于 09-09 23:23 收尾）。

## 实验设定（已定稿，可复现）

`configs/stat_ema_s{1024,2048,4096}.yaml`：受控 4-cycle（`nEpochs: 3`、cosine `lr_T_max: 3`、
Adam `lr 2e-4 / wd 5e-4`、batch 16、res 256、cons off），**纯 detector**（`spsa: false`, `SWA: false`）
+ **被动 EMA**（decay `0.999`，对全部 `requires_grad` 参数 = 354 个）。配对 CTRL = 同 seed 纯受控
`stat_ctrl_s*.yaml`；EMA 配置与其**仅差 `EMA: true` + `ema_decay: 0.999`**（`log_dir` 除外）——干净配对。

**测法（与论文 7pt 口径一致的关键）**：7 个测试点全部在 **EMA 权重**上评估并落 `testpoints/step_%d.pth`
（EMA arm 的 online trajectory 与 CTRL 相同，若测 online 权重则增益恒 ≡ 0、无意义；测 EMA 权重才反映
"实际部署的权重平均制品"）。train.py 另有 Final EMA 测试（镜像 Final SWA）。实现见 `code/trainer.py`
（`_ema_init`/`_ema_sd`/每步 EMA 更新/测试点 try-finally 换入换出）+ `code/train.py`；推导脚本
`scripts/patch_ema.py`、`scripts/gen_ema_yamls.py`、`scripts/chain_ema_a05.sh`。

## 结果（回填完毕 2026-09-10）

| seed | run dir | CTRL mean7 | EMA mean7 | 配对增益 |
|---|---|---|---|---|
| 1024 | `ucf_2026-09-09-11-29-41` | 0.7137 | 0.6802 | **−3.35** |
| 2048 | `ucf_2026-09-09-15-30-59` | 0.7224 | 0.6651 | **−5.73** |
| 4096 | `ucf_2026-09-09-19-27-30` | 0.7091 | 0.6989 | **−1.02** |
| mean | — | 0.7151 | 0.6814 | **−3.37** |

- CTRL 列 = `sam_verified_data.md` 链式 `stat_ctrl_s*.yaml` 3 seed（0.7137/0.7224/0.7091），与 SAM 臂同基准。
- 每 seed 7pt（dataset avg，step 5743→22975）+ EMA 终权重（step:0）：
  - s1024：0.5229/0.6857/0.7109/0.7131/0.7043/0.7125/0.7120；终 0.7120
  - s2048：0.5097/0.7281/0.6697/0.6604/0.7000/0.6920/0.6958；终 0.6958
  - s4096：0.6813/0.6544/0.7325/0.7166/0.6763/0.7133/0.7176；终 0.7176

**判读：EMA 3-seed 全部无增益，7pt 均值 −3.37 pt（全带外/边缘，±0.95 噪声带）。**
- 首点 lag 伪影：decay 0.999 → 有效窗 ~1000 步，EMA 在 ~5743 步仍被近 init 权重稀释
  （s1024/s2048 首点 0.52/0.51 崩落即此因；s4096 首点 0.68 较轻）。去首点后 6 点：
  s1024 0.7064（vs CTRL 0.7114 带内）、s2048 0.6910（仍负）、s4096 0.7018（vs CTRL 6pt 口径）。
- 部署口径（EMA 终权重 vs CTRL mean7）：s1024 −0.17 / s2048 −2.66 / s4096 +0.85。
- **结论与表 3 定位**：EMA 作为"持续平均聚合"对照臂，受控协议下无增益——与 SAM（+0.28）
  同列；"通用稳化/聚合类策略全无效、唯 SPSA 结构化约束有效"叙事再添一条 3-seed 完整证据。
- **入稿口径注**：7pt 测 EMA 权重 vs 主表 7pt 测 online 权重，非同口径 → 表 3 若列 EMA 须附
  lag 伪影注或以部署口径呈现。入稿形态仍待用户在 paper 层面拍板（对比范围已锚 DeepfakeBench 官方）。

> 结论对照：受控主表 SPSA+SWA（同配对 CTRL 基准）3-seed 增益为**显著正、出带**（见受控 2×2 主表
> 及强骨干 EFNB4 已定稿 +0.59/+1.38/+2.25 → mean **+1.41**，`cross_backbone_verified_data.md`/EFNB4 记录），
> EMA 为 **−3.37** —— 方向相反、量级均出带，互为反面证据。
