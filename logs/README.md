# 官方 cosine 自适应收集窗训练日志（表 4c 主口径）

来源：AutoDL `/root/autodl-tmp/train_*.log`。自适应窗 = `swa_start_lr_ratio: 0.4`（lr ≤ 40% lr_max 起收集，官方 6 轮实际平均 E3/E4/E5 三点）。

- `train_official_cos_spsa1swa_s1024_adapt.log` / `s2048_adapt.log` / `s4096_adapt.log` — 自适应 3 seed
- `train_official_cos_spsa1swa_s2048_adapt_rerun.log` — s2048 首跑后独立重跑（中途打断，弃用，保留痕迹）
- `train_official_cos_spsa1swa_s2048_adapt_rerun2.log` — s2048 收口独立重跑（7 点 0.7131，配 CTRL rerun 0.7080 → +0.51）

**s2048 配对口径 = 两次独立运行 vs 两次独立运行**：首跑 0.7009 vs CTRL 首跑 0.7115 = −1.07；rerun2 0.7131 vs CTRL rerun 0.7080 = +0.51 → 坑是评估随机落点、非 SPSA 系统性崩，论文如实呈现"一负一正 + 噪声地板分析"。
