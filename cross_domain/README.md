# 表 2 跨域数据（受控 SPSA+SWA s2048 × 5 数据集）

论文表 2：跨域 5 数据集（Celeb-DF-v2 / DFDCP / DFDC / FaceShifter / DeepFakeDetection）×(CTRL 7 点均值 vs SPSA+SWA 7 点均值)。

| 文件 | 内容 |
|---|---|
| cross_domain_results_7points.json | SPSA+SWA（复现 s2048, 5090）7 个 step 权重逐点跨域指标 |
| cross_domain_results_swa.json | 同上，SWA 最终模型 |
| cross_domain_ctrl_s2048_19_11_55.json | CTRL（Task10, `ucf_2026-09-02-19-11-55`）7 个 step 权重跨域指标（2026-09-02 补跑的表 2 baseline） |
| ucf_2026-09-02-19-11-55/training.log | CTRL 跨域 run 的 UCF 训练日志 |
| train_task10_ctrl_s2048.log | 上述 run 启动 stdout |
| cross_eval_ctrl_s2048_19_11_55.log | 上述 run 跨域评估 stdout |
| cross_domain_eval.py | 跨域评估脚本 |
| crossdomain_final_matrix_20260831.md | **表 2 数值矩阵**（CTRL/SPSA 7 点均值 + SWA 单测） |
| crossdomain_7ptmean_report_STOP_20260830.md | 说明受控 12 组无中间检查点、7 点轨迹仅存于 training.log |
| auth3_*.json | 授权第三方（auth3）复现的 7 点/跨域评估数值 |

复现 s2048 的中间权重已随云端目录丢失（见 `../logs/ctrl_training_logs/README.md`），本目录 json 为表 2 最终可回源数值载体。
