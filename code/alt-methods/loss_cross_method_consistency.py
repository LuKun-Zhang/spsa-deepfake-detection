'''
Cross-Method Consistency Loss for UCF (our innovation) — v2: NT-Xent.

Idea: For each pristine video v in FF++, there exist four forged versions
(DF/F2F/FS/NT) generated from the SAME source video (same-source, different
methods). UCF only uses the same-source structure to build the leave-one-out
cross-domain splits (FF-wo-X) but never uses it as a supervision signal.
This loss pulls the COMMON fingerprints (f_share) of two forged versions
from the same source video together, forcing the model to discard
method-specific traces and keep only cross-method-stable traces.

v1 (triplet + margin=3.0) 诊断结论 (2026-08-19, 4 模型 x 3200 对距离分布):
- 无约束 UCF 的 f_share 天然无同源一致性 (A/B ratio d_diff/d_same ~0.95)
- margin=3.0 相对特征距离尺度（中位数 ~0.6）过大，模型靠"整体放大特征
  (norm 3.3->15.4) + 推远异源"满足 margin，特征空间变形干扰主任务 -> 泛化掉分。
v2 修复：
- 特征 L2 归一化：禁止通过缩放作弊
- SimCLR 式 NT-Xent (InfoNCE)：temperature 固定对比尺度，无需 margin 校准
- 负样本 = 批内其余全部样本（anchor + positive 双向）
'''

import torch
import torch.nn.functional as F

from .abstract_loss_func import AbstractLossClass
from metrics.registry import LOSSFUNC


@LOSSFUNC.register_module(module_name="cross_method_consistency")
class CrossMethodConsistencyLoss(AbstractLossClass):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor, positive):
        """
        Args:
            anchor: [B, D] pooled common fingerprints of fake frames
                    (method A, source video v_i)
            positive: [B, D] pooled common fingerprints of the SAME source
                      videos forged by another method (method B != A)
        Returns:
            NT-Xent (InfoNCE) scalar loss
        """
        B = anchor.size(0)
        # L2 normalize to prevent scale cheating
        z = F.normalize(torch.cat([anchor, positive], dim=0), dim=1)  # [2B, D]
        sim = (z @ z.T) / self.temperature                            # [2B, 2B]
        # remove self-similarity (diagonal) from the logits -> [2B, 2B-1]
        eye = torch.eye(2 * B, dtype=torch.bool, device=sim.device)
        logits = sim[~eye].view(2 * B, 2 * B - 1)
        # positive pair: index i <-> index i+B.
        # 去掉对角线后：
        #   - 行 i (0..B-1)：正样本列 i+B 因删除本行对角线(列 i) 左移 1 -> 标签 i+B-1
        #   - 行 i+B (B..2B-1)：正样本列 i 位于删除的对角线之前 -> 标签 i
        labels = torch.cat([
            torch.arange(B - 1, 2 * B - 1, device=sim.device),
            torch.arange(0, B, device=sim.device),
        ])  # [2B]
        loss = F.cross_entropy(logits, labels)
        return loss
