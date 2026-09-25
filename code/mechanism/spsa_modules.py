# -*- coding: utf-8 -*-
"""
SPSA: Pyramid-Split Attention + PoolFormer pooling token mixing
================================================================
思路来源（他人提供的创新点描述）：
  1. YOLOv10 的 PSA：输入 1x1 conv 后 split 成两半——一半过自注意力（MHSA），
     另一半直接 concat（为速度牺牲精度）。
  2. MetaFormer / PoolFormer：用 pooling 代替多头自注意力做 token mixing，
     以极小代价提供全局/局部聚合能力。
  3. SPSA = 把 PSA 中"恒等 concat"的那一半换成 PoolFormer 式 pooling attention，
     使两半都参与信息聚合，另加 1x1 conv 跨层连接（shortcut）保梯度。

插入位置（UCF decoder 基准）：
  Conditional_UNet.forward 中 `x = self.dconv_up2(x)` 之后（128 通道, 32x32），
  与对方提供的 `self.spsa = SPSA(128, 128)` 通道一致。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MHSA(nn.Module):
    """Global multi-head self-attention（尺寸无关，任意 HxW）。

    与 YOLOv10 的窗口 MHSA 相比用全局 attention，实现简单可靠；
    插入点特征图 32x32=1024 tokens，计算量/显存可接受。
    """

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        assert d_model % n_heads == 0, f'd_model {d_model} % n_heads {n_heads} != 0'
        self.n_heads = n_heads
        self.dh = d_model // n_heads
        self.qkv = nn.Conv2d(d_model, 3 * d_model, 1)
        self.proj = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x)  # B, 3C, H, W
        q, k, v = qkv.chunk(3, dim=1)  # each: B, C, H, W
        # 展成 heads 维度
        q = q.reshape(B, self.n_heads, self.dh, H * W).transpose(2, 3)  # B,H,HW,dh
        k = k.reshape(B, self.n_heads, self.dh, H * W)                  # B,H,dh,HW
        v = v.reshape(B, self.n_heads, self.dh, H * W).transpose(2, 3)  # B,H,HW,dh
        attn = (q @ k) * (self.dh ** -0.5)   # B,H,HW,HW
        attn = F.softmax(attn, dim=-1)
        out = attn @ v                        # B,H,HW,dh
        out = out.transpose(2, 3).reshape(B, C, H, W)
        return self.proj(out)


class PoolMix(nn.Module):
    """PoolFormer 式 pooling token mixing：pooling 作为廉价 attention。

    PoolFormer 原式：x = x + Pool(LN(x))。这里用 GroupNorm（避免 BN 的 batch 依赖，
    UCF 的 decoder 原始没有任何 BN/LN，保持风格一致更稳）。
    """

    def __init__(self, dim: int, pool_kernel: int = 3):
        super().__init__()
        self.pool = nn.AvgPool2d(pool_kernel, stride=1,
                                 padding=pool_kernel // 2, count_include_pad=False)
        self.norm = nn.GroupNorm(8, dim)

    def forward(self, x):
        return x + self.pool(self.norm(x))


class SPSA(nn.Module):
    """PSA + PoolFormer 组合。

    结构：
      x --[cv1: 1x1, 2*c_]--> split(c_) --> 一半 MHSA / 一半 PoolMix --[cat]--> [cv2: 1x1]--> + [shortcut: 1x1]
    """

    def __init__(self, c1: int, c2: int, e: float = 0.5, n_heads: int = 4, pool_kernel: int = 3):
        super().__init__()
        c_ = max(int(c1 * e), 1)
        self.c_ = c_
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, 2 * c_, 1),
            nn.GroupNorm(8, 2 * c_),
            nn.SiLU(inplace=True),
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d(2 * c_, c2, 1),
            nn.GroupNorm(8, c2),
            nn.SiLU(inplace=True),
        )
        self.attn = MHSA(c_, n_heads=n_heads)
        self.poolmix = PoolMix(c_, pool_kernel=pool_kernel)
        # 1x1 跨层连接：c1 != c2 时也自适应
        self.shortcut = nn.Conv2d(c1, c2, 1)

    def forward(self, x):
        y, m = self.cv1(x).split(self.c_, dim=1)  # 各 B,c_,H,W
        out = torch.cat([self.attn(y), self.poolmix(m)], dim=1)  # B,2c_,H,W
        out = self.cv2(out)                        # B,c2,H,W
        return out + self.shortcut(x)

class SPSAPlaceholder(nn.Module):
    """Equal-parameter structural control for SPSA (mechanism leg A).

    Same residual-block shape, GroupNorm/SiLU style and 1x1 shortcut as SPSA,
    but the pyramid-split (MHSA half + PoolFormer poolmix half) is REMOVED:
    this is a plain two-layer channel MLP (two 1x1 convs), i.e. no self-attention
    and no spatial token mixing. Total params ~66.6K == SPSA (within 0.3%).

    Question it answers: is the (SPSA+SWA) super-additivity specific to the SPSA
    structure, or does any equal-parameter residual adapter with SWA suffice?
    """

    def __init__(self, c1: int = 128, c2: int = 128, hidden: int = 192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c1, hidden, 1),
            nn.GroupNorm(8, hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c2, 1),
            nn.GroupNorm(8, c2),
            nn.SiLU(inplace=True),
        )
        self.shortcut = nn.Conv2d(c1, c2, 1)

    def forward(self, x):
        return self.net(x) + self.shortcut(x)
