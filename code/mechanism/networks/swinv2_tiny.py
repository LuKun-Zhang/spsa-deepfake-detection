'''
SwinV2-Tiny backbone for the UCF detector (cross-backbone directionality).

Written against the contract verified on 871 (2026-09-19):
  1. features(x) -> 4D NCHW
  2. exactly 512 channels   (encoder_feat_dim: 512; dconv_up3 = r_double_conv(512,256))
  3. spatial == input/32    (256 -> 8x8); the reconstruction UNet upsamples 2*2*2*4 = 32x

**THE ONE THING THAT MATTERS HERE: timm's Swin returns BHWC, not BCHW.**
Measured on 871 CPU probe:
    swinv2_tiny_window8_256.ms_in1k  ->  (1, 8, 8, 768)     <- NHWC
    convnext_tiny.fb_in22k_ft_in1k   ->  (1, 768, 8, 8)     <- NCHW
Without the permute(0,3,1,2) below, the next op is adjust_channel's
nn.Conv2d(768, 512, 1, 1) fed a tensor whose dim-1 is 8 -> loud channel-mismatch
crash (not a silent wrong answer), but it will look like a plumbing bug.

Why this tag and not the 224 one:
    swin_tiny_patch4_window7_224  ->  AssertionError: Input height (256) doesn't match model (224)
    maxvit_tiny_tf_224            ->  AssertionError: height (64) must be divisible by window (7)
256/4 = 64, and 64 % 7 != 0, so the window-7 models simply cannot take 256 input.
window8 variants are the ones actually built for 256, so the pretrained rel_pos_bias
table matches. Do NOT try to force window_size=8 onto the 224 tag -- the pretrained
relative-position-bias table is sized (2*7-1)^2 and will not load.

Needs HF_ENDPOINT=https://hf-mirror.com (huggingface.co is unreachable from 871).
'''

import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from metrics.registry import BACKBONE

logger = logging.getLogger(__name__)

TIMM_TAG = 'swinv2_tiny_window8_256.ms_in1k'


@BACKBONE.register_module(module_name="swinv2_tiny")
class SwinV2Tiny(nn.Module):
    def __init__(self, swinv2_tiny_config):
        super(SwinV2Tiny, self).__init__()
        self.num_classes = swinv2_tiny_config["num_classes"]
        inc = swinv2_tiny_config["inc"]
        self.dropout = swinv2_tiny_config["dropout"]
        self.mode = swinv2_tiny_config["mode"]

        if not os.environ.get('HF_ENDPOINT'):
            logger.warning('HF_ENDPOINT is unset; huggingface.co is unreachable from this '
                           'host. Export HF_ENDPOINT=https://hf-mirror.com or the timm '
                           'weight download will hang/fail.')
        self.backbone = timm.create_model(
            TIMM_TAG, pretrained=True, num_classes=0, in_chans=inc)
        feat_dim = self.backbone.num_features          # 768 for SwinV2-Tiny
        logger.info('SwinV2-Tiny built: feat_dim=%d (expect 768)', feat_dim)

        self.last_layer = nn.Linear(feat_dim, self.num_classes)
        if self.dropout:
            self.dropout_layer = nn.Dropout(p=self.dropout)

        if self.mode == 'adjust_channel':
            self.adjust_channel = nn.Sequential(
                nn.Conv2d(feat_dim, 512, 1, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            )

    def features(self, x):
        x = self.backbone.forward_features(x)          # (B, H, W, C)   <- NHWC
        x = x.permute(0, 3, 1, 2).contiguous()         # -> (B, C, H, W)  REQUIRED
        if self.mode == 'adjust_channel':
            x = self.adjust_channel(x)
        return x

    def classifier(self, x):
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        if self.dropout:
            x = self.dropout_layer(x)
        return self.last_layer(x)

    def forward(self, x):
        return self.classifier(self.features(x))
