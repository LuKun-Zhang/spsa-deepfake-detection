'''
ConvNeXt-Tiny backbone for the UCF detector (cross-backbone directionality).

Written against the contract verified on 871 (2026-09-19):
  1. features(x) -> 4D NCHW
  2. exactly 512 channels   (encoder_feat_dim: 512; dconv_up3 = r_double_conv(512,256))
  3. spatial == input/32    (256 -> 8x8); the reconstruction UNet upsamples 2*2*2*4 = 32x
Measured on 871 CPU probe: timm 'convnext_tiny.fb_in22k_ft_in1k' returns (1, 768, 8, 8) NCHW.
So this one is a drop-in: NO permute, NO spatial hack.

Pretrained weights come from timm/HuggingFace at construction time, NOT from the
top-level `pretrained:` path -- exactly the pattern resnet34.py already uses.
That top-level load in ucf_detector.build_backbone() is load_state_dict(..., strict=False),
so pointing it at the no-op stub leaves this model's timm weights untouched.

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

TIMM_TAG = 'convnext_tiny.fb_in22k_ft_in1k'


@BACKBONE.register_module(module_name="convnext_tiny")
class ConvNeXtTiny(nn.Module):
    def __init__(self, convnext_tiny_config):
        super(ConvNeXtTiny, self).__init__()
        self.num_classes = convnext_tiny_config["num_classes"]
        inc = convnext_tiny_config["inc"]
        self.dropout = convnext_tiny_config["dropout"]
        self.mode = convnext_tiny_config["mode"]

        if not os.environ.get('HF_ENDPOINT'):
            logger.warning('HF_ENDPOINT is unset; huggingface.co is unreachable from this '
                           'host. Export HF_ENDPOINT=https://hf-mirror.com or the timm '
                           'weight download will hang/fail.')
        self.backbone = timm.create_model(
            TIMM_TAG, pretrained=True, num_classes=0, in_chans=inc)
        feat_dim = self.backbone.num_features          # 768 for ConvNeXt-Tiny
        logger.info('ConvNeXt-Tiny built: feat_dim=%d (expect 768)', feat_dim)

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
        # timm ConvNeXt forward_features already returns NCHW, post final LayerNorm.
        x = self.backbone.forward_features(x)
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
