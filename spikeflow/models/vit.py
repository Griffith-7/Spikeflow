"""Spiking Vision Transformer (ViT) — ImageNet-scale model.

Accuracy targets (from published SNN papers):
    SpikingViTTiny:   ~75% ImageNet
    SpikingViTSmall:  ~80% ImageNet
    SpikingViTBase:   ~83% ImageNet
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from spikeflow.attention.feedforward import SpikingFFN
from spikeflow.attention.self_attention import SpikingSelfAttention
from spikeflow.layers.linear import SpikingLinear


class SpikingPatchEmbed(nn.Module):
    """Patch embedding: image -> sequence of patch tokens."""

    def __init__(self, img_size=224, patch_size=16, in_channels=3, d_model=768):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class SpikingTransformerBlock(nn.Module):
    """Pre-norm transformer block with spiking neurons."""

    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 12,
        d_ff: int | None = None,
        dropout: float = 0.0,
        attention_mode: str = "xnor",
        threshold: float = 1.0,
        tau: float = 2.0,
    ):
        super().__init__()
        d_ff = d_ff or int(d_model * 4)

        self.attn_norm = nn.LayerNorm(d_model)
        self.self_attn = SpikingSelfAttention(
            d_model, n_heads, dropout=dropout,
            attention_mode=attention_mode, threshold=threshold, tau=tau,
        )
        self.ff_norm = nn.LayerNorm(d_model)
        self.ffn = SpikingFFN(d_model, d_ff, dropout=dropout, threshold=threshold, tau=tau)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.self_attn(self.attn_norm(x))
        x = x + self.ffn(self.ff_norm(x))
        return x

    def reset_state(self):
        self.self_attn.reset_state()
        self.ffn.reset_state()

    def set_sfa_mode(self, enabled: bool):
        self.self_attn.set_sfa_mode(enabled)
        self.ffn.set_sfa_mode(enabled)


class SpikingViT(nn.Module):
    """Spiking Vision Transformer for image classification."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        d_ff: int | None = None,
        dropout: float = 0.0,
        attention_mode: str = "xnor",
        threshold: float = 1.0,
        tau: float = 2.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes

        self.patch_embed = SpikingPatchEmbed(img_size, patch_size, in_channels, d_model)
        n_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, d_model))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            SpikingTransformerBlock(
                d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                dropout=dropout, attention_mode=attention_mode,
                threshold=threshold, tau=tau,
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.head = SpikingLinear(d_model, num_classes, threshold=threshold, tau=tau, readout=True) if num_classes > 0 else nn.Identity()

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = x[:, 0]  # CLS token
        return self.head(x)

    def reset_state(self):
        for m in self.children():
            if hasattr(m, "reset_state"):
                m.reset_state()

    def set_sfa_mode(self, enabled: bool):
        for m in self.children():
            if hasattr(m, "set_sfa_mode"):
                m.set_sfa_mode(enabled)


def SpikingViTTiny(num_classes=1000, **kwargs):
    return SpikingViT(
        d_model=192, n_layers=12, n_heads=3, num_classes=num_classes,
        attention_mode="xnor", **kwargs,
    )

def SpikingViTSmall(num_classes=1000, **kwargs):
    return SpikingViT(
        d_model=384, n_layers=12, n_heads=6, num_classes=num_classes,
        attention_mode="xnor", **kwargs,
    )

def SpikingViTBase(num_classes=1000, **kwargs):
    return SpikingViT(
        d_model=768, n_layers=12, n_heads=12, num_classes=num_classes,
        attention_mode="hybrid", **kwargs,
    )
