"""The three `timm.models.layers` helpers `network_swin2sr.py` imports.

timm is a large dependency and only three small functions are needed, so they
are supplied here rather than installed. The implementations are timm's
(https://github.com/huggingface/pytorch-image-models, Apache-2.0)."""
import math

import torch
import torch.nn as nn


def to_2tuple(x):
    return (x, x) if not isinstance(x, (tuple, list)) else tuple(x)


class DropPath(nn.Module):
    """Per-sample stochastic depth. A no-op at eval, which is the only mode this
    repository runs the network in."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    with torch.no_grad():
        tensor.normal_(mean, std).clamp_(a * std + mean, b * std + mean)
    return tensor
