from __future__ import annotations

from typing import Any

from .segmentors import V2LiteSegmentor
from .unet import UNet


def build_model(name: str, **kwargs: Any):
    name = name.lower()
    if name == "unet":
        return UNet(**kwargs)
    if name in {"v2lite", "mdu_v2lite", "v2-lite"}:
        return V2LiteSegmentor(**kwargs)
    raise ValueError(f"Unsupported model: {name}")
