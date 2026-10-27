"""Crop, preprocess and embed object crops with a frozen ImageNet backbone.

Training (train_svm.py) and inference (infer.py) both go through this module,
so crops are cropped, resized and normalized identically in both.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from PIL import Image
from torchvision.transforms.v2 import functional as TF

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
RESIZE_MODE = "letterbox"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _resnet18() -> tuple[nn.Module, int]:
    model = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    return model, 512


def _resnet50() -> tuple[nn.Module, int]:
    model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Identity()
    return model, 2048


# name -> factory returning (model that outputs pooled features, feature dim)
BACKBONES: dict[str, Callable[[], tuple[nn.Module, int]]] = {
    "resnet18": _resnet18,
    "resnet50": _resnet50,
}


def load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))


def crop_box(image: np.ndarray, box_xyxy: Sequence[float], pad: float = 0.15) -> np.ndarray:
    """Crop a detector box, expanded by `pad` x its width/height on each side.

    This is the crop convention: training crops must be made with the same
    function and `pad` that inference uses. Works for RGB or BGR images.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box_xyxy)
    bw, bh = x2 - x1, y2 - y1
    x1 = min(max(0, round(x1 - pad * bw)), w - 1)
    y1 = min(max(0, round(y1 - pad * bh)), h - 1)
    x2 = max(min(w, round(x2 + pad * bw)), x1 + 1)
    y2 = max(min(h, round(y2 + pad * bh)), y1 + 1)
    return image[y1:y2, x1:x2]


def preprocess(crop: np.ndarray, size: int = 224) -> torch.Tensor:
    """HxWx3 uint8 RGB crop -> normalized 3 x size x size float tensor, letterboxed."""
    if crop.ndim != 3 or crop.shape[2] != 3 or crop.dtype != np.uint8:
        raise ValueError(f"expected HxWx3 uint8 RGB crop, got {crop.dtype} {crop.shape}")
    x = torch.from_numpy(np.array(crop, order="C")).permute(2, 0, 1)
    x = TF.to_dtype(x, torch.float32, scale=True)
    h, w = x.shape[1:]
    scale = size / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    x = TF.resize(x, [nh, nw], antialias=True)
    x = TF.normalize(x, IMAGENET_MEAN, IMAGENET_STD)
    # Zero padding after normalization == padding with the ImageNet mean color.
    left, top = (size - nw) // 2, (size - nh) // 2
    return F.pad(x, (left, size - nw - left, top, size - nh - top))


class FeatureExtractor:
    def __init__(self, backbone: str = "resnet18", device: str = "auto",
                 input_size: int = 224, batch_size: int = 64):
        if backbone not in BACKBONES:
            raise ValueError(f"unknown backbone {backbone!r}; available: {sorted(BACKBONES)}")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.backbone = backbone
        self.input_size = input_size
        self.batch_size = batch_size
        self.model, self.feature_dim = BACKBONES[backbone]()
        self.model.eval().to(self.device)

    @property
    def meta(self) -> dict:
        """Settings that determine the features; saved with the SVMs and checked on load."""
        return {
            "backbone": self.backbone,
            "input_size": self.input_size,
            "resize_mode": RESIZE_MODE,
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
            "feature_dim": self.feature_dim,
        }

    @torch.inference_mode()
    def embed_batch(self, batch: torch.Tensor) -> np.ndarray:
        """Preprocessed N x 3 x S x S batch -> N x D float32 features (not normalized)."""
        feats = self.model(batch.to(self.device, non_blocking=True))
        return feats.float().cpu().numpy()

    def embed(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """RGB uint8 crops -> N x D float32 features."""
        out = []
        for i in range(0, len(crops), self.batch_size):
            batch = torch.stack([preprocess(c, self.input_size) for c in crops[i:i + self.batch_size]])
            out.append(self.embed_batch(batch))
        if not out:
            return np.empty((0, self.feature_dim), dtype=np.float32)
        return np.concatenate(out)
