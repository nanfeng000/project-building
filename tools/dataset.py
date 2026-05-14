from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


PROJECT_ROOT = Path("/root/autodl-tmp/project-building")
META_ROOT = PROJECT_ROOT / "data" / "meta"
INRIA_PATCH_ROOT = PROJECT_ROOT / "data" / "processed" / "inria_patch512_s512"

DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


def build_transforms(
    split: str,
    use_augment: bool = True,
    mean: tuple[float, float, float] = DEFAULT_MEAN,
    std: tuple[float, float, float] = DEFAULT_STD,
) -> A.Compose:
    transforms: list[Any] = []
    if split == "train" and use_augment:
        transforms.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ]
        )
    transforms.append(A.Normalize(mean=mean, std=std, max_pixel_value=255.0))
    return A.Compose(transforms)


def resolve_manifest_path(
    source: str,
    split: str,
    manifest_path: str | Path | None = None,
) -> Path:
    if manifest_path is not None:
        return Path(manifest_path)

    source = source.lower()
    split = split.lower()

    if source == "whu":
        return META_ROOT / f"whu_{split}.csv"
    if source == "inria_patch":
        return INRIA_PATCH_ROOT / f"{split}_patches.csv"

    raise ValueError(f"Unsupported source: {source}")


class BuildingDataset(Dataset):
    """
    统一建筑物提取数据集读取模块。

    支持：
    - WHU-Building：读取 `data/meta/whu_{split}.csv`
    - Inria patch：读取 `data/processed/inria_patch512_s512/{split}_patches.csv`

    输出：
    - image: float32 tensor, CxHxW
    - mask: float32 tensor, 1xHxW, 值域严格 0/1
    """

    def __init__(
        self,
        source: str,
        split: str,
        manifest_path: str | Path | None = None,
        transform: A.Compose | None = None,
        use_augment: bool = True,
    ) -> None:
        self.source = source.lower()
        self.split = split.lower()
        self.manifest_path = resolve_manifest_path(self.source, self.split, manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        self.transform = transform or build_transforms(
            split=self.split,
            use_augment=use_augment,
        )
        self.samples = self._load_samples()

    def _load_samples(self) -> list[dict[str, Any]]:
        with open(self.manifest_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError(f"Empty manifest: {self.manifest_path}")
        return rows

    def __len__(self) -> int:
        return len(self.samples)

    def _read_image(self, image_path: Path) -> np.ndarray:
        image = np.array(Image.open(image_path).convert("RGB"))
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Invalid image shape {image.shape} for {image_path}")
        return image

    def _read_mask(self, sample: dict[str, Any], image_shape: tuple[int, int, int]) -> tuple[np.ndarray, bool]:
        mask_path = sample.get("mask_path", "")
        has_mask = bool(mask_path)

        if has_mask and Path(mask_path).exists():
            mask = np.array(Image.open(mask_path))
        else:
            # Inria unlabeled test 使用全零 mask，方便保持统一输出格式。
            mask = np.zeros(image_shape[:2], dtype=np.uint8)
            has_mask = False

        if mask.ndim == 3:
            mask = mask[..., 0]

        # WHU 为 bool / mode=1；Inria 为 uint8 {0,255}。统一转成 0/1 float32。
        mask = (mask > 0).astype(np.float32)
        return mask, has_mask

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image_key = "image_path"
        id_key = "patch_name" if "patch_name" in sample else "file_name"

        image_path = Path(sample[image_key])
        image = self._read_image(image_path)
        mask, has_mask = self._read_mask(sample, image.shape)

        transformed = self.transform(image=image, mask=mask)
        image_np = transformed["image"].astype(np.float32)
        mask_np = transformed["mask"].astype(np.float32)
        mask_np = (mask_np > 0.5).astype(np.float32)

        image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask_np[None, ...]).float()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "id": sample[id_key],
            "source": self.source,
            "split": self.split,
            "image_path": str(image_path),
            "mask_path": sample.get("mask_path", ""),
            "has_mask": has_mask,
            "meta": sample,
        }


def build_dataset(
    source: str,
    split: str,
    manifest_path: str | Path | None = None,
    transform: A.Compose | None = None,
    use_augment: bool = True,
) -> BuildingDataset:
    return BuildingDataset(
        source=source,
        split=split,
        manifest_path=manifest_path,
        transform=transform,
        use_augment=use_augment,
    )
