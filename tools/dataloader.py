from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .dataset import BuildingDataset, build_dataset


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_dataloader(
    source: str,
    split: str,
    batch_size: int,
    num_workers: int = 4,
    manifest_path: str | Path | None = None,
    shuffle: bool | None = None,
    drop_last: bool | None = None,
    pin_memory: bool = True,
    persistent_workers: bool | None = None,
    use_augment: bool = True,
    max_samples: int | None = None,
    seed: int | None = None,
) -> DataLoader:
    dataset: BuildingDataset = build_dataset(
        source=source,
        split=split,
        manifest_path=manifest_path,
        use_augment=use_augment,
    )
    if max_samples is not None:
        max_samples = min(max_samples, len(dataset))
        dataset = Subset(dataset, list(range(max_samples)))

    if shuffle is None:
        shuffle = split.lower() == "train"
    if drop_last is None:
        drop_last = split.lower() == "train"
    if persistent_workers is None:
        persistent_workers = num_workers > 0
    generator = None
    worker_init_fn = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
        worker_init_fn = _seed_worker

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )
