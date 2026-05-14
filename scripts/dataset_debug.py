#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path("/root/autodl-tmp/project-building")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "outputs" / "dataset_debug"

from tools.dataset import build_dataset


def denormalize(image_chw: np.ndarray) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    image = image_chw * std + mean
    image = np.clip(image, 0.0, 1.0)
    return np.transpose(image, (1, 2, 0))


def save_visualization(sample: dict, out_path: Path) -> None:
    image = sample["image"].cpu().numpy()
    mask = sample["mask"].cpu().numpy()[0]

    image_vis = denormalize(image)
    overlay = image_vis.copy()
    overlay[mask > 0.5] = [1.0, 0.1, 0.1]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image_vis)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Mask (fg={mask.mean() * 100:.1f}%)")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    fig.suptitle(f"{sample['source']} / {sample['split']} / {sample['id']}", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset debug visualization")
    parser.add_argument("--source", choices=["whu", "inria_patch"], required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable train-time augmentation and only apply Normalize.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(
        source=args.source,
        split=args.split,
        use_augment=not args.no_augment,
    )

    indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))
    saved_files: list[str] = []

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        out_path = OUT_DIR / f"{args.source}_{args.split}_{i:02d}_{sample['id']}.png"
        save_visualization(sample, out_path)
        saved_files.append(out_path.name)

    print(f"dataset size: {len(dataset)}")
    print(f"saved: {len(saved_files)}")
    print(f"output_dir: {OUT_DIR}")
    for name in saved_files:
        print(name)


if __name__ == "__main__":
    main()
