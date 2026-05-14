from __future__ import annotations

import torch


class BinarySegmentationMeter:
    def __init__(self, threshold: float = 0.5, eps: float = 1e-7) -> None:
        self.threshold = threshold
        self.eps = eps
        self.reset()

    def reset(self) -> None:
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        probs = torch.sigmoid(logits)
        preds = (probs >= self.threshold).float()
        targets = (targets >= 0.5).float()

        self.tp += float((preds * targets).sum().item())
        self.fp += float((preds * (1.0 - targets)).sum().item())
        self.fn += float(((1.0 - preds) * targets).sum().item())

    def compute(self) -> dict[str, float]:
        iou = self.tp / (self.tp + self.fp + self.fn + self.eps)
        dice = 2.0 * self.tp / (2.0 * self.tp + self.fp + self.fn + self.eps)
        precision = self.tp / (self.tp + self.fp + self.eps)
        recall = self.tp / (self.tp + self.fn + self.eps)
        return {
            "iou": float(iou),
            "dice": float(dice),
            "precision": float(precision),
            "recall": float(recall),
        }
