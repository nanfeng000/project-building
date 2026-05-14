# WHU Qualitative Strong-Baseline Comparison

**Caption (suggested):** Qualitative comparison on the WHU test set. Red boxes highlight challenging regions, including small buildings, dense building areas, complex boundaries, and adjacent buildings. Compared with U-Net and DeepLabV3-ResNet50, the proposed method produces more complete building regions and more accurate boundaries.

**Combined grid:** `whu_strong_baseline_comparison.png` (PDF: `whu_strong_baseline_comparison.pdf`)

## Selected samples

| # | Category | WHU id | U-Net IoU | DeepLabV3 IoU | Ours IoU | Per-sample figure |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | small buildings | 2_630 | 0.0400 | 0.2539 | 0.6778 | `visualizations/small_buildings_2_630.png` |
| 2 | small buildings | 2_568 | 0.4640 | 0.4882 | 0.7951 | `visualizations/small_buildings_2_568.png` |
| 3 | dense buildings | 317 | 0.3892 | 0.2634 | 0.5768 | `visualizations/dense_buildings_317.png` |
| 4 | dense buildings | 2_1438 | 0.8161 | 0.8051 | 0.8918 | `visualizations/dense_buildings_2_1438.png` |
| 5 | complex boundary | 451 | 0.2228 | 0.3858 | 0.6714 | `visualizations/complex_boundary_451.png` |
| 6 | complex boundary | 2_863 | 0.7485 | 0.6432 | 0.8160 | `visualizations/complex_boundary_2_863.png` |
| 7 | adjacent buildings | 2_1687 | 0.7131 | 0.0574 | 0.9720 | `visualizations/adhesive_buildings_2_1687.png` |
| 8 | adjacent buildings | 446 | 0.9092 | 0.8849 | 0.9611 | `visualizations/adhesive_buildings_446.png` |

## Notes

- Red boxes are auto-detected to highlight the largest connected disagreement region between the worse of (U-Net, DeepLabV3) and the ground truth; if no such region exists, no box is drawn.
- Per-sample IoU is computed at threshold 0.5 on the raw prediction (no CRF / post-processing), consistent with the rest of the project.

## Per-image binary predictions

- U-Net: `preds/unet/<id>.png` (8 files)
- DeepLabV3-ResNet50: `preds/deeplabv3/<id>.png` (8 files)
- Ours (C+boundary): `preds/ours/<id>.png` (8 files)
- Note: ``--save-all-preds`` was NOT enabled; only predictions for the selected qualitative samples were saved. Re-run with ``--save-all-preds`` to dump every WHU test image.
