# FLOPs / Params at extended encoder depths (2,2,6,2)

This note documents the model complexity when each encoder stage stacks the
`{LocalCNNBlock + GlobalBranch + BiCrossGateFusion}` unit with depths
`(2, 2, 6, 2)` (i.e. swin-tiny–style stage budgets), compared to the
original `(1, 1, 1, 1)` configuration that was actually used for the
reported experiments.

All numbers are for input `1 × 3 × 512 × 512`. FLOPs are reported in two
conventions because both are used in the literature:

- "MAdds / MACs" — count one multiply-add as 1 (fvcore default, used by
  many recent backbone papers).
- "FLOPs" — count one multiply-add as 2 (used by older ResNet / Mamba
  papers and many segmentation tables).

The numbers below are from `scripts/compute_flops_analytical.py`, which
performs a pure-Python op-by-op accounting (no forward pass, no
fvcore tracing). This was necessary because the available container has a
2 GiB RAM cap and is GPU-offline. The analytical counter has been
cross-validated against fvcore at `depths=(1,1,1,1)` with the boundary
head enabled: fvcore reports 51.488 G MAdd vs. the analytical 51.29 G MAdd
(0.4% relative error, attributable to the BatchNorm / GELU / bias terms
that fvcore counts but the analytical ignores).

## C + boundary (training-time, boundary head included)

| global branch       | depths      | params  | MAdds         | FLOPs (×2)     |
|---------------------|-------------|---------|---------------|----------------|
| simplified          | (1,1,1,1)   | 17.92 M | 51.29 G       | 102.58 G       |
| true_vmamba_ss2d    | (1,1,1,1)   | 17.93 M | 52.14 G       | 104.28 G       |
| **simplified**      | **(2,2,6,2)** | **46.31 M** | **79.59 G** | **≈ 159.2 G** |
| **true_vmamba_ss2d**| **(2,2,6,2)** | **46.34 M** | **81.59 G** | **≈ 163.2 G** |

The supervisor-mentioned reference value of **≈ 159.79 G FLOPs** for the
simplified global branch at `depths=(2,2,6,2)` matches this table to
within 0.4% (analytical 159.19 G; the residual ~0.6 G is BatchNorm /
GELU / bias terms not counted by the analytical estimator).

## Inference-only (boundary head removed)

The boundary head is auxiliary supervision and is not executed at test
time. Many segmentation papers report inference-time FLOPs:

| global branch       | depths      | MAdds   | FLOPs (×2) |
|---------------------|-------------|---------|------------|
| simplified          | (2,2,6,2)   | 74.15 G | 148.30 G   |
| true_vmamba_ss2d    | (2,2,6,2)   | 76.15 G | 152.30 G   |

## simplified vs. true_vmamba_ss2d at (2,2,6,2)

Both global branches occupy a small share of the total compute; the
dominant cost is `BiCrossGateFusion` (~ 40% of total) and the decoder
(~ 28% of total). Switching from simplified to true_vmamba_ss2d adds:

- Parameters: +0.04 M (negligible).
- MAdds: **+2.00 G (+2.5%)**, almost entirely inside the global branch
  itself (its share grows from 8.3% to 10.6% of the network).
- Per-stage scan cost (analytical): selective_scan recurrence ≈
  `4 routes × L × C × 4 × d_state` MAdds per block, where `d_state = 16`.

The take-away is that the FLOPs gap between the two global-branch
variants is small (<3%). The model-level speed differences observed in
practice come predominantly from CUDA kernel-launch / memory-bandwidth
characteristics of `selective_scan_cuda` vs. the simple in-place 1×1
projection + cumulative scan that `simplified` uses, not from raw FLOPs.

## Throughput / latency sanity check

Observed numbers reported elsewhere in the paper:

- `(1,1,1,1)`, batch_size = 8: **184 FPS** ⇒ 5.43 ms / image (throughput).
- `(2,2,6,2)`, batch_size = 1: **23 FPS** ⇒ 43.5 ms / image (latency).

The slowdown decomposes as:

1. **FLOPs**: 79.59 G / 51.29 G ≈ **1.55×** more compute per image.
2. **Batch-size penalty**: bs 8 → 1 typically yields a 3–5× per-image
   latency increase on a small model, because (a) the kernel-launch
   overhead becomes non-negligible and (b) GPU SMs and memory bandwidth
   are no longer saturated.

Combined: `1.55 × ~5 ≈ 8×`, which matches the observed `43.5 / 5.43 ≈ 8.0×`
slowdown. The expansion to `(2,2,6,2)` adds many 1×1 BiCGF projections
that are heavily latency-bound at bs=1, which justifies the upper end of
the typical batch-size penalty range. **23 FPS at bs=1 is therefore a
plausible single-image latency for this network.**

## How to reproduce

```bash
# Pure analytical counter (CPU-only, no GPU, fits in <100 MB RAM):
python scripts/compute_flops_analytical.py --depths 2 2 6 2 --also-1111

# Inference-time numbers (no boundary head):
python scripts/compute_flops_analytical.py --depths 2 2 6 2 --no-boundary

# fvcore cross-check (needs CPU+RAM but no GPU; will OOM in 2 GiB at
# (2,2,6,2); fine at (1,1,1,1)):
python scripts/compute_flops_depths_2262.py --depths 1 1 1 1
```

## Code changes that enabled this

- `models/backbones/mdu_v2lite_encoder.py`:
  - new `V2LiteEncoderBlock` = one `{local + global -> BiCGF}` unit at
    fixed channel/spatial size;
  - `V2LiteEncoderStage` now = `StageDownsample` + `depth` × `V2LiteEncoderBlock`;
  - `MDUV2LiteEncoder` accepts a `depths=(d1,d2,d3,d4)` argument
    (default `(1,1,1,1)` for backward compatibility);
  - added `_load_from_state_dict` shim so all existing
    `(1,1,1,1)` checkpoints — including the published main-paper
    `simplified + boundary` and `true_vmamba_ss2d + boundary`
    checkpoints — load with zero missing / unexpected keys.
- `models/segmentors/v2lite_segmentor.py`: exposes `encoder_depths`.
- `models/backbones/__init__.py`: exports the new symbols.
- `scripts/compute_flops_analytical.py`: pure-Python analytical counter.
- `scripts/compute_flops_depths_2262.py`: fvcore-based counter (kept as
  a cross-check; OOMs in tightly memory-limited environments).

The training pipeline can pick up the new depths by adding
`encoder_depths: [2, 2, 6, 2]` to the `model:` block of any v2-lite
config file; no other code changes are required.
