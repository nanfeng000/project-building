"""
Analytical FLOPs counter for v2-lite at arbitrary encoder depths.

Pure-Python, no forward pass, no fvcore tracing -- so it runs in any RAM
budget (works inside a 2 GiB cgroup with GPU offline).

Compares two global-branch variants at a chosen input size:
  - global_branch_type = 'simplified'
  - global_branch_type = 'true_vmamba_ss2d'

FLOP convention: 1 multiply-add = 1 FLOP (i.e. MAdds / MACs).
This matches the common "FLOPs" number reported in segmentation papers
such as VMamba, SegFormer, ConvNeXt, etc. We also print the 2*MAdd
version for those who treat 1 MAdd = 2 FLOPs.

Usage:
  python scripts/compute_flops_analytical.py --depths 2 2 6 2
  python scripts/compute_flops_analytical.py --depths 1 1 1 1
  python scripts/compute_flops_analytical.py --depths 2 2 6 2 --input 1 3 512 512
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


# --------------------------------------------------------------------------- #
# Frozen v2-lite topology (matches models/segmentors/v2lite_segmentor.py)     #
# --------------------------------------------------------------------------- #
STEM_CHANNELS = 64
ENCODER_CHANNELS = (96, 192, 384, 512)
DECODER_CHANNELS = (256, 192, 128, 96)


@dataclass
class Tally:
    label: str
    flops: int  # MAdds


def conv_madd(in_c: int, out_c: int, k: int, h_out: int, w_out: int, groups: int = 1) -> int:
    """Conv2d MAdds: out_c * (in_c / groups) * k * k * H_out * W_out."""
    return out_c * (in_c // groups) * k * k * h_out * w_out


def linear_madd(b: int, l: int, in_f: int, out_f: int) -> int:
    return b * l * in_f * out_f


# --------------------------------------------------------------------------- #
# Module-by-module analytical FLOPs                                            #
# --------------------------------------------------------------------------- #
def stem_block_flops(in_c: int, out_c: int, h_in: int, w_in: int) -> Tally:
    # stride-2 conv 3x3
    h1, w1 = h_in // 2, w_in // 2
    f = conv_madd(in_c, out_c, 3, h1, w1)
    # stride-1 conv 3x3
    f += conv_madd(out_c, out_c, 3, h1, w1)
    return Tally("stem", f)


def stage_downsample_flops(in_c: int, out_c: int, h_out: int, w_out: int) -> Tally:
    # single stride-2 conv 3x3
    return Tally("downsample", conv_madd(in_c, out_c, 3, h_out, w_out))


def local_cnn_block_flops(c: int, h: int, w: int) -> Tally:
    # two repeats of {dwconv 3x3 + pwconv 1x1}
    f = 0
    for _ in range(2):
        f += conv_madd(c, c, 3, h, w, groups=c)   # dwconv
        f += conv_madd(c, c, 1, h, w)             # pwconv
    return Tally("local_cnn", f)


def global_simplified_flops(c: int, h: int, w: int) -> Tally:
    # in_proj 1x1, mix_proj 1x1, local_refine 3x3 dw, gate_proj 1x1, out_proj 1x1
    f = 0
    f += conv_madd(c, c, 1, h, w)   # in_proj
    f += conv_madd(c, c, 1, h, w)   # mix_proj
    f += conv_madd(c, c, 3, h, w, groups=c)  # local_refine (dwconv)
    f += conv_madd(c, c, 1, h, w)   # gate_proj
    f += conv_madd(c, c, 1, h, w)   # out_proj
    # bidirectional cumsum: O(C*H*W) adds, negligible vs convs
    return Tally("global_simplified", f)


def global_true_ss2d_flops(
    c: int, h: int, w: int, d_state: int = 16, expand_ratio: float = 1.0, num_routes: int = 4,
) -> Tally:
    """FLOPs for GlobalTrueSS2DBlock with mamba-ssm selective_scan.

    All sub-ops below match the implementation in models/blocks/global_mamba_block.py:
      norm  ->  in_proj (Linear)  ->  dwconv 3x3 (groups=d_inner)
            ->  x_proj einsum  ->  dt_proj einsum
            ->  selective_scan (recurrence, analytical)
            ->  out_norm  ->  out_proj (Linear)
    """
    d_inner = max(c, int(c * expand_ratio))
    L = h * w
    dt_rank = math.ceil(c / 16)

    f = 0
    # in_proj: Linear C -> 2 * d_inner over L tokens
    f += linear_madd(1, L, c, 2 * d_inner)
    # dwconv 3x3 over [B, d_inner, H, W]
    f += conv_madd(d_inner, d_inner, 3, h, w, groups=d_inner)
    # x_proj_weight einsum: "bkdl, kcd -> bkcl"; per batch: K * L * c_dim * d_inner
    c_dim = dt_rank + 2 * d_state
    f += num_routes * L * c_dim * d_inner
    # dt_projs_weight einsum: "bkrl, kdr -> bkdl"
    f += num_routes * L * d_inner * dt_rank
    # selective_scan analytical: per (step, channel) ~ 4 MAdds * d_state
    # Total over K routes: K * L * d_inner * 4 * d_state
    f += num_routes * L * d_inner * 4 * d_state
    # out_proj: Linear d_inner -> C
    f += linear_madd(1, L, d_inner, c)
    # norms negligible
    return Tally("global_true_ss2d", f)


def gate_generator_flops(c: int, h: int, w: int) -> int:
    f = conv_madd(c, c, 1, h, w)              # pre 1x1
    f += conv_madd(c, c, 3, h, w, groups=c)   # dw 3x3
    f += conv_madd(c, c, 1, h, w)             # out 1x1
    return f


def bicross_gate_fusion_flops(c: int, h: int, w: int, with_bidirectional_gate: bool = True) -> Tally:
    if not with_bidirectional_gate:
        # simple_fuse only: Conv 1x1 (2C -> C)
        f = conv_madd(2 * c, c, 1, h, w)
        return Tally("bicross_fusion(simple)", f)
    f = 0
    # two _GateGenerators
    f += 2 * gate_generator_flops(c, h, w)
    # two 1x1 projections (local_to_global_proj, global_to_local_proj)
    f += 2 * conv_madd(c, c, 1, h, w)
    # final fuse Sequential: Conv1x1 (4C -> C) + Conv3x3 (C -> C)
    f += conv_madd(4 * c, c, 1, h, w)
    f += conv_madd(c, c, 3, h, w)
    return Tally("bicross_fusion", f)


def decoder_block_flops(in_c: int, skip_c: int, out_c: int, h_out: int, w_out: int) -> Tally:
    # compress: Conv 1x1 (in + skip -> out)
    f = conv_madd(in_c + skip_c, out_c, 1, h_out, w_out)
    # refine: 2 × Conv 3x3 (out -> out)
    f += 2 * conv_madd(out_c, out_c, 3, h_out, w_out)
    return Tally("decoder_block", f)


def seg_head_flops(in_c: int, h: int, w: int) -> Tally:
    f = conv_madd(in_c, in_c, 3, h, w) + conv_madd(in_c, 1, 1, h, w)
    return Tally("seg_head", f)


def boundary_head_flops(in_c: int, h: int, w: int) -> Tally:
    f = conv_madd(in_c, in_c, 3, h, w) + conv_madd(in_c, 1, 1, h, w)
    return Tally("boundary_head", f)


# --------------------------------------------------------------------------- #
# Full network FLOPs                                                           #
# --------------------------------------------------------------------------- #
def encoder_stage_flops(
    in_c: int,
    out_c: int,
    h_out: int,
    w_out: int,
    depth: int,
    global_branch_type: str,
    with_mamba_branch: bool,
    with_bidirectional_gate: bool,
) -> list[Tally]:
    tallies: list[Tally] = []
    tallies.append(stage_downsample_flops(in_c, out_c, h_out, w_out))
    for _ in range(depth):
        tallies.append(local_cnn_block_flops(out_c, h_out, w_out))
        if with_mamba_branch:
            if global_branch_type == "simplified":
                tallies.append(global_simplified_flops(out_c, h_out, w_out))
            elif global_branch_type == "true_vmamba_ss2d":
                tallies.append(global_true_ss2d_flops(out_c, h_out, w_out))
            else:
                raise ValueError(f"Unsupported global_branch_type: {global_branch_type}")
            tallies.append(bicross_gate_fusion_flops(out_c, h_out, w_out, with_bidirectional_gate))
    return tallies


def model_flops(
    input_h: int,
    input_w: int,
    depths: tuple[int, int, int, int],
    global_branch_type: str = "simplified",
    with_mamba_branch: bool = True,
    with_bidirectional_gate: bool = True,
    with_boundary_head: bool = True,
) -> tuple[int, list[Tally]]:
    h, w = input_h, input_w
    breakdown: list[Tally] = []

    # stem: H,W -> H/2,W/2
    breakdown.append(stem_block_flops(3, STEM_CHANNELS, h, w))
    h, w = h // 2, w // 2

    # encoder stages
    prev_c = STEM_CHANNELS
    stage_spatial = []  # for later decoder skip-connections
    stage_spatial.append((STEM_CHANNELS, h, w))
    for stage_idx, (out_c, depth) in enumerate(zip(ENCODER_CHANNELS, depths, strict=True)):
        h, w = h // 2, w // 2
        breakdown += encoder_stage_flops(
            prev_c, out_c, h, w, depth,
            global_branch_type=global_branch_type,
            with_mamba_branch=with_mamba_branch,
            with_bidirectional_gate=with_bidirectional_gate,
        )
        stage_spatial.append((out_c, h, w))
        prev_c = out_c

    # decoder topology mirrors v2lite_segmentor.py:
    # d4 = decoder4(e4, e3); d3 = decoder3(d4, e2); d2 = decoder2(d3, e1); d1 = decoder1(d2, stem)
    _, h_e4, w_e4 = stage_spatial[-1]  # e4
    skip_specs = [stage_spatial[-2], stage_spatial[-3], stage_spatial[-4], stage_spatial[-5]]
    # decoder upsamples to skip spatial:
    cur_c = ENCODER_CHANNELS[3]  # e4 channels
    for i, (skip_c, skip_h, skip_w) in enumerate(skip_specs):
        out_c = DECODER_CHANNELS[i]
        breakdown.append(decoder_block_flops(cur_c, skip_c, out_c, skip_h, skip_w))
        cur_c = out_c

    # seg head operates at d1 spatial == stem spatial
    _, h_stem, w_stem = stage_spatial[0]
    breakdown.append(seg_head_flops(DECODER_CHANNELS[-1], h_stem, w_stem))
    # boundary head similarly
    if with_boundary_head:
        breakdown.append(boundary_head_flops(DECODER_CHANNELS[-1], h_stem, w_stem))

    total = sum(t.flops for t in breakdown)
    return total, breakdown


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def fmt(x: int) -> str:
    return f"{x / 1e9:.2f} G"


def aggregate_breakdown(breakdown: list[Tally]) -> dict[str, int]:
    agg: dict[str, int] = {}
    for t in breakdown:
        agg[t.label] = agg.get(t.label, 0) + t.flops
    return agg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depths", type=int, nargs=4, default=(2, 2, 6, 2))
    parser.add_argument("--input", type=int, nargs=4, default=(1, 3, 512, 512))
    parser.add_argument("--no-boundary", action="store_true")
    parser.add_argument("--also-1111", action="store_true")
    args = parser.parse_args()

    B, _C, H, W = args.input
    depths = tuple(args.depths)

    print(f"input = {tuple(args.input)}  depths = {depths}  boundary_head = {not args.no_boundary}")
    print(f"(Per-image numbers. Multiply by batch_size={B} to get per-batch.)\n")

    results = {}
    for branch in ["simplified", "true_vmamba_ss2d"]:
        total, breakdown = model_flops(
            H, W, depths,
            global_branch_type=branch,
            with_mamba_branch=True,
            with_bidirectional_gate=True,
            with_boundary_head=not args.no_boundary,
        )
        results[branch] = (total, breakdown)
        agg = aggregate_breakdown(breakdown)
        print(f"--- branch = {branch} ---")
        for label, val in sorted(agg.items(), key=lambda kv: -kv[1]):
            print(f"  {label:<26s} {fmt(val):>10s}   ({100*val/total:5.1f}%)")
        print(f"  {'TOTAL':<26s} {fmt(total):>10s}     (×2 = {fmt(2*total)} for the '1 MAdd = 2 FLOPs' convention)\n")

    sim_total = results["simplified"][0]
    tru_total = results["true_vmamba_ss2d"][0]
    diff = tru_total - sim_total
    pct = 100.0 * diff / sim_total
    print(f"true_vmamba_ss2d - simplified : Δ = {fmt(diff)} ({pct:+.2f}%)")
    print(f"   (×2 convention: Δ = {fmt(2*diff)})")

    if args.also_1111:
        print("\n--- context: depths=(1,1,1,1) ---")
        for branch in ["simplified", "true_vmamba_ss2d"]:
            total, _ = model_flops(
                H, W, (1, 1, 1, 1),
                global_branch_type=branch,
                with_mamba_branch=True,
                with_bidirectional_gate=True,
                with_boundary_head=not args.no_boundary,
            )
            print(f"  {branch:>18s}  total = {fmt(total)}   (×2 = {fmt(2*total)})")


if __name__ == "__main__":
    main()
