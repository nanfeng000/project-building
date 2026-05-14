"""
Compute FLOPs (and params) for v2-lite at encoder depths=(2,2,6,2) on CPU.

Compares:
  - global_branch_type = 'simplified'        (CPU-compatible directly)
  - global_branch_type = 'true_vmamba_ss2d'  (CPU stub + analytical scan FLOPs)

Both at input size 1x3x512x512, with boundary head, with bidirectional gate.

GPU is not required. The selective_scan_fn (mamba_ssm CUDA op) is monkey-patched
to a CPU shape-preserving stub so the network can run forward on CPU. Its
compute cost is added analytically afterwards.

FLOP convention used here: fvcore counts 1 multiply-add as 1 FLOP (i.e. MACs).
That matches the convention typically reported in segmentation papers like
VMamba/SegFormer/Mamba-based works. We also print the 2*MACs version for
those who prefer "1 MAdd = 2 FLOPs".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
# CPU stub for selective_scan_fn so we can construct GlobalTrueSS2DBlock      #
# without a CUDA build of mamba-ssm. The stub returns zeros of the expected   #
# shape so the forward pass completes. We add the scan FLOPs back manually.   #
# --------------------------------------------------------------------------- #
from models.blocks import global_mamba_block as _gmb


def _cpu_selective_scan_stub(
    u, delta, A, B, C, D=None, z=None, delta_bias=None,
    delta_softplus=False, return_last_state=False,
):
    # u: [B, K*d_inner, L]  delta: [B, K*d_inner, L]
    # Output y matches u's shape.
    out = torch.zeros_like(u)
    if return_last_state:
        b = u.shape[0]
        d = u.shape[1]
        n = A.shape[-1] if A.ndim >= 2 else 1
        last_state = torch.zeros(b, d, n, dtype=u.dtype, device=u.device)
        return out, last_state
    return out


def _patched_loader():
    return _cpu_selective_scan_stub


_gmb._load_selective_scan_fn = _patched_loader  # noqa: SLF001


# Now we can safely import models that may build the true_vmamba_ss2d branch.
from models import build_model  # noqa: E402
from fvcore.nn import FlopCountAnalysis  # noqa: E402


# --------------------------------------------------------------------------- #
# Encoder stage spatial/channel topology used to add analytical scan FLOPs.   #
# --------------------------------------------------------------------------- #
# Each entry: (channels_after_downsample, spatial_HxW_after_downsample)
# Stem brings 512x512 -> 256x256@64. Each stage further halves to:
STAGE_SHAPES = [
    (96, 128),   # stage1 output spatial 128x128, 96 channels
    (192, 64),   # stage2
    (384, 32),   # stage3
    (512, 16),   # stage4
]


def analytical_ss2d_scan_flops(
    depths: tuple[int, int, int, int],
    expand_ratio: float = 1.0,
    d_state: int = 16,
    num_routes: int = 4,
    one_madd_is_2_flops: bool = False,
) -> int:
    """Analytical FLOPs for the selective_scan core (only the scan recurrence).

    Per (timestep, channel, state):
      4 multiply-adds  (A*h_prev,  dt*B*x,   accumulate,   C*h)

    Total for one route at sequence length L = H*W and inner dim D:
      4 * L * D * d_state  multiply-adds

    Four routes per block, multiplied by per-stage depth and spatial size.

    These FLOPs are IN ADDITION to the einsum/x_proj/dt_proj/dwconv/Linear ops,
    which fvcore already counts (or will count, via the CPU stub trace).
    """
    total_madd = 0
    for depth, (c, hw) in zip(depths, STAGE_SHAPES, strict=True):
        d_inner = max(c, int(c * expand_ratio))
        L = hw * hw
        per_route = 4 * L * d_inner * d_state
        per_block = num_routes * per_route
        total_madd += depth * per_block
    return total_madd * (2 if one_madd_is_2_flops else 1)


def count_model_flops_and_params(model: nn.Module, input_size=(1, 3, 512, 512)) -> tuple[int, int, dict]:
    model.eval()
    x = torch.randn(*input_size)
    with torch.no_grad():
        flops = FlopCountAnalysis(model, x)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        total = int(flops.total())
        by_module = flops.by_module()
    params = sum(p.numel() for p in model.parameters())
    return total, params, by_module


def build_and_measure(branch_type: str, depths: tuple[int, int, int, int]) -> dict:
    model = build_model(
        "v2lite",
        in_channels=3,
        num_classes=1,
        stem_channels=64,
        encoder_channels=[96, 192, 384, 512],
        decoder_channels=[256, 192, 128, 96],
        encoder_depths=depths,
        dropout=0.0,
        with_mamba_branch=True,
        with_bidirectional_gate=True,
        global_branch_type=branch_type,
        with_boundary_head=True,
    )
    flops_madd, params, _by_module = count_model_flops_and_params(model)
    extra_scan_madd = 0
    if branch_type == "true_vmamba_ss2d":
        extra_scan_madd = analytical_ss2d_scan_flops(depths)
    total_madd = flops_madd + extra_scan_madd
    return {
        "branch_type": branch_type,
        "depths": depths,
        "params": params,
        "fvcore_flops_madd": flops_madd,        # standard ops counted by fvcore
        "extra_scan_flops_madd": extra_scan_madd,  # selective_scan recurrence (analytic)
        "total_flops_madd": total_madd,         # convention: 1 MAdd = 1 FLOP
        "total_flops_2x": total_madd * 2,       # convention: 1 MAdd = 2 FLOPs
    }


def format_g(x: int) -> str:
    return f"{x / 1e9:.2f} G"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depths", type=int, nargs=4, default=(2, 2, 6, 2))
    parser.add_argument("--also-1111", action="store_true",
                        help="Also report the historical depths=(1,1,1,1) for context.")
    args = parser.parse_args()

    depths = tuple(args.depths)

    rows = []
    for branch in ["simplified", "true_vmamba_ss2d"]:
        r = build_and_measure(branch, depths)
        rows.append(r)
        print(
            f"[{branch:>18s}] depths={depths}  params={r['params']/1e6:.2f}M  "
            f"fvcore MAdd={format_g(r['fvcore_flops_madd'])}  "
            f"+scan MAdd={format_g(r['extra_scan_flops_madd'])}  "
            f"total MAdd={format_g(r['total_flops_madd'])}  "
            f"(2xFLOPs={format_g(r['total_flops_2x'])})"
        )

    if args.also_1111:
        print("\n--- context: historical depths=(1,1,1,1) ---")
        for branch in ["simplified", "true_vmamba_ss2d"]:
            r = build_and_measure(branch, (1, 1, 1, 1))
            print(
                f"[{branch:>18s}] depths=(1,1,1,1)  params={r['params']/1e6:.2f}M  "
                f"fvcore MAdd={format_g(r['fvcore_flops_madd'])}  "
                f"+scan MAdd={format_g(r['extra_scan_flops_madd'])}  "
                f"total MAdd={format_g(r['total_flops_madd'])}  "
                f"(2xFLOPs={format_g(r['total_flops_2x'])})"
            )

    # Difference
    sim = next(r for r in rows if r["branch_type"] == "simplified")
    tru = next(r for r in rows if r["branch_type"] == "true_vmamba_ss2d")
    diff = tru["total_flops_madd"] - sim["total_flops_madd"]
    pct = 100.0 * diff / max(1, sim["total_flops_madd"])
    print()
    print(f"true_vmamba_ss2d - simplified : Δ MAdd = {format_g(diff)}  ({pct:+.2f}%)")


if __name__ == "__main__":
    main()
