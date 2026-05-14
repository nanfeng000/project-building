from __future__ import annotations

import math
import ctypes
import importlib.util
import sys
import sysconfig
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _bidirectional_cumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Minimal VSS/Mamba-style global mixing.

    We approximate directional scan behavior by aggregating information
    in both forward and backward directions along one spatial axis.
    This keeps the first v2-lite implementation dependency-free while
    preserving a clear "global branch" inductive bias.
    """

    steps = x.size(dim)
    denom = torch.arange(1, steps + 1, device=x.device, dtype=x.dtype)
    view_shape = [1] * x.ndim
    view_shape[dim] = steps
    denom = denom.view(*view_shape)

    forward = torch.cumsum(x, dim=dim) / denom
    backward = torch.flip(torch.cumsum(torch.flip(x, dims=[dim]), dim=dim) / denom, dims=[dim])
    return 0.5 * (forward + backward)


class GlobalMambaBlock(nn.Module):
    """
    Simplified global branch for v2-lite.

    This is not a full external Mamba/VSS dependency. Instead, it keeps
    the same architectural role:
    - project features
    - perform directional global mixing along H/W
    - gate and project back
    """

    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.in_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(1, channels)
        self.mix_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.local_refine = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.gate_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out_dtype = x.dtype
        device_type = x.device.type if x.is_cuda else "cpu"

        # The global branch is the numerically most sensitive part of v2-lite.
        # Full training showed AMP-only NaNs, while float32 re-forward stayed
        # finite. We therefore keep the entire branch in float32 and cast the
        # result back to the outer dtype afterwards. The topology is unchanged;
        # only the precision policy is tightened for stability.
        with torch.autocast(device_type=device_type, enabled=False):
            x = x.float()
            residual_fp32 = x
            x = self.in_proj(x)
            x = self.norm(x)

            scan_h = _bidirectional_cumsum(x, dim=2)
            scan_w = _bidirectional_cumsum(x, dim=3)
            mixed = 0.5 * (scan_h + scan_w)
            mixed = self.mix_proj(mixed)
            mixed = mixed + self.local_refine(mixed)

            gate = torch.sigmoid(self.gate_proj(x))
            x = self.out_proj(self.act(mixed) * gate)
            x = self.dropout(x)
            x = x + residual_fp32

        return x.to(dtype=out_dtype)


class GlobalSS2DBlock(nn.Module):
    """
    Dependency-free SS2D-style global branch for the screening experiment.

    This is an implementation close to the standard VSS/PixMamba SS2D block
    layout, but intentionally keeps the scan core minimal and stable for this
    project:
    - channels-last LayerNorm
    - input projection split into content x and gate z
    - depthwise convolution before scanning
    - four directional 2D routes: HW, WH, reverse-HW, reverse-WH
    - route merge, output normalization, gated projection, residual

    Difference from official SS2D: this version does not depend on the CUDA
    selective_scan operator or its dt/B/C parameterization. The directional
    scan is a normalized cumulative scan, so it is "SS2D-style" rather than a
    bit-for-bit official selective state-space layer. The goal is a controlled,
    minimal, stable replacement for the current simplified global branch.
    """

    def __init__(
        self,
        channels: int,
        dropout: float = 0.0,
        expand_ratio: float = 1.0,
    ) -> None:
        super().__init__()
        inner_channels = max(channels, int(channels * expand_ratio))
        self.channels = channels
        self.inner_channels = inner_channels

        self.norm = nn.LayerNorm(channels)
        self.in_proj = nn.Linear(channels, inner_channels * 2, bias=False)
        self.dwconv = nn.Conv2d(
            inner_channels,
            inner_channels,
            kernel_size=3,
            padding=1,
            groups=inner_channels,
            bias=True,
        )
        self.route_logits = nn.Parameter(torch.zeros(4))
        self.out_norm = nn.LayerNorm(inner_channels)
        self.out_proj = nn.Linear(inner_channels, channels, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.SiLU()

    @staticmethod
    def _normalized_cumsum(seqs: torch.Tensor) -> torch.Tensor:
        """Cumulative directional scan for [B, K, C, L] sequences."""
        steps = seqs.size(-1)
        denom = torch.arange(1, steps + 1, device=seqs.device, dtype=seqs.dtype)
        denom = denom.view(1, 1, 1, steps)
        return torch.cumsum(seqs, dim=-1) / denom

    def _ss2d_scan(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        seq_hw = x.flatten(2)
        seq_wh = x.transpose(2, 3).contiguous().flatten(2)
        seqs = torch.stack(
            [
                seq_hw,
                seq_wh,
                torch.flip(seq_hw, dims=[-1]),
                torch.flip(seq_wh, dims=[-1]),
            ],
            dim=1,
        )
        scanned = self._normalized_cumsum(seqs)

        y_hw = scanned[:, 0].view(b, c, h, w)
        y_wh = scanned[:, 1].view(b, c, w, h).transpose(2, 3).contiguous()
        y_hw_rev = torch.flip(scanned[:, 2], dims=[-1]).view(b, c, h, w)
        y_wh_rev = torch.flip(scanned[:, 3], dims=[-1]).view(b, c, w, h).transpose(2, 3).contiguous()

        routes = torch.stack([y_hw, y_wh, y_hw_rev, y_wh_rev], dim=1)
        weights = F.softmax(self.route_logits, dim=0).view(1, 4, 1, 1, 1)
        return (routes * weights).sum(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out_dtype = x.dtype
        device_type = x.device.type if x.is_cuda else "cpu"

        # Match the existing global branch precision policy: keep global scan
        # math in fp32 to avoid AMP-only instability in long directional scans.
        with torch.autocast(device_type=device_type, enabled=False):
            x = x.float()
            residual_fp32 = x
            x_nhwc = x.permute(0, 2, 3, 1).contiguous()
            x_nhwc = self.norm(x_nhwc)
            x_proj, z = self.in_proj(x_nhwc).chunk(2, dim=-1)

            x_proj = x_proj.permute(0, 3, 1, 2).contiguous()
            x_proj = self.act(self.dwconv(x_proj))
            x_proj = self._ss2d_scan(x_proj)
            x_proj = x_proj.permute(0, 2, 3, 1).contiguous()
            x_proj = self.out_norm(x_proj)

            gated = x_proj * self.act(z)
            out = self.out_proj(gated)
            out = self.dropout(out)
            out = out.permute(0, 3, 1, 2).contiguous()
            out = out + residual_fp32

        return out.to(dtype=out_dtype)


def _load_selective_scan_fn():
    conda_libstdcpp = Path(sys.prefix) / "lib" / "libstdc++.so.6"
    if conda_libstdcpp.exists():
        try:
            ctypes.CDLL(str(conda_libstdcpp), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass

    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        return selective_scan_fn
    except Exception as exc:  # pragma: no cover - depends on optional CUDA extension
        direct_import_error = exc

    # Some mamba-ssm releases import heavyweight model/HuggingFace modules from
    # mamba_ssm.__init__ before reaching the ops package. For this project we
    # only need the compiled selective_scan operator, so try loading the ops file
    # directly. This still requires real `selective_scan_cuda`; it is not a
    # fallback to the minimal/cumsum implementation.
    for site_dir in sys.path + [sysconfig.get_paths().get("purelib", "")]:
        if not site_dir:
            continue
        interface_path = Path(site_dir) / "mamba_ssm" / "ops" / "selective_scan_interface.py"
        if not interface_path.exists():
            continue
        try:
            spec = importlib.util.spec_from_file_location("_project_mamba_selective_scan_interface", interface_path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.selective_scan_fn
        except Exception as exc:
            raise RuntimeError(
                "global_branch_type='true_vmamba_ss2d' requires the real mamba-ssm "
                "selective_scan extension. Found selective_scan_interface.py but "
                "could not load it with the compiled CUDA operator; no minimal/cumsum "
                "fallback is used."
            ) from exc

    raise RuntimeError(
        "global_branch_type='true_vmamba_ss2d' requires the real mamba-ssm "
        "selective_scan extension. Install a working `mamba-ssm` build for "
        "the current PyTorch/CUDA environment; no minimal/cumsum fallback is used."
    ) from direct_import_error


class GlobalTrueSS2DBlock(nn.Module):
    """
    VMamba/SS2D-style global branch backed by real selective_scan.

    This block follows the core SS2D data flow used by VMamba:
    channels-last norm -> x/z projection -> depthwise conv -> four 2D routes
    (HW, WH, reverse-HW, reverse-WH) -> selective_scan over each route ->
    directional merge -> output norm -> z gate -> output projection.

    Unlike GlobalSS2DBlock, this module does not approximate the scan with
    cumulative sums. It requires `mamba_ssm.ops.selective_scan_interface` and
    intentionally fails at construction if the CUDA selective_scan operator is
    unavailable.
    """

    def __init__(
        self,
        channels: int,
        dropout: float = 0.0,
        d_state: int = 16,
        dt_rank: int | None = None,
        expand_ratio: float = 1.0,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
    ) -> None:
        super().__init__()
        self.selective_scan_fn = _load_selective_scan_fn()
        self.channels = channels
        self.d_state = d_state
        self.dt_rank = dt_rank or math.ceil(channels / 16)
        self.inner_channels = max(channels, int(channels * expand_ratio))
        self.num_routes = 4

        d_inner = self.inner_channels
        self.norm = nn.LayerNorm(channels)
        self.in_proj = nn.Linear(channels, d_inner * 2, bias=False)
        self.dwconv = nn.Conv2d(
            d_inner,
            d_inner,
            kernel_size=3,
            padding=1,
            groups=d_inner,
            bias=True,
        )
        self.act = nn.SiLU()

        # Per-route projections produce dt, B, C for the selective state scan.
        self.x_proj_weight = nn.Parameter(torch.empty(self.num_routes, self.dt_rank + 2 * d_state, d_inner))
        self.dt_projs_weight = nn.Parameter(torch.empty(self.num_routes, d_inner, self.dt_rank))
        self.dt_projs_bias = nn.Parameter(torch.empty(self.num_routes, d_inner))

        # A is parameterized in log-space and negated in forward, matching Mamba/VMamba.
        a = torch.arange(1, d_state + 1, dtype=torch.float32)
        a = a[None, :].repeat(self.num_routes * d_inner, 1)
        self.A_logs = nn.Parameter(torch.log(a))
        self.Ds = nn.Parameter(torch.ones(self.num_routes * d_inner, dtype=torch.float32))

        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, channels, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.reset_parameters(dt_min=dt_min, dt_max=dt_max, dt_init_floor=dt_init_floor)

    def reset_parameters(self, dt_min: float, dt_max: float, dt_init_floor: float) -> None:
        nn.init.xavier_uniform_(self.x_proj_weight)
        dt_scale = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_projs_weight, -dt_scale, dt_scale)

        dt = torch.exp(
            torch.rand(self.num_routes, self.inner_channels) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse softplus so softplus(dt_bias) starts in [dt_min, dt_max].
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_projs_bias.copy_(inv_dt)

    def _make_routes(self, x: torch.Tensor) -> torch.Tensor:
        b, d, h, w = x.shape
        l = h * w
        route_hw = x.view(b, d, l)
        route_wh = x.transpose(2, 3).contiguous().view(b, d, l)
        routes = torch.stack([route_hw, route_wh], dim=1)
        return torch.cat([routes, torch.flip(routes, dims=[-1])], dim=1)

    def _merge_routes(self, ys: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, _, d, l = ys.shape
        y_hw = ys[:, 0]
        y_wh = ys[:, 1].view(b, d, w, h).transpose(2, 3).contiguous().view(b, d, l)
        y_rev = torch.flip(ys[:, 2:4], dims=[-1])
        y_hw_rev = y_rev[:, 0]
        y_wh_rev = y_rev[:, 1].view(b, d, w, h).transpose(2, 3).contiguous().view(b, d, l)
        return y_hw + y_wh + y_hw_rev + y_wh_rev

    def _ss2d_selective_scan(self, x: torch.Tensor) -> torch.Tensor:
        b, d, h, w = x.shape
        l = h * w
        k = self.num_routes

        xs = self._make_routes(x)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, bs, cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.contiguous().view(b, k * d, l)
        dts = dts.contiguous().view(b, k * d, l)
        bs = bs.contiguous().float()
        cs = cs.contiguous().float()
        a = -torch.exp(self.A_logs.float())
        ds = self.Ds.float()
        dt_bias = self.dt_projs_bias.float().view(-1)

        ys = self.selective_scan_fn(
            xs.float(),
            dts.float(),
            a,
            bs,
            cs,
            ds,
            z=None,
            delta_bias=dt_bias,
            delta_softplus=True,
            return_last_state=False,
        )
        ys = ys.view(b, k, d, l)
        y = self._merge_routes(ys, h, w)
        return y.view(b, d, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out_dtype = x.dtype
        device_type = x.device.type if x.is_cuda else "cpu"

        with torch.autocast(device_type=device_type, enabled=False):
            x = x.float()
            residual_fp32 = x
            x_nhwc = x.permute(0, 2, 3, 1).contiguous()
            x_nhwc = self.norm(x_nhwc)
            x_proj, z = self.in_proj(x_nhwc).chunk(2, dim=-1)

            x_proj = x_proj.permute(0, 3, 1, 2).contiguous()
            x_proj = self.act(self.dwconv(x_proj))
            x_proj = self._ss2d_selective_scan(x_proj)
            x_proj = x_proj.permute(0, 2, 3, 1).contiguous()
            x_proj = self.out_norm(x_proj)

            out = self.out_proj(x_proj * self.act(z))
            out = self.dropout(out)
            out = out.permute(0, 3, 1, 2).contiguous()
            out = out + residual_fp32

        return out.to(dtype=out_dtype)
