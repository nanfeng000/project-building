from __future__ import annotations

import torch.nn as nn

import torch

from models.blocks import (
    BiCrossGateFusion,
    GlobalMambaBlock,
    GlobalSS2DBlock,
    GlobalTrueSS2DBlock,
    LocalCNNBlock,
    StageDownsample,
    StemBlock,
)


def build_global_branch(branch_type: str, channels: int, dropout: float) -> nn.Module:
    branch_type = branch_type.lower()
    if branch_type == "simplified":
        return GlobalMambaBlock(channels, dropout=dropout)
    if branch_type in {"ss2d", "ss2d_minimal"}:
        return GlobalSS2DBlock(channels, dropout=dropout)
    if branch_type == "true_vmamba_ss2d":
        return GlobalTrueSS2DBlock(channels, dropout=dropout)
    raise ValueError(
        f"Unsupported global_branch_type: {branch_type}. "
        "Expected 'simplified', 'ss2d_minimal'/'ss2d', or 'true_vmamba_ss2d'."
    )


class V2LiteEncoderBlock(nn.Module):
    """A single ``{local + global -> BiCGF}`` block at fixed channel count.

    A stage is composed of one downsample followed by ``depth`` such blocks.
    """

    def __init__(
        self,
        channels: int,
        dropout: float = 0.0,
        with_mamba_branch: bool = True,
        with_bidirectional_gate: bool = True,
        global_branch_type: str = "simplified",
    ) -> None:
        super().__init__()
        self.with_mamba_branch = with_mamba_branch

        self.local_branch = LocalCNNBlock(channels, dropout=dropout)
        self.global_branch = (
            build_global_branch(global_branch_type, channels, dropout) if with_mamba_branch else None
        )
        self.fusion = BiCrossGateFusion(channels, with_bidirectional_gate=with_bidirectional_gate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        local_feat = self.local_branch(x)
        global_feat = self.global_branch(x) if self.global_branch is not None else local_feat

        if self.global_branch is None:
            return local_feat + residual
        return self.fusion(local_feat, global_feat, residual)


class V2LiteEncoderStage(nn.Module):
    """One hierarchical stage = ``StageDownsample`` + ``depth`` × ``V2LiteEncoderBlock``.

    The downsample changes ``in_channels -> out_channels`` and halves the
    spatial resolution once per stage. The depth-stack then iteratively
    refines features at the same resolution/channel count.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int = 1,
        dropout: float = 0.0,
        with_mamba_branch: bool = True,
        with_bidirectional_gate: bool = True,
        global_branch_type: str = "simplified",
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"Stage depth must be >= 1, got {depth}.")
        self.depth = depth
        self.with_mamba_branch = with_mamba_branch

        self.downsample = StageDownsample(in_channels, out_channels)
        self.blocks = nn.ModuleList(
            [
                V2LiteEncoderBlock(
                    channels=out_channels,
                    dropout=dropout,
                    with_mamba_branch=with_mamba_branch,
                    with_bidirectional_gate=with_bidirectional_gate,
                    global_branch_type=global_branch_type,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        for block in self.blocks:
            x = block(x)
        return x


def _remap_legacy_stage_keys(state_dict: dict, prefix: str = "") -> dict:
    """Rewrite legacy v2-lite encoder state_dict keys to the new depth-aware layout.

    Old layout (single block per stage, depth=1 implicit):
        encoder.stageN.local_branch.*
        encoder.stageN.global_branch.*
        encoder.stageN.fusion.*

    New layout (explicit `blocks` ModuleList; depth>=1):
        encoder.stageN.blocks.0.local_branch.*
        encoder.stageN.blocks.0.global_branch.*
        encoder.stageN.blocks.0.fusion.*

    Only the legacy-style keys (no `.blocks.` segment after `stageN.`) are
    rewritten; ``encoder.stageN.downsample.*`` keys are kept unchanged.

    The ``prefix`` argument lets this helper work both on bare-encoder state
    dicts (``stageN.*``) and on segmentor state dicts (``encoder.stageN.*``).
    """
    out = {}
    legacy_subkeys = ("local_branch", "global_branch", "fusion")
    for k, v in state_dict.items():
        rewritten = k
        for stage_idx in (1, 2, 3, 4):
            head = f"{prefix}stage{stage_idx}."
            if not k.startswith(head):
                continue
            tail = k[len(head):]
            if tail.split(".", 1)[0] in legacy_subkeys:
                rewritten = f"{head}blocks.0.{tail}"
            break
        out[rewritten] = v
    return out


class MDUV2LiteEncoder(nn.Module):
    """
    Frozen encoder topology for v2-lite:
    stem -> 4 hierarchical stages
    """

    def __init__(
        self,
        in_channels: int = 3,
        stem_channels: int = 64,
        encoder_channels: tuple[int, int, int, int] = (96, 192, 384, 512),
        depths: tuple[int, int, int, int] = (1, 1, 1, 1),
        dropout: float = 0.0,
        with_mamba_branch: bool = True,
        with_bidirectional_gate: bool = True,
        global_branch_type: str = "simplified",
    ) -> None:
        super().__init__()
        if len(depths) != 4:
            raise ValueError(f"`depths` must have 4 entries, got {depths}")
        self.depths = tuple(depths)
        self.stem = StemBlock(in_channels=in_channels, out_channels=stem_channels)

        c1, c2, c3, c4 = encoder_channels
        d1, d2, d3, d4 = depths
        self.stage1 = V2LiteEncoderStage(
            stem_channels,
            c1,
            depth=d1,
            dropout=dropout,
            with_mamba_branch=with_mamba_branch,
            with_bidirectional_gate=with_bidirectional_gate,
            global_branch_type=global_branch_type,
        )
        self.stage2 = V2LiteEncoderStage(
            c1,
            c2,
            depth=d2,
            dropout=dropout,
            with_mamba_branch=with_mamba_branch,
            with_bidirectional_gate=with_bidirectional_gate,
            global_branch_type=global_branch_type,
        )
        self.stage3 = V2LiteEncoderStage(
            c2,
            c3,
            depth=d3,
            dropout=dropout,
            with_mamba_branch=with_mamba_branch,
            with_bidirectional_gate=with_bidirectional_gate,
            global_branch_type=global_branch_type,
        )
        self.stage4 = V2LiteEncoderStage(
            c3,
            c4,
            depth=d4,
            dropout=dropout,
            with_mamba_branch=with_mamba_branch,
            with_bidirectional_gate=with_bidirectional_gate,
            global_branch_type=global_branch_type,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        stem = self.stem(x)      # [B, 64, 256, 256]
        e1 = self.stage1(stem)   # [B, 96, 128, 128]
        e2 = self.stage2(e1)     # [B, 192, 64, 64]
        e3 = self.stage3(e2)     # [B, 384, 32, 32]
        e4 = self.stage4(e3)     # [B, 512, 16, 16]

        return {
            "stem": stem,
            "e1": e1,
            "e2": e2,
            "e3": e3,
            "e4": e4,
        }

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Backwards-compatible loading for checkpoints saved before the
        # depth-aware refactor. Only legacy keys (without a `.blocks.`
        # segment under stageN) are rewritten in-place; new checkpoints
        # pass through unchanged.
        legacy_present = any(
            k.startswith(f"{prefix}stage")
            and ".blocks." not in k[len(prefix):]
            and any(sub in k for sub in (".local_branch.", ".global_branch.", ".fusion."))
            for k in state_dict
        )
        if legacy_present and self.depths == (1, 1, 1, 1):
            remapped = _remap_legacy_stage_keys(state_dict, prefix=prefix)
            # Only replace the keys that belong to this module's subtree to
            # avoid surprising callers that pass shared dicts.
            for k in list(state_dict.keys()):
                if k.startswith(prefix):
                    state_dict.pop(k)
            for k, v in remapped.items():
                if k.startswith(prefix):
                    state_dict[k] = v

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
