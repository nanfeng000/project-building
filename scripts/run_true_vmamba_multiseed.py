#!/usr/bin/env python3
"""Run true VMamba final-candidate multi-seed screening.

This script intentionally does not introduce new model variants. It only runs
the already-defined training entrypoints with seed/output overrides, and can be
re-run safely after interruptions.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path("/root/autodl-tmp/project-building")
OUT_ROOT = PROJECT_ROOT / "outputs" / "true_vmamba_multiseed"


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    model_key: str
    seed: int
    script: str
    config: str

    @property
    def exp_name(self) -> str:
        return f"{self.dataset}_{self.model_key}_seed{self.seed}"

    @property
    def output_dir(self) -> Path:
        return OUT_ROOT / self.dataset / self.model_key / f"seed_{self.seed}"


SEEDS_TO_TRAIN = (123, 3407)

RUNS = [
    RunSpec("whu", "simplified_boundary", seed, "scripts/run_boundary_train.py", "configs/whu_v2lite_boundary.yaml")
    for seed in SEEDS_TO_TRAIN
] + [
    RunSpec("whu", "true_vmamba_no_boundary", seed, "scripts/run_ablation_train.py", "configs/whu_true_vmamba_C_full_true_vmamba_ss2d.yaml")
    for seed in SEEDS_TO_TRAIN
] + [
    RunSpec("whu", "true_vmamba_boundary", seed, "scripts/run_boundary_train.py", "configs/whu_true_vmamba_boundary.yaml")
    for seed in SEEDS_TO_TRAIN
] + [
    RunSpec("inria", "simplified_boundary", seed, "scripts/run_boundary_train.py", "configs/inria_v2lite_boundary.yaml")
    for seed in SEEDS_TO_TRAIN
] + [
    RunSpec("inria", "true_vmamba_no_boundary", seed, "scripts/run_ablation_train.py", "configs/inria_true_vmamba_C_full_true_vmamba_ss2d.yaml")
    for seed in SEEDS_TO_TRAIN
] + [
    RunSpec("inria", "true_vmamba_boundary", seed, "scripts/run_boundary_train.py", "configs/inria_true_vmamba_boundary.yaml")
    for seed in SEEDS_TO_TRAIN
]


def run_one(spec: RunSpec, dry_run: bool = False) -> None:
    metrics_path = spec.output_dir / "test_metrics.json"
    if metrics_path.exists():
        print(f"[skip] {spec.exp_name}: metrics already exist at {metrics_path}", flush=True)
        return

    cmd = [
        sys.executable,
        spec.script,
        "--config",
        spec.config,
        "--seed",
        str(spec.seed),
        "--output-dir",
        str(spec.output_dir),
        "--experiment-name",
        spec.exp_name,
    ]
    last_ckpt = spec.output_dir / "checkpoints" / "last.pth"
    if last_ckpt.exists():
        cmd += ["--resume", str(last_ckpt)]

    print(f"[run] {spec.exp_name}", flush=True)
    print("      " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    env_lib = os.environ.get("CONDA_PREFIX")
    if env_lib:
        os.environ["LD_LIBRARY_PATH"] = f"{env_lib}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"
    os.environ.setdefault("OMP_NUM_THREADS", "4")

    for spec in RUNS:
        run_one(spec, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
