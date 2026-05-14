#!/usr/bin/env python3
"""Run deterministic WHU final-candidate comparison.

Only compares:
  - simplified + boundary
  - true_vmamba_ss2d + boundary
for seeds 42 / 123 / 3407.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path("/root/autodl-tmp/project-building")
OUT_ROOT = PROJECT_ROOT / "outputs" / "whu_final_deterministic_compare"
SEEDS = (42, 123, 3407)


@dataclass(frozen=True)
class RunSpec:
    model_key: str
    seed: int
    config: str

    @property
    def exp_name(self) -> str:
        return f"whu_deterministic_{self.model_key}_seed{self.seed}"

    @property
    def output_dir(self) -> Path:
        return OUT_ROOT / self.model_key / f"seed_{self.seed}"


RUNS = [
    RunSpec("simplified_boundary", seed, "configs/whu_v2lite_boundary.yaml")
    for seed in SEEDS
] + [
    RunSpec("true_vmamba_boundary", seed, "configs/whu_true_vmamba_boundary.yaml")
    for seed in SEEDS
]


def run_one(spec: RunSpec, dry_run: bool) -> None:
    metrics_path = spec.output_dir / "test_metrics.json"
    if metrics_path.exists():
        print(f"[skip] {spec.exp_name}: metrics already exist at {metrics_path}", flush=True)
        return
    cmd = [
        sys.executable,
        "scripts/run_boundary_train.py",
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
    if not dry_run:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        os.environ["LD_LIBRARY_PATH"] = f"{conda_prefix}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("OMP_NUM_THREADS", "4")

    for spec in RUNS:
        run_one(spec, args.dry_run)


if __name__ == "__main__":
    main()
