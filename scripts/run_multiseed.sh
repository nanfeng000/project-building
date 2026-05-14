#!/bin/bash
# Multi-seed robustness runner. Sequentially trains all (dataset × model × seed) combos
# that have not been trained yet. Existing seed=42 runs are reused.
#
# Total: 2 datasets × 3 models × 2 extra seeds = 12 runs.
#
# Each run writes into outputs/multiseed_robustness/<dataset>/<model>/seed<S>/
# (shares same structure as the single-run output so reports can aggregate easily).

set -e

PROJECT_ROOT=/root/autodl-tmp/project-building
BASE_OUT=$PROJECT_ROOT/outputs/multiseed_robustness
MASTER_LOG=$PROJECT_ROOT/logs/multiseed_runner.log
mkdir -p "$BASE_OUT" "$PROJECT_ROOT/logs"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate building

run_job() {
  local ds=$1          # whu or inria
  local model=$2       # unet / cfull / cbnd
  local seed=$3
  local cfg=$4
  local script=$5      # run_ablation_train.py or run_boundary_train.py
  local exp_name="${ds}_${model}_seed${seed}"
  local out_dir="$BASE_OUT/$ds/$model/seed${seed}"
  local log="$PROJECT_ROOT/logs/multiseed_${exp_name}.log"

  if [ -f "$out_dir/test_metrics.json" ]; then
    echo "[$(date '+%F %T')] SKIP $exp_name (already done)" | tee -a "$MASTER_LOG"
    return
  fi

  mkdir -p "$out_dir"
  echo "[$(date '+%F %T')] START $exp_name" | tee -a "$MASTER_LOG"
  cd "$PROJECT_ROOT"
  python -u scripts/$script \
    --config "$cfg" \
    --seed "$seed" \
    --output-dir "$out_dir" \
    --experiment-name "$exp_name" \
    > "$log" 2>&1
  echo "[$(date '+%F %T')] DONE  $exp_name" | tee -a "$MASTER_LOG"
}

# Seed=42 runs already exist in outputs/{whu_unet_baseline, whu_v2lite, whu_v2lite_boundary,
# inria_unet_baseline, inria_v2lite_full, inria_v2lite_boundary}. We'll reuse them.
# Here we only train two extra seeds: 123 and 3407.

# To save time: run all WHU jobs first (faster), then Inria.
SEEDS=(123 3407)

for SEED in "${SEEDS[@]}"; do
  # WHU
  run_job whu   unet  "$SEED" configs/whu_unet_baseline.yaml    run_ablation_train.py
  run_job whu   cfull "$SEED" configs/whu_v2lite.yaml           run_ablation_train.py
  run_job whu   cbnd  "$SEED" configs/whu_v2lite_boundary.yaml  run_boundary_train.py
done

for SEED in "${SEEDS[@]}"; do
  # Inria
  run_job inria unet  "$SEED" configs/inria_unet_baseline.yaml  run_ablation_train.py
  run_job inria cfull "$SEED" configs/inria_v2lite_full.yaml    run_ablation_train.py
  run_job inria cbnd  "$SEED" configs/inria_v2lite_boundary.yaml run_boundary_train.py
done

echo "[$(date '+%F %T')] ALL MULTISEED RUNS FINISHED" | tee -a "$MASTER_LOG"
