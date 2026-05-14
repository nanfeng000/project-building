#!/usr/bin/env bash
# Watcher: wait for DeepLabV3 training to finish, then run qualitative comparison.
# This script is launched in background by the agent and intentionally has no args.

TARGET=/root/autodl-tmp/project-building/outputs/whu_deeplabv3_resnet50_seed42/test_metrics.json
LOG=/root/autodl-tmp/project-building/logs/train_logs/whu_strong_baseline_pipeline.log

mkdir -p /root/autodl-tmp/project-building/logs/train_logs
echo "[pipeline $(date '+%F %T')] waiting for $TARGET" > "$LOG"

while [ ! -f "$TARGET" ]; do
    sleep 60
done

# small grace period so report.md / metrics.csv finish writing
sleep 10
echo "[pipeline $(date '+%F %T')] DeepLabV3 training detected as complete" >> "$LOG"

cd /root/autodl-tmp/project-building
echo "[pipeline $(date '+%F %T')] running scripts/whu_strong_baseline_qualitative.py" >> "$LOG"
/root/miniconda3/envs/building/bin/python scripts/whu_strong_baseline_qualitative.py >> "$LOG" 2>&1
RC=$?
echo "[pipeline $(date '+%F %T')] qualitative pipeline finished, exit_code=${RC}" >> "$LOG"
exit $RC
