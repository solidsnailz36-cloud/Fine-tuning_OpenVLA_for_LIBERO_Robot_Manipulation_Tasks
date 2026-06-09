#!/usr/bin/env bash
set -e

LOG_ROOT=${1:-"experiments/logs/EVAL-libero_spatial-openvla-2026_06_04-17_12_40--official_ft_libero_spatial_horizon_1x"}

RUN_DIR="results/run_$(date +"%Y_%m_%d-%H_%M_%S")"
mkdir -p "$RUN_DIR"

echo "LOG_ROOT = $LOG_ROOT"
echo "RUN_DIR  = $RUN_DIR"

python scripts/summarize_results.py \
  --log_root "$LOG_ROOT" \
  --output "$RUN_DIR/results_summary.csv"

python scripts/analyze_rollout_actions.py \
  --log_root "$LOG_ROOT" \
  --output "$RUN_DIR/latency_summary.csv"

python scripts/classify_failures.py \
  --log_root "$LOG_ROOT" \
  --output "$RUN_DIR/failure_cases.csv"

python scripts/plot_action_traces.py \
  --log_root "$LOG_ROOT" \
  --output_dir "$RUN_DIR"

echo ""
echo "Done."
echo "Saved results to:"
echo "$RUN_DIR"
