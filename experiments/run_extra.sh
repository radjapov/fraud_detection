#!/usr/bin/env bash
# Additional NTCC experiments, run from the project root:  bash experiments/run_extra.sh
# Needs runs/models from run_all.sh. Every stage is resumable: if it stops, run it again.
#   calibration  seconds        Brier / ECE / precision at top-k alerts       -> runs/calibration
#   LIME         ~20-30 min     LIME settings: samples, kernel, discretisation -> runs/lime_sensitivity
#   ablation     ~20-30 min     LightGBM without drifting features, 3 seeds    -> runs/ablation
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p runs
python -m experiments.run_calibration --runs runs --out runs/calibration 2>&1 | tee -a runs/calibration.log
python -m experiments.run_lime_sensitivity --models LightGBM --out runs/lime_sensitivity 2>&1 | tee -a runs/lime_sensitivity.log
python -m experiments.run_ablation --seeds 0 1 2 --out runs/ablation 2>&1 | tee -a runs/ablation.log
echo "ALL DONE"
