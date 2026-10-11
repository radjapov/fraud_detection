#!/usr/bin/env bash
# NTCC experiments, run from the project root:  bash experiments/run_all.sh
# Every stage is resumable: if it stops, run the same command again.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
uv pip install -q xgboost catboost optuna lime
mkdir -p runs
python -m experiments.run_models --stage all --trials 25 --seeds 0 1 2 3 4 --out runs/models 2>&1 | tee -a runs/models.log
python -m experiments.run_protocols --out runs/protocols 2>&1 | tee -a runs/protocols.log
python -m experiments.run_explain --pipelines runs/models/pipelines --out runs/explain 2>&1 | tee -a runs/explain.log
echo "ALL DONE"
