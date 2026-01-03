#!/usr/bin/env bash
# Retrain ML models and export champion pipeline
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=========================================================="
echo " Starting Credit Card Churn Model Training & Evaluation"
echo "=========================================================="

cd "$PROJECT_ROOT"
python3 src/train.py

echo " Training pipeline completed successfully."
echo " Artifacts saved in: $PROJECT_ROOT/models/"
