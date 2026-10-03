#!/usr/bin/env bash
# Run the full pipeline: synthetic data -> cleaning -> EDA -> model training & evaluation
set -euo pipefail
cd "$(dirname "$0")"
python src/generate_data.py --patients 20000 --out data/raw/encounters.csv
python src/clean.py --raw data/raw/encounters.csv --out data/processed/encounters_clean.parquet
python src/eda.py --data data/processed/encounters_clean.parquet
python src/train.py --data data/processed/encounters_clean.parquet
echo "Done. See outputs/ for figures, eda_findings.csv and metrics.json"
