# Synthetic competition starter

This repository builds a synthetic version of the 2026 quant competition data and a minimal end-to-end model pipeline.

## What it does

- Generates competition-shaped training and test tables with `row_id`, `time_id`, `asset_id`, `feature_*`, `responder_*`, `target`, and `weight`
- Makes `weight` depend on a small set of known features so feature recovery can be checked
- Makes `target` a continuous regression target centered around zero with positive and negative values
- Trains a weighted linear ensemble and exports a JSON model bundle
- Runs offline inference through `main.py`

## Files

- `generate_data.py`: create the synthetic dataset
- `train.py`: train the weighted target model and auxiliary weight model
- `main.py`: load the saved model bundle and produce a submission CSV
- `synthetic_competition/`: reusable data, metrics, model, and analysis code

## Quick start

```bash
python generate_data.py --output-dir data/synthetic
python train.py --data-dir data/synthetic --artifacts-dir artifacts
python main.py --data-dir data/synthetic --model-path artifacts/model_bundle.json --output-path submission.csv
```

## Notes

- If `pyarrow` is installed, the data writer will use Parquet automatically. Otherwise it falls back to CSV.
- The training split uses a strict time-based holdout to avoid future leakage.
- `weight` is used during target training and also trained as an auxiliary prediction target so the important feature set can be inspected.

