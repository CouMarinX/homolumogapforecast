"""Command line interface for HOMO-LUMO gap prediction project."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from data_preparation import load_and_sample, featurize
from model_training import cross_validate, feature_importance, summarize_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HOMO-LUMO gap prediction")
    parser.add_argument("cation", type=Path, help="CSV file with cation molecules")
    parser.add_argument("anion", type=Path, help="CSV file with anion molecules")
    parser.add_argument("neutral", type=Path, help="CSV file with neutral molecules")
    parser.add_argument("radical", type=Path, help="CSV file with radical molecules")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100_000,
        help="Number of samples to draw from each CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("combined_dataset.csv"),
        help="Where to save the combined sampled dataset",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_paths = [args.cation, args.anion, args.neutral, args.radical]

    # Step 1: load and sample data
    combined = load_and_sample(csv_paths, n_samples=args.sample_size, seed=1000)
    combined.to_csv(args.output, index=False)

    # Step 2: featurize
    X, y, smiles = featurize(combined)
    print(f"Generated features for {len(smiles)} molecules")

    # Step 3: model training with cross-validation
    seeds = [1000, 2000, 3000, 4000, 5000]
    results = cross_validate(X, y, seeds=seeds, n_splits=5)
    summarize_results(results)

    # Step 4: evaluate feature importance on full dataset
    importances = feature_importance(X, y, seed=1000)
    top_idx = np.argsort(importances)[::-1][:10]
    print("Top 10 fingerprint bits by importance:")
    for idx in top_idx:
        print(f"  fp{idx}: {importances[idx]:.4f}")


if __name__ == "__main__":
    main()
