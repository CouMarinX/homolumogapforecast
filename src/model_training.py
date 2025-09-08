"""Model training and evaluation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


@dataclass
class FoldMetrics:
    rmse: float
    mae: float
    r2: float


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    seeds: Iterable[int] = (1000, 2000, 3000, 4000, 5000),
    n_splits: int = 5,
) -> Dict[int, List[FoldMetrics]]:
    """Perform K-fold cross-validation for multiple random seeds."""
    results: Dict[int, List[FoldMetrics]] = {}

    for seed in seeds:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        metrics: List[FoldMetrics] = []
        for train_idx, test_idx in kf.split(X):
            model = RandomForestRegressor(
                n_estimators=100,
                random_state=seed,
                n_jobs=-1,
            )
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict(X[test_idx])
            metrics.append(
                FoldMetrics(
                    rmse=mean_squared_error(y[test_idx], preds, squared=False),
                    mae=mean_absolute_error(y[test_idx], preds),
                    r2=r2_score(y[test_idx], preds),
                )
            )
        results[seed] = metrics
    return results


def summarize_results(results: Dict[int, List[FoldMetrics]]) -> None:
    """Print averaged metrics for each seed."""
    for seed, folds in results.items():
        rmse = np.mean([f.rmse for f in folds])
        mae = np.mean([f.mae for f in folds])
        r2 = np.mean([f.r2 for f in folds])
        print(f"Seed {seed}: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")


def feature_importance(X: np.ndarray, y: np.ndarray, seed: int = 1000) -> np.ndarray:
    """Train a random forest model and return feature importances."""
    model = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
    model.fit(X, y)
    return model.feature_importances_


__all__ = [
    "FoldMetrics",
    "cross_validate",
    "summarize_results",
    "feature_importance",
]
