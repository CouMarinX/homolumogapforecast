"""Model training and evaluation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


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
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            model = RandomForestRegressor(
                n_estimators=100,
                random_state=seed,
                n_jobs=-1,
            )
            model.fit(X_train, y[train_idx])
            preds = model.predict(X_test)
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


def feature_importance(
    X: np.ndarray, y: np.ndarray, descriptor_names: Sequence[str], seed: int = 1000
) -> Dict[str, float]:
    """Train a random forest model and return feature importances mapped to descriptors."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
    model.fit(X_scaled, y)
    importances = model.feature_importances_
    return {name: imp for name, imp in zip(descriptor_names, importances)}


__all__ = [
    "FoldMetrics",
    "cross_validate",
    "summarize_results",
    "feature_importance",
]
