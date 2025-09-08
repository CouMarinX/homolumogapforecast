import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_dataset(descriptor_path: str, target_path: str, names_path: str):
    """Load descriptor matrix, target array and descriptor names.

    Parameters
    ----------
    descriptor_path: str
        Path to ``.npy`` file containing ``(n_samples, n_features)`` descriptor
        matrix.
    target_path: str
        Path to ``.npy`` file containing ``(n_samples,)`` target values.
    names_path: str
        Path to text file with one descriptor name per line.
    """
    X = np.load(descriptor_path)
    y = np.load(target_path)
    with open(names_path) as f:
        descriptor_names = [line.strip() for line in f if line.strip()]

    if X.shape[1] != len(descriptor_names):
        raise ValueError(
            "Descriptor name count does not match feature dimension: "
            f"{len(descriptor_names)} names vs {X.shape[1]} columns"
        )

    logging.info(
        "Loaded descriptors from %s with shape %s and targets from %s",
        descriptor_path,
        X.shape,
        target_path,
    )
    return X, y, descriptor_names


def cross_validate_rf(X, y, descriptor_names, seeds=None):
    """Run cross-validation over a range of seeds and report feature importance.

    ``StandardScaler`` is applied to normalise descriptor magnitudes before
    ``RandomForestRegressor`` training.
    """
    if seeds is None:
        seeds = [1000, 2000, 3000, 4000, 5000]

    for seed in seeds:
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestRegressor(random_state=seed)),
        ])
        scores = cross_val_score(
            model,
            X,
            y,
            cv=kf,
            scoring="neg_mean_squared_error",
        )
        rmse = np.sqrt(-scores.mean())
        logging.info("Seed %d: CV RMSE %.4f", seed, rmse)

    # Fit final model to obtain feature importances
    final_model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(random_state=seeds[0])),
    ])
    final_model.fit(X, y)
    importances = final_model.named_steps["rf"].feature_importances_
    importance_df = pd.DataFrame(
        {
            "descriptor": descriptor_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)
    logging.info(
        "Top feature importances:\n%s",
        importance_df.head(10).to_string(index=False),
    )
    return final_model, importance_df


def main():
    parser = argparse.ArgumentParser(description="RandomForest training")
    parser.add_argument(
        "--descriptors", required=True, help="Path to descriptor array (.npy)"
    )
    parser.add_argument(
        "--targets", required=True, help="Path to target array (.npy)"
    )
    parser.add_argument(
        "--descriptor-names", required=True, help="Descriptor name text file"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    X, y, names = load_dataset(
        args.descriptors, args.targets, args.descriptor_names
    )
    cross_validate_rf(X, y, names)


if __name__ == "__main__":
    main()
