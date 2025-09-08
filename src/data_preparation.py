"""Data preparation utilities for HOMO-LUMO gap prediction."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def load_and_sample(paths: Iterable[Path | str], n_samples: int = 100_000, seed: int = 1000) -> pd.DataFrame:
    """Load CSV files and randomly sample ``n_samples`` rows from each.

    Parameters
    ----------
    paths:
        Iterable of CSV file paths. Each file should contain ``smiles`` and
        ``HOMO-LUMO Gap(Hartree)`` columns.
    n_samples:
        Number of rows to sample from each CSV.
    seed:
        Random seed used for sampling and shuffling.

    Returns
    -------
    pd.DataFrame
        A shuffled DataFrame combining samples from all provided files.
    """
    frames: List[pd.DataFrame] = []
    for path in paths:
        df = pd.read_csv(path)
        if len(df) < n_samples:
            raise ValueError(f"File {path} contains fewer than {n_samples} rows")
        frames.append(df.sample(n=n_samples, random_state=seed))

    combined = (
        pd.concat(frames, ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    return combined


def featurize(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Convert SMILES strings to Morgan fingerprints.

    Invalid SMILES strings are skipped.

    Parameters
    ----------
    df:
        DataFrame containing ``smiles`` and ``HOMO-LUMO Gap(Hartree)`` columns.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, List[str]]
        Feature matrix ``X`` of shape ``(n_samples, 2048)``, target vector ``y``
        and a list of valid SMILES strings.
    """
    features: List[np.ndarray] = []
    gaps: List[float] = []
    valid_smiles: List[str] = []

    for smiles, gap in zip(df["smiles"], df["HOMO-LUMO Gap(Hartree)"]):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        arr = np.zeros((2048,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, arr)
        features.append(arr)
        gaps.append(gap)
        valid_smiles.append(smiles)

    X = np.stack(features)
    y = np.array(gaps, dtype=float)
    return X, y, valid_smiles


def prepare_dataset(paths: Iterable[Path | str], n_samples: int = 100_000, seed: int = 1000) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """High-level convenience function returning features and targets."""
    df = load_and_sample(paths, n_samples=n_samples, seed=seed)
    X, y, smiles = featurize(df)
    return X, y, smiles


__all__ = [
    "load_and_sample",
    "featurize",
    "prepare_dataset",
]
