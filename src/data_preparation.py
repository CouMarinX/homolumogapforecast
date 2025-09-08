"""Utility functions for preparing datasets using RDKit descriptors."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

# Names of descriptors computed by ``compute_descriptors``.  These are exposed so
# that downstream code can easily determine the dimensionality of the feature
# vectors produced by :func:`featurize_smiles`.
DESCRIPTOR_NAMES = [
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
]


def compute_descriptors(mol) -> np.ndarray:
    """Return a vector of simple RDKit descriptors for ``mol``.

    Parameters
    ----------
    mol
        RDKit molecule object from which descriptors are calculated.

    Returns
    -------
    numpy.ndarray
        Array containing ``MolWt``, ``MolLogP``, ``TPSA``, ``NumHDonors`` and
        ``NumHAcceptors`` in that order.
    """
    from rdkit.Chem import Descriptors  # Imported lazily to avoid hard dependency

    descriptor_funcs = [
        Descriptors.MolWt,
        Descriptors.MolLogP,
        Descriptors.TPSA,
        Descriptors.NumHDonors,
        Descriptors.NumHAcceptors,
    ]

    return np.array([func(mol) for func in descriptor_funcs], dtype=float)


def featurize_smiles(smiles_list: Sequence[str]) -> np.ndarray:
    """Convert an iterable of SMILES strings into descriptor vectors."""
    from rdkit import Chem  # Imported lazily to avoid hard dependency

    features = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {s}")
        features.append(compute_descriptors(mol))

    if not features:
        return np.empty((0, len(DESCRIPTOR_NAMES)))

    return np.vstack(features)


def prepare_dataset(
    csv_path: str,
    output_path: str,
    smiles_column: str = "smiles",
    target_column: str = "target",
) -> pd.DataFrame:
    """Read a CSV file and append descriptor features.

    The input CSV is expected to contain a column with SMILES strings.  The
    descriptors computed from those strings are appended as new columns and the
    resulting DataFrame is written to ``output_path``.

    Parameters
    ----------
    csv_path
        Path to the input CSV file containing at least ``smiles_column`` and
        ``target_column``.
    output_path
        Where the combined dataset will be written.
    smiles_column
        Name of the column containing SMILES strings.
    target_column
        Name of the column containing the target property.  This column is
        preserved to make the combined dataset immediately usable for model
        training.

    Returns
    -------
    pandas.DataFrame
        The combined dataset with descriptor columns appended.
    """
    df = pd.read_csv(csv_path)

    descriptor_array = featurize_smiles(df[smiles_column].tolist())
    descriptor_df = pd.DataFrame(descriptor_array, columns=DESCRIPTOR_NAMES)

    combined = pd.concat(
        [df.drop(columns=[smiles_column]).reset_index(drop=True), descriptor_df],
        axis=1,
    )
    combined.to_csv(output_path, index=False)
    return combined
