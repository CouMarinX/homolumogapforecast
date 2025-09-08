import sys
from pathlib import Path

import numpy as np
from rdkit import Chem

# Make src modules importable
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

from data_preparation import DESCRIPTOR_NAMES, compute_descriptors
from model_training import cross_validate


def test_compute_descriptors_ethanol():
    mol = Chem.MolFromSmiles('CCO')
    desc = compute_descriptors(mol)
    from rdkit.Chem import Descriptors
    expected = np.array([
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
    ])
    assert np.allclose(desc, expected)
    assert list(DESCRIPTOR_NAMES) == ['MolWt', 'MolLogP', 'TPSA', 'NumHDonors', 'NumHAcceptors']


def test_cross_validate_runs():
    smiles = ['CC', 'CCC', 'CCCC', 'CCO']
    y = np.array([0.1, 0.2, 0.3, 0.4])
    X = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        X.append(compute_descriptors(mol))
    X = np.array(X)
    results = cross_validate(X, y, seeds=[1000], n_splits=2)
    assert 1000 in results
    assert len(results[1000]) == 2
