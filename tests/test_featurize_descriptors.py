import os
import sys
import numpy as np
import pandas as pd
from rdkit.Chem import Descriptors

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_preparation import featurize

def test_featurize_returns_rdkit_descriptors():
    df = pd.DataFrame({
        'smiles': ['CC', 'O'],
        'HOMO-LUMO Gap(Hartree)': [0.1, 0.2],
    })
    X, y, smiles = featurize(df)
    excluded = {
        "BalabanJ",
        "BertzCT",
        "Chi0",
        "Chi0n",
        "Chi0v",
        "Chi1",
        "Chi1n",
        "Chi1v",
        "Chi2n",
        "Chi2v",
        "Chi3n",
        "Chi3v",
        "Chi4n",
        "Chi4v",
        "HallKierAlpha",
        "Ipc",
        "Kappa1",
        "Kappa2",
        "Kappa3",
    }
    n_desc = len([d for d in Descriptors._descList if "EState" not in d[0] and d[0] not in excluded])
    assert X.shape == (2, n_desc)
    assert np.allclose(y, [0.1, 0.2])
    assert smiles == ['CC', 'O']
