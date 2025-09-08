"""Utilities for computing simple molecular descriptors."""
from typing import Iterable, List, Dict

def compute_descriptors(smiles_list: Iterable[str]) -> List[Dict[str, int]]:
    """Compute simple descriptors for each SMILES string.

    The descriptors are intentionally trivial: the length of the SMILES string
    and the count of carbon atoms ("C" characters).
    """
    descriptors = []
    for smi in smiles_list:
        descriptors.append({
            "length": len(smi),
            "num_C": smi.count("C"),
        })
    return descriptors
