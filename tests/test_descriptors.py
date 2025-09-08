import csv
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from hlgap.descriptors import compute_descriptors

def test_compute_descriptors_matches_dataset():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic.csv')
    with open(data_path, newline='') as f:
        rows = list(csv.DictReader(f))
    smiles = [r['smiles'] for r in rows]
    expected = [
        {'length': int(r['length']), 'num_C': int(r['num_C'])}
        for r in rows
    ]
    result = compute_descriptors(smiles)
    assert result == expected
