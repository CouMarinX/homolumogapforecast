import csv
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from hlgap.train import train

def test_training_produces_correct_shape():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic.csv')
    preds = train(data_path)
    with open(data_path, newline='') as f:
        rows = list(csv.DictReader(f))
    labels = [float(r['label']) for r in rows]
    assert len(preds) == len(labels)
    for p, y in zip(preds, labels):
        assert abs(p - y) < 1e-6
