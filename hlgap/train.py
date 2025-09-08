"""Simple training routine using linear regression without external deps."""
import csv
from typing import List
from .descriptors import compute_descriptors

def _det3(m):
    return (m[0][0]*(m[1][1]*m[2][2] - m[1][2]*m[2][1])
            - m[0][1]*(m[1][0]*m[2][2] - m[1][2]*m[2][0])
            + m[0][2]*(m[1][0]*m[2][1] - m[1][1]*m[2][0]))

def _linear_regression(x1: List[float], x2: List[float], y: List[float]):
    n = len(y)
    Sx1 = sum(x1)
    Sx2 = sum(x2)
    Sy = sum(y)
    Sx1x1 = sum(a*a for a in x1)
    Sx2x2 = sum(a*a for a in x2)
    Sx1x2 = sum(a*b for a, b in zip(x1, x2))
    Sx1y = sum(a*b for a, b in zip(x1, y))
    Sx2y = sum(a*b for a, b in zip(x2, y))

    M = [[n, Sx1, Sx2],
         [Sx1, Sx1x1, Sx1x2],
         [Sx2, Sx1x2, Sx2x2]]
    R = [Sy, Sx1y, Sx2y]

    M0 = [[R[0], Sx1, Sx2],
          [R[1], Sx1x1, Sx1x2],
          [R[2], Sx1x2, Sx2x2]]
    M1 = [[n, R[0], Sx2],
          [Sx1, R[1], Sx1x2],
          [Sx2, R[2], Sx2x2]]
    M2 = [[n, Sx1, R[0]],
          [Sx1, Sx1x1, R[1]],
          [Sx2, Sx1x2, R[2]]]

    D = _det3(M)
    if D == 0:
        raise ValueError("Design matrix is singular")
    w0 = _det3(M0) / D
    w1 = _det3(M1) / D
    w2 = _det3(M2) / D
    return w0, w1, w2


def train(data_path: str) -> List[float]:
    """Train a linear model on the given dataset and return predictions."""
    with open(data_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    smiles = [row["smiles"] for row in rows]
    y = [float(row["label"]) for row in rows]
    desc = compute_descriptors(smiles)
    x1 = [d["length"] for d in desc]
    x2 = [d["num_C"] for d in desc]
    w0, w1, w2 = _linear_regression(x1, x2, y)
    preds = [w0 + w1*a + w2*b for a, b in zip(x1, x2)]
    return preds
