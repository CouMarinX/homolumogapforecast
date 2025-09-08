# homolumogapforecast

A project for teaching basic AI knowledge for chemists.

## Feature representation

Rather than representing molecules with 2048-bit Morgan fingerprints, this project now uses a small set of interpretable RDKit descriptors:

- **MolWt** – molecular weight
- **MolLogP** – octanol/water partition coefficient
- **TPSA** – topological polar surface area
- **NumHDonors** – hydrogen bond donors
- **NumHAcceptors** – hydrogen bond acceptors

These descriptors convey physicochemical properties that chemists routinely use, allowing model predictions to be rationalized in terms of familiar concepts rather than opaque bit vectors.

## Example

```python
from rdkit import Chem
from rdkit.Chem import Descriptors

mol = Chem.MolFromSmiles("CCO")
features = [
    Descriptors.MolWt(mol),
    Descriptors.MolLogP(mol),
    Descriptors.TPSA(mol),
    Descriptors.NumHDonors(mol),
    Descriptors.NumHAcceptors(mol),
]
```

Replace any previous steps generating 2048-bit fingerprints with code like the above to compute the descriptors before training or evaluating models.
