# homolumogapforecast

This project demonstrates how artificial intelligence can be applied to a simple
chemistry task: predicting the HOMO–LUMO gap of a molecule from its SMILES
string.  It is intended as an educational example for traditional chemists who
would like to learn basic AI workflows.

## Workflow

1. **Sampling and preprocessing**
   - Four CSV files are required: cations, anions, neutral molecules and
     radicals.  Each file must contain two columns: `smiles` and
     `HOMO-LUMO Gap(Hartree)`.
   - Using seed 1000, 100000 molecules are sampled from each dataset.  The four
     samples are concatenated, shuffled and saved as `combined_dataset.csv`.
   - RDKit validates the SMILES strings and computes a small set of molecular
     descriptors (`MolWt`, `MolLogP`, `TPSA`, `NumHDonors`, `NumHAcceptors`).
2. **Model training**
   - A Random Forest regressor is trained to predict the HOMO–LUMO gap.
   - Five-fold cross‑validation is performed with seeds 1000, 2000, 3000,
     4000 and 5000.
3. **Descriptor evaluation**
   - After training, feature importances highlight which descriptors contribute
     most to the prediction.

## Usage

```bash
pip install -r requirements.txt
python -m src.main cation.csv anion.csv neutral.csv radical.csv
```

The script prints cross‑validation metrics for each seed and the importance of
each descriptor.
