# Diabetic Retinopathy Severity Grader

An imbalance-focused five-class PyTorch baseline for APTOS 2019 retinal images. It prioritizes quadratic weighted kappa (QWK), macro-F1, and class-wise sensitivity—not accuracy.

## Setup

Python is not installed on this machine yet, so create the requested environment after installing Python 3.10+:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

The supplied archive's nested image folders are detected automatically.

```powershell
python train.py --config config.yaml
python train.py --config config.yaml --ablation
```

`--ablation` writes `outputs/ablation_table.csv`, comparing CE, inverse-frequency weighted CE, focal loss, and class-balanced focal loss with weighted sampling. Each row includes QWK, macro-F1, and recall for all five grades. The best run also writes raw and normalized confusion matrices plus its checkpoint.

## Design notes

- `dataset.py`: black-border crop, Ben Graham local colour normalization, conservative geometric/brightness augmentation, and the inverse-frequency `WeightedRandomSampler`.
- `losses.py`: effective-number class balancing (Cui et al.) and correctly weighted focal loss. The sampler and loss are separately selected in `config.yaml`.
- `evaluate.py`: QWK, macro-F1, per-class recall, raw and row-normalized confusion matrices.
- The provided `train_1.csv` and `valid.csv` are used as the source split. `stratified_split()` is also available in `dataset.py` when working from one labeled manifest.

This is a research/educational tool, not a clinical diagnostic device.
