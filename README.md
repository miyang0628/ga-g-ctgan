# GA-Optimized G-CTGAN: An Automated Oversampling Framework for Imbalanced Data Classification

> **Anonymous submission** — author information withheld for double-blind review.

---

## Overview

This repository contains the code and experimental results accompanying the paper:

> *GA-Optimized G-CTGAN: An Automated Oversampling Framework for Imbalanced Data Classification*
> Submitted to [Target Journal], 2025.

We propose **GA-G-CTGAN**, a framework that combines:
- **GMM clustering** to decompose the minority class into homogeneous subgroups
- **Independent CTGAN** models trained per cluster for high-fidelity synthesis
- **Genetic Algorithm (GA)** optimisation to automatically select per-cluster oversampling ratios using 5-fold cross-validated AUC as the fitness function

---

## Repository Structure

```
.
├── 01_load_datasets.ipynb       # Dataset loading and summary statistics
├── 02_baselines.ipynb           # Baseline oversampling methods (SMOTE, ADASYN, G-SMOTE, CTGAN, TVAE)
├── 03_ctabgan.ipynb             # CTAB-GAN+ experiments (7-dataset subset)
├── 04_gctgan.ipynb              # G-CTGAN experiments with BIC-optimal GMM clustering
├── 05_ga_gctgan.ipynb           # Proposed GA-G-CTGAN method
├── 06_classification.ipynb      # Consolidated AUC/F1 results and heatmap
├── 07_statistical_tests.ipynb   # Friedman, Wilcoxon, Nemenyi post-hoc analysis
├── results/                     # CSV outputs and figures from each notebook
└── README.md
```

---

## Datasets

Eleven publicly available benchmark datasets spanning imbalance ratios (IR) from **1.87 to 577.88**:

| # | Dataset | N | IR | Source |
|---|---|---|---|---|
| 1 | Pima Diabetes | 768 | 1.87 | UCI |
| 2 | Credit Default | 30,000 | 3.52 | UCI |
| 3 | IBM HR Attrition | 1,470 | 5.20 | Kaggle |
| 4 | Ecoli | 336 | 8.60 | KEEL |
| 5 | Wine Quality | 4,898 | 25.77 | UCI |
| 6 | Yeast ME2 | 1,484 | 28.10 | KEEL |
| 7 | Mammography | 11,183 | 42.01 | KEEL |
| 8 | Protein Homology | 145,751 | 111.46 | KDD Cup 2004 |
| 9 | Abalone 19 | 4,177 | 129.53 | KEEL |
| 10 | PageBlocks | 5,473 | 194.46 | KEEL |
| 11 | Fraud Detection | 284,807 | 577.88 | Kaggle (ULB) |

All datasets are loaded via `imbalanced-learn` or public sources. No proprietary data is used.

---

## Methods Compared

| Method | Type | Reference |
|---|---|---|
| None | Baseline (no oversampling) | — |
| SMOTE | Interpolation | Chawla et al., 2002 |
| ADASYN | Interpolation | He et al., 2008 |
| G-SMOTE | Geometric interpolation | Douzas & Bacao, 2019 |
| CTGAN | GAN-based | Xu et al., NeurIPS 2019 |
| TVAE | VAE-based | Xu et al., NeurIPS 2019 |
| G-CTGAN | Cluster-wise GAN | — |
| CTAB-GAN+ | GAN-based (advanced) | Zhao et al., IEEE TNNLS 2023 |
| **GA-G-CTGAN** | **Proposed** | **This work** |

---

## Key Results

On the **6 datasets where GA optimisation succeeds** (IR < 100):

| Classifier | GA-G-CTGAN (mean AUC) | Best baseline |
|---|---|---|
| Random Forest | **0.8745** | 0.8627 (TVAE) |
| LightGBM | **0.8648** | 0.8466 (G-CTGAN) |
| MLP | 0.7861 | **0.8121** (G-CTGAN) |

Statistical tests (RF, n=11 datasets):
- **Friedman test**: χ²=16.32, p=0.012 ✓ significant
- **Wilcoxon vs G-CTGAN**: p=0.032 (RF), p=0.042 (LGBM) ✓ significant

> **Note:** GA optimisation fails (CV AUC=0) on 4 high-IR datasets (IR ≥ 100) due to surrogate collapse under extreme class sparsity. Downstream test AUC on these datasets remains competitive as maximum-ratio synthesis still provides meaningful augmentation.

---

## Environment

| Component | Specification |
|---|---|
| CPU | Intel Core i5 |
| RAM | 128 GB |
| GPU | NVIDIA RTX 4060Ti |
| Python | 3.10 |
| Key libraries | scikit-learn 1.4, sdv 1.x, lightgbm 4.x, imbalanced-learn 1.x |

### Installation

```bash
git clone https://github.com/anonymous/ga-g-ctgan.git
cd ga-g-ctgan
pip install -r requirements.txt
```

### Reproduction

Run notebooks in order:

```bash
jupyter nbconvert --to notebook --execute 01_load_datasets.ipynb
jupyter nbconvert --to notebook --execute 02_baselines.ipynb
jupyter nbconvert --to notebook --execute 03_ctabgan.ipynb
jupyter nbconvert --to notebook --execute 04_gctgan.ipynb
jupyter nbconvert --to notebook --execute 05_ga_gctgan.ipynb
jupyter nbconvert --to notebook --execute 06_classification.ipynb
jupyter nbconvert --to notebook --execute 07_statistical_tests.ipynb
```

Results are saved to `results/` as CSV files and PNG figures.

---

## GA Configuration

| Parameter | Value |
|---|---|
| Population size | 20 |
| Generations | 20 |
| Crossover | SBX (η_c = 15), p = 0.9 |
| Mutation | Polynomial (η_m = 20), p = 0.1 |
| Fitness | 5-fold CV AUC (RF surrogate) |
| Minority ratio constraint | [0.20, 0.50] |
| Random seed | 42 |

---

## Limitations

- GA optimisation is not reliable for datasets with IR ≥ 100 under the current RF surrogate design.
- The RF-optimised ratio vector does not generalise well to MLP (consistent negative delta across all 6 applicable datasets).
- CTAB-GAN+ could not be applied to IBM HR Attrition (mixed-type incompatibility) or high-IR datasets (training timeout).

---

## License

This repository is released for **anonymous peer review only**.
Code will be re-released under MIT License upon acceptance.

---

## Citation

```bibtex
@article{anonymous2025gagctgan,
  title   = {GA-Optimized G-CTGAN: An Automated Oversampling Framework
             for Imbalanced Data Classification},
  author  = {Anonymous},
  journal = {Under review},
  year    = {2025}
}
```
