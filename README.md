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

The central contribution is **automation**: under a leakage-free evaluation in which every baseline is granted the same target-ratio search freedom, GA-G-CTGAN matches or exceeds their tuned performance *without any per-dataset ratio search*, within the low-to-moderate imbalance regime (IR ≤ 50).

---

## Repository Structure

```
.
├── 01_load_datasets.ipynb       # Dataset loading and summary statistics
├── 02_baselines.ipynb           # Baseline oversampling (None, SMOTE, ADASYN, G-SMOTE, CTGAN, TVAE); C0 fixed + C1 tuned ratios
├── 02b_ffsmote.ipynb            # FF-SMOTE (Firefly-metaheuristic) baseline
├── 02c_none.ipynb               # No-oversampling reference
├── 03_ctabgan.ipynb             # CTAB-GAN+ experiments (7-dataset subset; RF/MLP only)
├── 04_gctgan.ipynb              # G-CTGAN (fixed-ratio ablation baseline), BIC-optimal GMM clustering
├── 05_ga_gctgan.ipynb           # Proposed GA-G-CTGAN method
├── 06_classification.ipynb      # Consolidated AUC/F1 results and heatmap
├── 07_statistical_tests.ipynb   # Friedman, Wilcoxon, Nemenyi post-hoc; tables and figures
├── common_harness.py            # Shared leakage-free 5-fold CV harness used by all notebooks
├── gmmsampling_adapter.py       # GMMSampling (Naglik & Lango, 2024) wrapper
├── ffsmote_adapter.py           # FF-SMOTE (Kaur & Gosain, 2019) wrapper
├── results/                     # CSV outputs, tables, and figures from each notebook
└── README.md
```

All experiments share `common_harness.py`, which implements a single **leakage-free stratified 5-fold cross-validation** protocol: standardisation is fitted on the training fold only, oversampling is applied to the training fold only, and the held-out fold is left untouched.

---

## Datasets

Eleven publicly available benchmark datasets spanning imbalance ratios (IR) from **1.87 to 577.88**:

| # | Dataset | N | IR | GA-applicable (IR ≤ 50) | Source |
|---|---|---|---|---|---|
| 1 | Pima Diabetes | 768 | 1.87 | Yes | UCI |
| 2 | Credit Default | 30,000 | 3.52 | Yes | UCI |
| 3 | IBM HR Attrition | 1,470 | 5.20 | Yes | Kaggle |
| 4 | Ecoli | 336 | 8.60 | Yes | KEEL |
| 5 | Wine Quality | 4,898 | 25.77 | Yes | UCI |
| 6 | Yeast ME2 | 1,484 | 28.10 | Yes | KEEL |
| 7 | Mammography | 11,183 | 42.01 | Yes | KEEL |
| 8 | Protein Homology | 145,751 | 111.46 | No (fallback) | KDD Cup 2004 |
| 9 | Abalone 19 | 4,177 | 129.53 | No (fallback) | KEEL |
| 10 | PageBlocks | 5,473 | 194.46 | No (fallback) | KEEL |
| 11 | Fraud Detection | 284,807 | 577.88 | No (fallback) | Kaggle (ULB) |

The genetic algorithm optimises the per-cluster ratio on the **seven datasets with IR ≤ 50**. On the four high-IR datasets (IR ≥ 100) the GA surrogate degenerates and the method falls back to fixed-ratio synthesis (see **Limitations**).

---

## Methods Compared

| Method | Type | Reference |
|---|---|---|
| None | Baseline (no oversampling) | — |
| SMOTE | Interpolation | Chawla et al., 2002 |
| ADASYN | Interpolation | He et al., 2008 |
| G-SMOTE | Geometric interpolation | Douzas & Bacao, 2019 |
| FF-SMOTE | Firefly metaheuristic | Kaur & Gosain, 2019 |
| CTGAN | GAN-based | Xu et al., NeurIPS 2019 |
| TVAE | VAE-based | Xu et al., NeurIPS 2019 |
| GMMSampling | GMM difficulty-driven | Naglik & Lango, 2024 |
| G-CTGAN | Cluster-wise GAN (fixed ratio) | — |
| CTAB-GAN+ | GAN-based (advanced) | Zhao et al., IEEE TNNLS 2023 |
| **GA-G-CTGAN** | **Proposed** | **This work** |

Each oversampling method is evaluated under two conditions: **C0** (target minority ratio fixed at 0.50) and **C1** (target ratio tuned per dataset by grid search over {0.20, 0.30, 0.40, 0.50}), so that the baselines are granted the same ratio freedom as the proposed method.

---

## Key Results

On the **seven datasets where GA optimisation is applicable** (IR ≤ 50), mean AUC under the fixed-ratio (C0) setting:

| Classifier | GA-G-CTGAN | Best other method (7-dataset mean) |
|---|---|---|
| Random Forest | 0.8697 | 0.8713 (None) / 0.8701 (G-SMOTE) |
| LightGBM | **0.8755** | 0.8692 (G-CTGAN) |
| MLP | 0.8307 | 0.8429 (GMMSampling) |

Pooled **mean rank** across all eleven datasets and three classifiers (nine oversampling methods; lower is better):

| Rank | Method | Mean rank |
|---|---|---|
| **1** | **GA-G-CTGAN** | **2.95** |
| 2 | G-CTGAN | 4.79 |
| 3 | GMMSampling | 4.94 |
| 4 | ADASYN | 5.00 |
| 5 | G-SMOTE | 5.12 |
| 6 | TVAE | 5.23 |
| 7 | SMOTE | 5.32 |
| 8 | CTGAN | 5.48 |
| 9 | FF-SMOTE | 6.17 |

**Ablation** (GA-G-CTGAN vs. fixed-ratio G-CTGAN, seven GA-applicable datasets): GA-G-CTGAN wins on **6 of 7** datasets under **all three** classifiers, with mean AUC gains of +0.0062 (RF), +0.0064 (LightGBM), +0.0067 (MLP).

**Statistical tests** (nine methods, n = 11 datasets):
- **Friedman test**: RF χ² = 24.41, p = 0.002 (significant); MLP χ² = 30.83, p < 0.001 (significant); LightGBM χ² = 13.70, p = 0.090 (not significant).
- **Nemenyi post-hoc**: at n = 11 the rank advantage of GA-G-CTGAN is **not** statistically significant against the simplest baselines; the only significant pairwise differences involving GA-G-CTGAN are vs. CTGAN (RF) and vs. FF-SMOTE (MLP).
- **Wilcoxon ablation** (GA-G-CTGAN vs. G-CTGAN, n = 7): consistent 6/7 wins but not significant (p ≈ 0.11–0.16).

We report these results with deliberate restraint: the mean-AUC advantage over the simplest *tuned* baselines is modest and not statistically significant at n = 11. The contribution is the **automation** of the per-cluster ratio, which the baselines otherwise require manual per-dataset tuning to match.

> **Note on high-IR datasets:** On the four datasets with IR ≥ 100, the GA surrogate fitness collapses (CV AUC = 0) under extreme class sparsity, and all multipliers saturate at the upper bound. The method then reduces to fixed-ratio G-CTGAN. Downstream test AUC remains competitive there, but the GA contributes nothing beyond maximum-ratio synthesis.

---

## Computational Cost

Mean oversampling time per fold (single CPU core):

| Method | Time (s/fold) |
|---|---|
| SMOTE | 0.04 |
| ADASYN | 0.06 |
| G-SMOTE | 0.90 |
| TVAE | 6.34 |
| GMMSampling | 7.66 |
| CTGAN | 8.89 |
| **GA-G-CTGAN** | **70.63** |
| CTAB-GAN+ | 178.48 |
| FF-SMOTE | 813.24 |

GA-G-CTGAN is substantially more expensive than the interpolation baselines, but far cheaper than the other search-based methods. It is best suited to settings where misclassification cost is high and manual ratio tuning is impractical; for low-stakes or very large-scale settings, a tuned simple baseline is recommended.

---

## Environment

| Component | Specification |
|---|---|
| CPU | Intel Core i5 |
| RAM | 128 GB |
| GPU | NVIDIA RTX 4060Ti |
| Python | 3.10 |
| Key libraries | scikit-learn 1.4, sdv 1.x, lightgbm 4.x, imbalanced-learn 1.x, DEAP 1.4 |

> GMMSampling uses the authors' reference implementation from the `multi-imbalance` package (develop branch); it requires `pydantic<2`. FF-SMOTE follows the Firefly-algorithm description of Kaur & Gosain (2019).

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
jupyter nbconvert --to notebook --execute 02b_ffsmote.ipynb
jupyter nbconvert --to notebook --execute 02c_none.ipynb
jupyter nbconvert --to notebook --execute 03_ctabgan.ipynb
jupyter nbconvert --to notebook --execute 04_gctgan.ipynb
jupyter nbconvert --to notebook --execute 05_ga_gctgan.ipynb
jupyter nbconvert --to notebook --execute 06_classification.ipynb
jupyter nbconvert --to notebook --execute 07_statistical_tests.ipynb
```

Results are saved to `results/` as CSV files (`results/tables/`) and PNG/PDF figures (`results/figures/`).

---

## GA Configuration

| Parameter | Value |
|---|---|
| Population size | 20 |
| Generations | 20 |
| Crossover | SBX (η_c = 15), p = 0.9 |
| Mutation | Polynomial (η_m = 20), p = 0.1 |
| Selection | Tournament (t = 2) |
| Fitness | 5-fold CV AUC (RF surrogate) |
| Minority ratio constraint | [0.20, 0.50] |
| Multiplier bounds | [0.1, 10.0] |
| Random seed | 42 |
| Library | DEAP 1.4 |

---

## Limitations

- **GA fitness degeneracy at high IR:** The RF surrogate collapses for IR ≥ 100 (per-fold minority counts become too small), triggering the fixed-ratio fallback. Extending the automated regime beyond IR ≤ 50 requires a cost-sensitive or weighted-AUC fitness function.
- **Modest, not always significant gains:** On the seven GA-applicable datasets, the advantage over the simplest *tuned* baselines is small and not statistically significant at n = 11. The method is motivated by automation rather than a large accuracy margin.
- **Single-classifier surrogate:** The ratio vector is optimised with an RF surrogate. It transfers to LightGBM and MLP in our experiments (the earlier apparent MLP degradation did not reproduce under the leakage-free protocol), but an ensemble or multi-objective surrogate remains a natural extension. An ensemble surrogate was tested and did not improve MLP performance while roughly tripling cost, so it was not adopted.
- **Computational cost:** Substantially more expensive than simple oversampling (see Computational Cost).
- **CTAB-GAN+ coverage:** Could not be applied to IBM HR Attrition (mixed-type incompatibility) or the high-IR datasets (training timeout), and is incompatible with the LightGBM environment; its comparison is restricted to seven datasets and RF/MLP.

---

## License

This repository is released for **anonymous peer review only**.
Code will be re-released under the MIT License upon acceptance.

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
