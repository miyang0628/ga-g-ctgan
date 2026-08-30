# ============================================================================
# COMMON HARNESS  —  GA-G-CTGAN Revision (IDA-26-0590)
# ----------------------------------------------------------------------------
# Single shared foundation imported by every experiment notebook (02/03/04/05).
# Reflects all decisions confirmed during the revision:
#   - 5-fold stratified CV (replaces the single hold-out split)  -> R3 protocol
#   - Per-fold StandardScaler.fit on the TRAIN fold only         -> no leakage
#   - Oversampling applied to the TRAIN fold only                -> no leakage
#   - Adaptive fold count (handles ecoli=35, yeast_me2=51)
#   - Per-fold timing logs (for cost analysis, reviewer C)
#   - Per-dataset mean +/- std aggregation
#   - GA_APPLICABLE = IR <= 50 (7 datasets, single rule; includes mammography)
#
# Usage in a notebook (same directory):
#     from common_harness import *
# ============================================================================

import os
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# -- Paths -------------------------------------------------------------------
DATASET_DIR = "./datasets"
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# -- Reproducibility ---------------------------------------------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# -- Experiment scope --------------------------------------------------------
# (min_n, IR) as verified in 01_load_datasets.
DATASET_META = {
    "pima_diabetes":   {"min_n": 268,  "IR": 1.87},
    "credit_default":  {"min_n": 6636, "IR": 3.52},
    "ibm_attrition":   {"min_n": 237,  "IR": 5.20},
    "ecoli":           {"min_n": 35,   "IR": 8.60},
    "wine_quality":    {"min_n": 183,  "IR": 25.77},
    "yeast_me2":       {"min_n": 51,   "IR": 28.10},
    "mammography":     {"min_n": 260,  "IR": 42.01},
    "protein_homo":    {"min_n": 1296, "IR": 111.46},
    "abalone_19":      {"min_n": 32,   "IR": 129.53},
    "pageblocks":      {"min_n": 28,   "IR": 194.46},
    "fraud_detection": {"min_n": 492,  "IR": 577.88},
}

# Paper table order (ascending IR).
ALL_DATASETS = sorted(DATASET_META, key=lambda d: DATASET_META[d]["IR"])

CLASSIFIERS = ["RF", "LGBM", "MLP"]

METHOD_ORDER = [
    "None", "SMOTE", "ADASYN", "G-SMOTE",
    "CTGAN", "TVAE", "CTAB-GAN+", "GMMSampling", "FF-SMOTE",
    "G-CTGAN", "GA-G-CTGAN",
]

# -- GA-applicability rule (CONFIRMED: single rule IR <= 50) ------------------
# In the raw results (05_ga_gctgan_results.csv), the datasets where the GA
# actually optimised (best_cv_auc > 0) are exactly the 7 with IR <= 50
# (mammography included). The code comment "min_n >= 100" was incorrect.
GA_IR_THRESHOLD = 50.0

def is_ga_applicable(dataset_name):
    """Return True if the GA optimisation is valid for this dataset (IR <= 50)."""
    return DATASET_META[dataset_name]["IR"] <= GA_IR_THRESHOLD

GA_APPLICABLE = [d for d in ALL_DATASETS if is_ga_applicable(d)]
# -> ['pima_diabetes','credit_default','ibm_attrition','ecoli',
#    'wine_quality','yeast_me2','mammography']  (7 datasets)

# CTAB-GAN+ applicable set (kept from existing code; verify separately if needed).
CTABGAN_APPLICABLE = [
    "credit_default", "pima_diabetes", "yeast_me2",
    "mammography", "wine_quality", "ecoli", "pageblocks",
]

# -- CV configuration --------------------------------------------------------
N_SPLITS_DESIRED = 5
MIN_MINORITY_PER_FOLD = 8   # ecoli(35)->4-fold; most datasets stay at 5-fold


def safe_n_splits(y, desired=N_SPLITS_DESIRED, min_per_fold=MIN_MINORITY_PER_FOLD):
    """
    Adaptively lower the fold count so that each fold holds at least
    `min_per_fold` minority samples in datasets with a small minority class.
    Statistical tests use one mean-AUC observation per dataset, so an uneven
    fold count across datasets does not invalidate them.
    """
    n_min = int((np.asarray(y) == 1).sum())
    return int(min(desired, max(2, n_min // min_per_fold)))


# -- Classifiers (unified across 02/05) --------------------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier


def get_classifier(name, random_state=RANDOM_STATE):
    """Shared classifier definitions. Identical HPs across all oversampling
    methods to isolate the effect of data augmentation (no unfair advantage)."""
    if name == "RF":
        return RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=3,
            class_weight="balanced", random_state=random_state, n_jobs=-1,
        )
    if name == "LGBM":
        return LGBMClassifier(
            n_estimators=100, learning_rate=0.05, num_leaves=31,
            class_weight="balanced", random_state=random_state,
            n_jobs=-1, verbose=-1,
        )
    if name == "MLP":
        return MLPClassifier(
            hidden_layer_sizes=(128, 64), alpha=0.001,
            max_iter=300, random_state=random_state,
        )
    raise ValueError(f"Unknown classifier: {name}")


# -- Metrics (single definition) ---------------------------------------------
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    average_precision_score,
)


def evaluate(model, X_test, y_test):
    """AUC (primary), PR-AUC, F1, Precision, Recall.
    Rounding is deferred to the aggregation step to avoid precision loss."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    return {
        "AUC"      : roc_auc_score(y_test, y_prob),
        "PR_AUC"   : average_precision_score(y_test, y_prob),
        "F1"       : f1_score(y_test, y_pred, zero_division=0),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall"   : recall_score(y_test, y_pred, zero_division=0),
    }


# -- Core harness: evaluate one oversampler on one dataset via 5-fold CV ------
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def evaluate_oversampler_cv(
    X, y,
    oversample_fn,
    method_name,
    dataset_name,
    clf_names=CLASSIFIERS,
    seed=RANDOM_STATE,
    return_fold_rows=True,
):
    """
    Evaluate one oversampling method on one dataset using stratified k-fold CV.

    Parameters
    ----------
    X, y : np.ndarray
        Raw (pre-scaling) full data.
    oversample_fn : callable | None
        Signature (X_tr, y_tr) -> (X_res, y_res).
        None means no oversampling (baseline 'None').
        A 3-tuple (X_res, y_res, info) is also accepted when the method needs
        to report extra information (e.g. GA best_k, ga_time).
    method_name, dataset_name : str
        Labels used for logging and aggregation.
    clf_names : list[str]
        Classifiers to evaluate.
    seed : int
    return_fold_rows : bool
        If True, return per-fold rows (for aggregation and statistical tests).

    Returns
    -------
    pd.DataFrame
        One row per (fold, classifier). Columns:
        dataset, oversampling, classifier, fold, n_splits,
        AUC, PR_AUC, F1, Precision, Recall,
        oversample_time, train_time, n_resampled, minority_ratio, info(optional)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    n_splits = safe_n_splits(y, desired=N_SPLITS_DESIRED)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    rows = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # 1) Fit the scaler on the TRAIN fold only (prevents leakage).
        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # 2) Oversample the TRAIN fold only.
        t0 = time.time()
        info = None
        if oversample_fn is None:
            X_res, y_res = X_tr_s, y_tr
        else:
            out = oversample_fn(X_tr_s.copy(), y_tr.copy())
            if len(out) == 3:
                X_res, y_res, info = out
            else:
                X_res, y_res = out
        oversample_time = time.time() - t0

        n_resampled = len(y_res)
        minority_ratio = float((np.asarray(y_res) == 1).sum() / n_resampled)

        # 3) Train and evaluate each classifier on the same resampled fold.
        for clf_name in clf_names:
            t1 = time.time()
            try:
                clf = get_classifier(clf_name, random_state=seed).fit(X_res, y_res)
                train_time = time.time() - t1
                m = evaluate(clf, X_te_s, y_te)
            except Exception as e:
                print(f"    [{dataset_name}|{method_name}|{clf_name}] "
                      f"fold{fold} FAILED: {e}")
                continue

            row = {
                "dataset"        : dataset_name,
                "oversampling"   : method_name,
                "classifier"     : clf_name,
                "fold"           : fold,
                "n_splits"       : n_splits,
                "oversample_time": oversample_time,
                "train_time"     : train_time,
                "n_resampled"    : n_resampled,
                "minority_ratio" : minority_ratio,
                **m,
            }
            if info is not None:
                row["info"] = info
            rows.append(row)

    return pd.DataFrame(rows) if return_fold_rows else rows


# -- Dataset loader ----------------------------------------------------------
def load_dataset(dataset_name):
    """Load datasets/<name>.csv -> (X, y). Returns (None, None) if missing."""
    path = os.path.join(DATASET_DIR, f"{dataset_name}.csv")
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path)
    X = df.drop(columns=["target"]).values.astype(float)
    y = df["target"].values.astype(int)
    return X, y


# -- Per-fold DataFrame -> per (dataset, method, classifier) summary ----------
def summarize_folds(fold_df):
    """
    Aggregate per-fold results to the (dataset, oversampling, classifier) level.
    Returns AUC_mean / AUC_std and summed/averaged timing.
    This summary feeds the paper tables and the statistical tests
    (one observation per dataset).
    """
    agg = (fold_df
           .groupby(["dataset", "oversampling", "classifier"])
           .agg(AUC_mean=("AUC", "mean"),
                AUC_std=("AUC", "std"),
                PRAUC_mean=("PR_AUC", "mean"),
                F1_mean=("F1", "mean"),
                oversample_time=("oversample_time", "mean"),
                train_time=("train_time", "mean"),
                n_splits=("n_splits", "first"),
                minority_ratio=("minority_ratio", "mean"))
           .reset_index())
    return agg


# -- Self-check --------------------------------------------------------------
def _self_check():
    print("=" * 70)
    print("COMMON HARNESS READY")
    print("=" * 70)
    print(f"  Datasets (IR order) : {len(ALL_DATASETS)}")
    print(f"  Classifiers         : {CLASSIFIERS}")
    print(f"  GA rule             : IR <= {GA_IR_THRESHOLD}")
    print(f"  GA_APPLICABLE ({len(GA_APPLICABLE)})  : {GA_APPLICABLE}")
    print(f"  GA fallback ({len(ALL_DATASETS)-len(GA_APPLICABLE)})    : "
          f"{[d for d in ALL_DATASETS if d not in GA_APPLICABLE]}")
    print()
    print("  Adaptive fold counts (min_per_fold=8):")
    for d in ALL_DATASETS:
        mn = DATASET_META[d]["min_n"]
        ns = min(N_SPLITS_DESIRED, max(2, mn // MIN_MINORITY_PER_FOLD))
        flag = "  (reduced)" if ns < N_SPLITS_DESIRED else ""
        print(f"    {d:<17} min_n={mn:>5}  -> {ns}-fold"
              f"  (test~{mn/ns:.0f}/fold){flag}")


if __name__ == "__main__":
    _self_check()
