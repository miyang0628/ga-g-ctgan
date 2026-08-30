# ============================================================================
# GMMSampling adapter  —  GA-G-CTGAN Revision (IDA-26-0590)
# ----------------------------------------------------------------------------
# Wraps Naglik & Lango (2024) GMMSampling so it plugs into the common harness
# interface:  oversample_fn(X_tr, y_tr) -> (X_res, y_res).
#
# Source note (reproducibility):
#   GMMSampling ships only in the `develop` branch of the multi-imbalance repo
#   (Naglik & Lango, Machine Learning 113(8):5183-5202, 2024), NOT in the
#   released PyPI package (0.0.14). The class was written against
#   scikit-learn ~1.1 / imbalanced-learn ~0.9, whose BaseSampler validation
#   layer differs from current versions. We therefore:
#     (1) vendor the exact source file (vendor/gmm_sampler.py), and
#     (2) call the internal `_fit_resample`, bypassing the outer sklearn/
#         imblearn parameter-validation wrapper that is incompatible with
#         the current environment. `_fit_resample` contains the full method
#         logic (GMM subconcept sampling + optional majority cleaning) and is
#         self-contained, so the algorithm itself is unchanged.
#
# IMPORTANT (attribution): the method is by *Naglik & Lango (2024)*, not
# Koziarski. Fix any "Koziarski GMMSampling" citation in the manuscript.
# ============================================================================

import os
import sys
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# Make the vendored source importable.
_VENDOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)

from gmm_sampler import GMMSampler  # vendored develop-branch source


def make_gmmsampling_fn(undersample=True, random_state=42, **kwargs):
    """
    Build an oversample_fn for the harness.

    Parameters
    ----------
    undersample : bool
        True  -> author default: clean overlapping majority instances AND
                 oversample minority subconcepts (the method as published).
        False -> oversample minority only, majority preserved (matches the
                 "minority-only" convention of the other baselines; use for a
                 like-for-like oversampling comparison).
    random_state : int
    **kwargs : forwarded to GMMSampler (e.g. k_neighbors, covariance_type).

    Returns
    -------
    callable  (X_tr, y_tr) -> (X_res, y_res)
    """
    def _fn(X_tr, y_tr):
        X_tr = np.asarray(X_tr, dtype=float)
        y_tr = np.asarray(y_tr, dtype=int)
        sampler = GMMSampler(
            undersample=undersample,
            random_state=random_state,
            **kwargs,
        )
        # Bypass the incompatible outer validation wrapper (see header note).
        X_res, y_res = sampler._fit_resample(X_tr, y_tr)
        return np.asarray(X_res, dtype=float), np.asarray(y_res, dtype=int)

    return _fn


# Convenience presets used by the notebooks.
gmmsampling_fn      = make_gmmsampling_fn(undersample=True)    # "GMMSampling"
gmmsampling_os_fn   = make_gmmsampling_fn(undersample=False)   # "GMMSampling-OS"


if __name__ == "__main__":
    # Self-check across small/large minority cases like our datasets.
    from sklearn.datasets import make_classification
    from collections import Counter

    print("GMMSampling adapter self-check")
    print("=" * 68)
    for n, w, seed, tag in [(200, 0.86, 4, "very small (~28 min)"),
                            (400, 0.87, 2, "yeast-like (~52 min)"),
                            (2000, 0.95, 3, "large (~100 min)")]:
        X, y = make_classification(n_samples=n, n_features=8, n_informative=5,
                                   weights=[w], random_state=seed,
                                   n_clusters_per_class=1)
        for name, fn in [("GMMSampling", gmmsampling_fn),
                         ("GMMSampling-OS", gmmsampling_os_fn)]:
            Xr, yr = fn(X, y)
            print(f"  {tag:<22} {name:<15} "
                  f"{dict(Counter(y))} -> {dict(Counter(yr))}")
    print("OK")
