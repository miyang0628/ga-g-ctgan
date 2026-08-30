# ============================================================================
# FF-SMOTE adapter  —  GA-G-CTGAN Revision (IDA-26-0590)
# ----------------------------------------------------------------------------
# Wraps the FFSMOTE implementation (ff_smote.py) so it plugs into the common
# harness interface:  oversample_fn(X_tr, y_tr) -> (X_res, y_res).
#
# FF-SMOTE (Kaur & Gosain, Applied Artificial Intelligence, 2019) is a
# firefly-metaheuristic oversampler, included at reviewer R1's suggestion as a
# metaheuristic point of comparison against the GA.
#
# target_ratio convention:
#   The harness/baselines use target_ratio = minority / (minority + majority)
#   (fraction of the whole). FFSMOTE's float sampling_strategy is
#   minority / majority. Conversion:  strat = target_ratio / (1 - target_ratio).
#   e.g. target_ratio 0.50 -> strat 1.0 (balanced).
#
# COST WARNING:
#   FF-SMOTE runs an independent Firefly Algorithm (n_fireflies x max_iter
#   evaluations) PER synthetic sample, so wall-clock scales with the number of
#   samples to generate. It is cheap on small-minority datasets but can take
#   many minutes per fold on large ones (e.g. credit_default). Reduce
#   n_fireflies / max_iter, or restrict FF-SMOTE to small/medium datasets, if
#   runtime is a concern.
# ============================================================================

import numpy as np
from ff_smote import FFSMOTE


def make_ffsmote_fn(target_ratio=0.50, k_neighbors=5,
                    n_fireflies=10, max_iter=20, random_state=42):
    """
    Build an oversample_fn for the harness.

    Parameters
    ----------
    target_ratio : float
        Desired minority fraction of the whole (as elsewhere in this project).
    k_neighbors : int
        Minority neighbours for the brightness function and categorical voting.
    n_fireflies, max_iter : int
        Firefly Algorithm budget per synthetic sample (drives runtime).
    random_state : int

    Returns
    -------
    callable  (X_tr, y_tr) -> (X_res, y_res)
    """
    strat = target_ratio / (1 - target_ratio)

    def _fn(X_tr, y_tr):
        X_tr = np.asarray(X_tr, dtype=float)
        y_tr = np.asarray(y_tr, dtype=int)
        n_min = int((y_tr == 1).sum())
        if n_min < 2:
            return X_tr, y_tr
        ff = FFSMOTE(
            sampling_strategy=strat,
            k_neighbors=min(k_neighbors, n_min - 1),
            n_fireflies=n_fireflies,
            max_iter=max_iter,
            random_state=random_state,
        )
        X_res, y_res = ff.fit_resample(X_tr, y_tr)
        return np.asarray(X_res, dtype=float), np.asarray(y_res, dtype=int)

    return _fn


# Convenience preset used by the notebooks (fixed 0.50, matches baselines C0).
ffsmote_fn = make_ffsmote_fn(target_ratio=0.50)


if __name__ == "__main__":
    import warnings, time
    warnings.filterwarnings("ignore")
    from sklearn.datasets import make_classification
    from collections import Counter

    print("FF-SMOTE adapter self-check")
    print("=" * 60)
    for n, w, tag in [(400, 0.88, "small"), (1500, 0.93, "medium")]:
        X, y = make_classification(n_samples=n, n_features=8, n_informative=5,
                                   weights=[w], random_state=1,
                                   n_clusters_per_class=1)
        t = time.time()
        Xr, yr = ffsmote_fn(X, y)
        print(f"  {tag:<8} {dict(Counter(y))} -> {dict(Counter(yr))} "
              f"({time.time()-t:.1f}s)")
    print("OK")
