"""
ff_smote.py
===========

FF-SMOTE (Firefly SMOTE): a metaheuristic oversampling technique that replaces
SMOTE's linear interpolation with the Firefly Algorithm (FA) to synthesise
minority-class samples.

Reference
---------
Original method:
    "FF-SMOTE: A Metaheuristic Approach to Combat Class Imbalance in Binary
    Classification", Applied Artificial Intelligence, 2019.
    DOI: 10.1080/08839514.2019.1577017

Firefly Algorithm:
    Yang, X.-S. (2008). Nature-Inspired Metaheuristic Algorithms.

Design notes (IMPORTANT for reproducibility / reviewers)
--------------------------------------------------------
The original FF-SMOTE paper does NOT specify an explicit objective (brightness)
function for the fireflies; it only states that fireflies generate "optimised
values within the smaller-class region" using lower/upper bounds taken from
randomly selected minority points. Because the Firefly Algorithm requires an
objective function, this implementation adopts a **minority-class cohesion**
brightness function, which is the most direct operationalisation of the paper's
wording ("optimised values in the smaller class region"):

    f(x) = 1 / (1 + mean_{i=1..k} d(x, m_i))

where m_i are the k nearest *minority* neighbours of the candidate x and d is a
heterogeneous distance (HEOM) that supports categorical features. Brighter
(higher-fitness) candidates sit in denser minority regions, steering synthetic
points away from noise/majority territory.

To preserve diversity (a naive global FA collapses all fireflies onto a single
optimum), the algorithm runs an **independent, bounded FA around each minority
seed point** — mirroring SMOTE's "generate near each minority instance" spirit.

Categorical features are handled in the SMOTE-NC style:
    * continuous dimensions are optimised by the Firefly movement equation;
    * categorical dimensions are NOT optimised — each synthetic point copies the
      majority-vote category among the seed's k nearest minority neighbours.

This module has no hard dependency on imbalanced-learn. If imbalanced-learn is
installed, ``FFSMOTE`` subclasses ``BaseOverSampler`` so it plugs into
imblearn Pipelines; otherwise it falls back to a lightweight scikit-learn-style
base with the same ``fit_resample`` API.

Author: generated for research reproduction use.
License: MIT
"""

from __future__ import annotations

import numbers
import warnings
from collections import Counter

import numpy as np

# ---------------------------------------------------------------------------
# Optional imbalanced-learn integration
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised only when imblearn is present
    from imblearn.over_sampling.base import BaseOverSampler

    _HAS_IMBLEARN = True
except Exception:  # pragma: no cover
    _HAS_IMBLEARN = False

    class BaseOverSampler:  # minimal stand-in with the imblearn API surface
        """Fallback base class when imbalanced-learn is not installed."""

        _sampling_type = "over-sampling"

        def fit_resample(self, X, y):
            return self._fit_resample(X, y)

        # imblearn calls fit/fit_resample; keep both for drop-in behaviour
        def fit(self, X, y):
            self.fit_resample(X, y)
            return self


__all__ = ["FFSMOTE"]


# ===========================================================================
# Distance metric: HEOM (Heterogeneous Euclidean-Overlap Metric)
# ===========================================================================
class _HEOM:
    """Heterogeneous Euclidean-Overlap Metric.

    Continuous features contribute a range-normalised squared Euclidean term;
    categorical features contribute an overlap term (0 if equal, 1 otherwise).

    Parameters
    ----------
    cat_mask : np.ndarray of bool, shape (n_features,)
        True for categorical columns.
    ranges : np.ndarray, shape (n_features,)
        Per-feature (max - min) for continuous columns; 1.0 elsewhere.
    """

    def __init__(self, cat_mask, ranges):
        self.cat_mask = cat_mask
        self.cont_mask = ~cat_mask
        # Guard against zero range (constant column) to avoid divide-by-zero.
        safe_ranges = np.where(ranges == 0, 1.0, ranges)
        self.ranges = safe_ranges

    def distance(self, x, Y):
        """Distance from a single point ``x`` to every row of matrix ``Y``.

        Returns an array of shape (len(Y),).
        """
        x = np.asarray(x, dtype=float)
        Y = np.asarray(Y, dtype=float)

        d2 = np.zeros(len(Y), dtype=float)

        if self.cont_mask.any():
            diff = (Y[:, self.cont_mask] - x[self.cont_mask]) / self.ranges[self.cont_mask]
            d2 += np.sum(diff * diff, axis=1)

        if self.cat_mask.any():
            neq = (Y[:, self.cat_mask] != x[self.cat_mask]).astype(float)
            d2 += np.sum(neq, axis=1)

        return np.sqrt(d2)


# ===========================================================================
# FF-SMOTE
# ===========================================================================
class FFSMOTE(BaseOverSampler):
    """Firefly-based SMOTE oversampler.

    Parameters
    ----------
    k_neighbors : int, default=5
        Number of nearest minority neighbours used both for the brightness
        function and for categorical majority voting.

    n_fireflies : int, default=10
        Population size of the Firefly Algorithm run around each seed.

    max_iter : int, default=20
        Number of FA generations per synthetic sample.

    beta0 : float, default=1.0
        Attractiveness at distance 0 (beta_0 in the movement equation).

    gamma : float, default=1.0
        Light-absorption coefficient. Typical range 0.1-10; controls how
        quickly attraction decays with distance.

    alpha : float, default=0.2
        Randomisation parameter (scale of the random-walk term).

    alpha_damp : float, default=0.97
        Multiplicative decay applied to ``alpha`` each generation so the search
        cools down (exploration -> exploitation). Set to 1.0 to disable.

    bound_scale : float, default=1.0
        Width of the per-seed search box. The box is centred between the seed
        and its neighbours; ``bound_scale`` multiplies its half-width. Smaller
        values keep synthetic points tighter around the seed.

    categorical_features : array-like of int or bool, or None, default=None
        Indices (or boolean mask) of categorical columns. If None, all columns
        are treated as continuous. Categorical columns are set by majority vote,
        not optimised.

    sampling_strategy : {'auto', 'minority', float, dict}, default='auto'
        Same semantics as imbalanced-learn. 'auto'/'minority' balance every
        minority class up to the majority count. A float (binary only) sets the
        desired minority:majority ratio. A dict maps class label -> desired
        count.

    random_state : int, RandomState instance or None, default=None
        Controls reproducibility.

    Attributes
    ----------
    sampling_strategy_ : dict
        Resolved number of samples to GENERATE per class.

    Examples
    --------
    >>> from ff_smote import FFSMOTE
    >>> import numpy as np
    >>> X = np.random.rand(100, 4)
    >>> y = np.array([0] * 90 + [1] * 10)
    >>> ff = FFSMOTE(random_state=0)
    >>> X_res, y_res = ff.fit_resample(X, y)
    """

    def __init__(
        self,
        k_neighbors=5,
        n_fireflies=10,
        max_iter=20,
        beta0=1.0,
        gamma=1.0,
        alpha=0.2,
        alpha_damp=0.97,
        bound_scale=1.0,
        categorical_features=None,
        sampling_strategy="auto",
        random_state=None,
    ):
        self.k_neighbors = k_neighbors
        self.n_fireflies = n_fireflies
        self.max_iter = max_iter
        self.beta0 = beta0
        self.gamma = gamma
        self.alpha = alpha
        self.alpha_damp = alpha_damp
        self.bound_scale = bound_scale
        self.categorical_features = categorical_features
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

    # -- helpers ------------------------------------------------------------
    def _check_random_state(self):
        if self.random_state is None or isinstance(self.random_state, numbers.Integral):
            return np.random.RandomState(self.random_state)
        if isinstance(self.random_state, np.random.RandomState):
            return self.random_state
        raise ValueError("random_state must be None, an int, or a RandomState.")

    def _build_cat_mask(self, n_features):
        mask = np.zeros(n_features, dtype=bool)
        if self.categorical_features is None:
            return mask
        cat = np.asarray(self.categorical_features)
        if cat.dtype == bool:
            if cat.shape[0] != n_features:
                raise ValueError(
                    "Boolean categorical_features must have length n_features."
                )
            return cat
        mask[cat.astype(int)] = True
        return mask

    def _resolve_sampling_strategy(self, y):
        counts = Counter(y)
        n_max = max(counts.values())
        strat = self.sampling_strategy

        if strat in ("auto", "minority", "not majority", "all"):
            majority_label = max(counts, key=counts.get)
            to_generate = {}
            for label, cnt in counts.items():
                if strat == "minority":
                    # only the single smallest class
                    if label == min(counts, key=counts.get):
                        to_generate[label] = n_max - cnt
                elif label != majority_label:
                    to_generate[label] = n_max - cnt
            return {k: v for k, v in to_generate.items() if v > 0}

        if isinstance(strat, numbers.Real) and not isinstance(strat, bool):
            if len(counts) != 2:
                raise ValueError("float sampling_strategy is only valid for binary tasks.")
            minority_label = min(counts, key=counts.get)
            majority_label = max(counts, key=counts.get)
            target = int(round(strat * counts[majority_label]))
            gen = max(0, target - counts[minority_label])
            return {minority_label: gen} if gen > 0 else {}

        if isinstance(strat, dict):
            to_generate = {}
            for label, target in strat.items():
                gen = target - counts.get(label, 0)
                if gen < 0:
                    warnings.warn(
                        f"Requested {target} samples for class {label}, but it "
                        f"already has {counts.get(label, 0)}. Skipping."
                    )
                    continue
                if gen > 0:
                    to_generate[label] = gen
            return to_generate

        raise ValueError(f"Unrecognised sampling_strategy: {strat!r}")

    # -- core Firefly routine ----------------------------------------------
    def _generate_one(self, seed, minority_neighbors, cont_mask, cat_mask,
                      heom, rng):
        """Run a bounded Firefly Algorithm around one seed point.

        Returns a single synthetic sample (1D array).
        """
        n_features = seed.shape[0]
        # ---- search bounds from seed + its minority neighbours (paper: lb/ub
        #      from randomly selected minority data) -------------------------
        pool = np.vstack([seed[None, :], minority_neighbors])
        lb = pool[:, cont_mask].min(axis=0)
        ub = pool[:, cont_mask].max(axis=0)
        centre = (lb + ub) / 2.0
        half = (ub - lb) / 2.0 * self.bound_scale
        lb = centre - half
        ub = centre + half
        span = np.where((ub - lb) == 0, 1e-12, ub - lb)

        n_cont = int(cont_mask.sum())

        # ---- categorical part fixed once (SMOTE-NC majority vote) ----------
        cat_values = None
        if cat_mask.any():
            cat_values = np.empty(int(cat_mask.sum()), dtype=minority_neighbors.dtype)
            neigh_cat = minority_neighbors[:, cat_mask]
            for j in range(neigh_cat.shape[1]):
                vals, cnts = np.unique(neigh_cat[:, j], return_counts=True)
                cat_values[j] = vals[np.argmax(cnts)]

        # Degenerate case: no continuous dims -> return seed with voted cats.
        if n_cont == 0:
            out = seed.copy()
            if cat_values is not None:
                out[cat_mask] = cat_values
            return out

        # ---- initialise fireflies (continuous dims only) -------------------
        pos = lb + rng.rand(self.n_fireflies, n_cont) * (ub - lb)

        def full_point(cont_vec):
            """Assemble a full feature vector from a continuous sub-vector."""
            p = seed.copy().astype(float)
            p[cont_mask] = cont_vec
            if cat_values is not None:
                p[cat_mask] = cat_values
            return p

        def brightness(cont_vec):
            # minority-cohesion fitness: closeness to k nearest minority pts
            p = full_point(cont_vec)
            d = heom.distance(p, minority_neighbors)
            kk = min(self.k_neighbors, len(d))
            nearest = np.partition(d, kk - 1)[:kk] if kk < len(d) else d
            return 1.0 / (1.0 + nearest.mean())

        intensity = np.array([brightness(pos[i]) for i in range(self.n_fireflies)])

        alpha_t = self.alpha
        for _ in range(self.max_iter):
            for i in range(self.n_fireflies):
                for j in range(self.n_fireflies):
                    if intensity[j] > intensity[i]:
                        r2 = np.sum((pos[j] - pos[i]) ** 2)
                        beta = self.beta0 * np.exp(-self.gamma * r2)
                        rand_walk = alpha_t * (rng.rand(n_cont) - 0.5) * span
                        pos[i] = pos[i] + beta * (pos[j] - pos[i]) + rand_walk
                        pos[i] = np.clip(pos[i], lb, ub)
                        intensity[i] = brightness(pos[i])
            alpha_t *= self.alpha_damp

        best = int(np.argmax(intensity))
        return full_point(pos[best])

    # -- main API -----------------------------------------------------------
    def _fit_resample(self, X, y):
        rng = self._check_random_state()

        X = np.asarray(X, dtype=object) if self.categorical_features is not None else np.asarray(X, dtype=float)
        # Work in a float view for continuous ops but keep original for categoricals.
        X = np.asarray(X)
        y = np.asarray(y)

        n_features = X.shape[1]
        cat_mask = self._build_cat_mask(n_features)
        cont_mask = ~cat_mask

        # Per-feature ranges (continuous) for HEOM normalisation.
        ranges = np.ones(n_features, dtype=float)
        if cont_mask.any():
            Xc = X[:, cont_mask].astype(float)
            ranges[cont_mask] = Xc.max(axis=0) - Xc.min(axis=0)
        heom = _HEOM(cat_mask, ranges)

        to_generate = self._resolve_sampling_strategy(y)
        self.sampling_strategy_ = dict(to_generate)

        if not to_generate:
            warnings.warn("Nothing to resample; returning data unchanged.")
            return X, y

        new_X_parts = [X]
        new_y_parts = [y]

        for label, n_needed in to_generate.items():
            min_idx = np.where(y == label)[0]
            X_min = X[min_idx]
            n_min = len(X_min)

            if n_min <= 1:
                warnings.warn(
                    f"Class {label} has <=1 sample; cannot synthesise. Skipping."
                )
                continue

            k_eff = min(self.k_neighbors, n_min - 1)

            # Precompute, for each minority point, its k nearest minority peers.
            neigh_idx = np.empty((n_min, k_eff), dtype=int)
            for i in range(n_min):
                d = heom.distance(X_min[i], X_min)
                order = np.argsort(d)
                order = order[order != i][:k_eff]  # drop self
                neigh_idx[i] = order

            synth = np.empty((n_needed, n_features), dtype=X.dtype)
            for s in range(n_needed):
                seed_i = rng.randint(n_min)
                neighbors = X_min[neigh_idx[seed_i]]
                synth[s] = self._generate_one(
                    seed=X_min[seed_i].copy(),
                    minority_neighbors=neighbors,
                    cont_mask=cont_mask,
                    cat_mask=cat_mask,
                    heom=heom,
                    rng=rng,
                )

            new_X_parts.append(synth)
            new_y_parts.append(np.full(n_needed, label, dtype=y.dtype))

        X_res = np.vstack(new_X_parts)
        y_res = np.concatenate(new_y_parts)

        # Shuffle so synthetic rows are not all at the tail.
        perm = rng.permutation(len(y_res))
        return X_res[perm], y_res[perm]

    # Public alias so the class works even without imblearn's dispatcher.
    if not _HAS_IMBLEARN:
        def fit_resample(self, X, y):
            return self._fit_resample(X, y)
