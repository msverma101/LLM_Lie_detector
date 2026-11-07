"""Evaluation primitives: certified low-FPR thresholds, recall at fixed FPR, cluster bootstrap.

All uncertainty is computed by resampling GROUPS (scenarios), never rows.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


class CannotCertify(ValueError):
    """The negative sample is too small to certify the requested FPR at the requested confidence."""


def min_negatives(alpha: float, delta: float = 0.05) -> int:
    """Smallest n for which even the most conservative threshold (max negative) is certified: (1-a)^n <= d."""
    return math.ceil(math.log(delta) / math.log(1.0 - alpha))


def certified_threshold(neg_scores: np.ndarray, alpha: float, delta: float = 0.05) -> float:
    """Threshold tau with P(true FPR(tau) > alpha) <= delta; flag a case when score > tau.

    tau is the m-th largest negative. The true tail mass of that order statistic exceeds alpha
    iff Bin(n, alpha) <= m-1, so we take the largest m with P(Bin(n, alpha) <= m-1) <= delta.
    """
    neg = np.sort(np.asarray(neg_scores, dtype=float))[::-1]
    n = len(neg)
    if n < min_negatives(alpha, delta):
        raise CannotCertify(f"{n} negatives cannot certify FPR {alpha} at delta {delta}; need >= {min_negatives(alpha, delta)}")
    m = 1
    for cand in range(1, n + 1):
        if stats.binom.cdf(cand - 1, n, alpha) <= delta:
            m = cand
        else:
            break
    return float(neg[m - 1])


def recall_at(pos_scores: np.ndarray, threshold: float) -> float:
    return float(np.mean(np.asarray(pos_scores) > threshold))


def fpr_at(neg_scores: np.ndarray, threshold: float) -> float:
    return float(np.mean(np.asarray(neg_scores) > threshold))


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    a = 1 - conf
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    ranks = stats.rankdata(np.concatenate([pos, neg]))
    u = ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def _group_sums(values: np.ndarray, groups: np.ndarray):
    uniq, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=values, minlength=len(uniq))
    cnts = np.bincount(inv, minlength=len(uniq)).astype(float)
    return sums, cnts


def cluster_bootstrap_mean(values, groups, n_boot: int = 10000, seed: int = 0, conf: float = 0.95):
    sums, cnts = _group_sums(np.asarray(values, float), np.asarray(groups))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    means = sums[idx].sum(1) / cnts[idx].sum(1)
    a = (1 - conf) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def cluster_bootstrap_diff(a_vals, b_vals, groups, n_boot: int = 10000, seed: int = 0, conf: float = 0.95):
    """Paired difference mean(a)-mean(b) over the same rows, resampling groups."""
    sa, ca = _group_sums(np.asarray(a_vals, float), np.asarray(groups))
    sb, _ = _group_sums(np.asarray(b_vals, float), np.asarray(groups))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sa), size=(n_boot, len(sa)))
    diffs = (sa[idx].sum(1) - sb[idx].sum(1)) / ca[idx].sum(1)
    a = (1 - conf) / 2
    return float(np.quantile(diffs, a)), float(np.quantile(diffs, 1 - a))
