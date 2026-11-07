import numpy as np
import pytest

from liedet import evalkit as ev


def test_min_negatives_for_extreme_threshold():
    # (1-a)^n <= delta ; computed independently in the concept: 59 for 5 %, 299 for 1 %
    assert ev.min_negatives(0.05, 0.05) == 59
    assert ev.min_negatives(0.01, 0.05) == 299


def test_threshold_refuses_when_sample_cannot_certify():
    with pytest.raises(ev.CannotCertify):
        ev.certified_threshold(np.random.default_rng(0).normal(size=30), alpha=0.05, delta=0.05)


def test_certified_threshold_holds_target_fpr_in_simulation():
    rng = np.random.default_rng(1)
    n, alpha, delta, reps = 400, 0.05, 0.05, 400
    viol = 0
    for _ in range(reps):
        neg = rng.normal(size=n)
        thr = ev.certified_threshold(neg, alpha, delta)
        true_fpr = 1 - 0.5 * (1 + __import__("math").erf(thr / 2**0.5))
        viol += true_fpr > alpha
    assert viol / reps <= delta + 0.04  # certified coverage, small MC slack


def test_recall_and_fpr_at_threshold():
    pos = np.array([0.9, 0.8, 0.1, 0.7]); neg = np.array([0.2, 0.3, 0.95, 0.1])
    assert ev.recall_at(pos, 0.5) == 0.75
    assert ev.fpr_at(neg, 0.5) == 0.25


def test_clopper_pearson_known_value():
    lo, hi = ev.clopper_pearson(0, 59)
    assert lo == 0.0 and abs(hi - (1 - 0.025 ** (1 / 59))) < 1e-9  # two-sided 95 % bound for 0/59


def test_auroc_perfect_and_chance():
    assert ev.auroc(np.array([1, 2, 3]), np.array([0, 0.5, 0.7])) == 1.0
    rng = np.random.default_rng(0)
    assert abs(ev.auroc(rng.normal(size=2000), rng.normal(size=2000)) - 0.5) < 0.05


def test_cluster_bootstrap_resamples_groups_not_rows():
    # all rows of one group share the value; interval must reflect group count, not row count
    groups = np.repeat(np.arange(10), 50)
    vals = np.repeat(np.linspace(0, 1, 10), 50)
    lo, hi = ev.cluster_bootstrap_mean(vals, groups, n_boot=2000, seed=0)
    row_sd = vals.std() / np.sqrt(len(vals))
    assert (hi - lo) / 3.92 > 3 * row_sd


def test_paired_difference_interval_detects_real_gap():
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(40), 5)
    a = rng.normal(0.6, 0.05, 200); b = rng.normal(0.4, 0.05, 200)
    lo, hi = ev.cluster_bootstrap_diff(a, b, g, n_boot=2000, seed=0)
    assert lo > 0
