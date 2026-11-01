"""Split conformal on relative error, normalised by a label-free hardness proxy.

The nonconformity score is

    e = ||y - p|| / ||y|| / s,

where ``s`` is known at test time (a predicted-magnitude proxy, a cross-source
disagreement proxy, or the fitted hardness head).  The radius reported for a row
is ``q_hat * s``, with ``q_hat`` the finite-sample conformal quantile of the
calibration scores.  This is the construction behind Table 3: the *ordering*
supplied by ``s`` is what buys selection, not the marginal validity, which holds
for any ``s`` that is a function of the features alone.
"""
from __future__ import annotations

import numpy as np


def relative_error(y, p):
    """Per-row ||y - p|| / ||y||; the denominator is floored to avoid 0/0."""
    y = np.atleast_2d(np.asarray(y, float))
    p = np.atleast_2d(np.asarray(p, float))
    return np.linalg.norm(y - p, axis=1) / (np.linalg.norm(y, axis=1) + 1e-12)


def nonconformity(y, p, s=1.0):
    """e = relative error / hardness proxy."""
    return relative_error(y, p) / np.asarray(s, float)


def quantile(scores, alpha):
    """Conformal quantile: the ceil((n + 1)(1 - alpha))-th smallest score."""
    scores = np.sort(np.asarray(scores, float))
    n = scores.size
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    return np.inf if k > n else float(scores[k - 1])


def radii(cal_scores, s_test, alpha):
    """Per-row radius q_hat * s for the requested marginal level."""
    return quantile(cal_scores, alpha) * np.asarray(s_test, float)


def magnitude_proxy(P):
    """1 / ||p||: a row with a small predicted perturbation is the harder one."""
    P = np.atleast_2d(np.asarray(P, float))
    return 1.0 / (np.linalg.norm(P, axis=1) + 1e-12)


def agreement_proxy(P_by_source):
    """Mean pairwise correlation deficit across source profiles.

    Returns 1 - mean pairwise Pearson r, so that sources that disagree are
    ranked as harder.  Values are clipped to [0, 2].
    """
    mats = [np.atleast_2d(np.asarray(M, float)) for M in P_by_source]
    k = len(mats)
    if k < 2:
        return np.zeros(mats[0].shape[0])
    acc = np.zeros(mats[0].shape[0])
    n = 0
    for i in range(k):
        for j in range(i + 1, k):
            a, b = mats[i], mats[j]
            a = (a - a.mean(1, keepdims=True))
            b = (b - b.mean(1, keepdims=True))
            denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
            acc += (a * b).sum(1) / denom
            n += 1
    return np.clip(1.0 - acc / n, 0.0, 2.0)


def selective_curve(metric, hardness, coverages):
    """Mean of ``metric`` among the rows accepted at each coverage.

    ``hardness`` is ordered ascending: the least hard rows are accepted first.
    """
    metric = np.asarray(metric, float)
    order = np.argsort(np.asarray(hardness, float))
    n = metric.size
    out = []
    for c in coverages:
        k = max(1, int(round(c * n)))
        out.append(float(metric[order[:k]].mean()))
    return np.asarray(out)


def risk_coverage_auc(metric, hardness, coverages):
    """Average selective score over the coverage grid (Table 3's AUC column)."""
    return float(selective_curve(metric, hardness, coverages).mean())
