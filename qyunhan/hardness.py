"""Hardness head: ridge regression on the shape of the predicted programme profile.

Every feature is a function of the prediction alone, so the head is usable on a
target line in which the perturbation has not been measured.  The target is the
true relative error of the same row, which is a property of the perturbation and
its source geometry rather than of the line -- the reason a head fitted on three
lines transfers to the fourth (Table 6, bottom).
"""
from __future__ import annotations

import numpy as np

FEATURES = (
    "peak",
    "margin12",
    "margin23",
    "top3_mean",
    "mean_abs",
    "sd_abs",
    "norm",
    "profile_head",
)


def shape_features(P, profile_head=None):
    """Seven shape features of an (n x K) programme profile, plus an optional head value."""
    P = np.atleast_2d(np.asarray(P, float))
    s = -np.sort(-P, axis=1)
    cols = [
        s[:, 0],
        s[:, 0] - s[:, 1],
        s[:, 1] - s[:, 2],
        s[:, :3].mean(1),
        np.abs(P).mean(1),
        np.abs(P).std(1),
        np.linalg.norm(P, axis=1),
    ]
    F = np.stack(cols, 1)
    if profile_head is not None:
        F = np.column_stack([F, np.asarray(profile_head, float)])
    return F


class HardnessHead:
    """Standardise, then ridge with a closed-form solution."""

    def __init__(self, lam=1.0):
        self.lam = lam

    def fit(self, F, y):
        F = np.asarray(F, float)
        y = np.asarray(y, float)
        self.mu_ = F.mean(0)
        self.sd_ = F.std(0) + 1e-12
        self.ym_ = y.mean()
        A = (F - self.mu_) / self.sd_
        A = np.column_stack([np.ones(len(A)), A])
        pen = self.lam * np.eye(A.shape[1])
        pen[0, 0] = 0.0
        self.w_ = np.linalg.solve(A.T @ A + pen, A.T @ (y - self.ym_))
        return self

    def predict(self, F):
        F = np.atleast_2d(np.asarray(F, float))
        A = (F - self.mu_) / self.sd_
        A = np.column_stack([np.ones(len(A)), A])
        return self.ym_ + A @ self.w_
