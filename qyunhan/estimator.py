"""Closed-form shared-latent estimator.

One decoder per cell line,

    x(q, c) = mu_c + B_c z(q, c) + e,        B_c in R^(p x r),

fitted by a rank-r SVD of the centred perturbation-by-gene matrix.  Projection
uses ``B_c^T`` rather than the least-squares solve ``(B_c^T B_c)^-1 B_c^T``:
under assumption (A5) the Gram factor ``B_c^T B_c`` is common across source
lines and is absorbed by the fitted read-out, so the composite is the least
squares estimate up to that factor.
"""
from __future__ import annotations

import numpy as np


def optimal_rank(s, n_obs, n_gene):
    """Gavish-Donoho hard threshold, sigma_opt = omega(beta) * median(s)."""
    s = np.asarray(s, float)
    beta = min(n_obs, n_gene) / max(n_obs, n_gene)
    omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
    return int((s > omega * np.median(s)).sum())


def fit_line(X, r=None):
    """Fit one line's decoder.

    X : (n_perturbations, n_genes) log-fold-change matrix.
    r : latent dimension; chosen by :func:`optimal_rank` when omitted.
    """
    X = np.asarray(X, float)
    mu = X.mean(0)
    _, s, vt = np.linalg.svd(X - mu, full_matrices=False)
    if r is None:
        r = optimal_rank(s, *X.shape)
    return {"mu": mu, "B": np.ascontiguousarray(vt[:r].T), "s": s, "r": int(r)}


def project(model, X):
    """Latent code z = B^T (x - mu)."""
    return (np.atleast_2d(X) - model["mu"]) @ model["B"]


def decode(model, Z):
    """Observation-space prediction x_hat = mu + z B^T."""
    return model["mu"] + np.atleast_2d(Z) @ model["B"].T


def consensus_latents(z_by_source):
    """Mean latent over the source lines, sign-aligned to the first source.

    Decoders are fitted per line, so latent axes carry an arbitrary sign.  Each
    source is oriented against the reference by the sign of its largest-magnitude
    correlation with it.
    """
    ref_key = next(iter(z_by_source))
    ref = np.atleast_2d(z_by_source[ref_key])
    aligned = []
    for k, z in z_by_source.items():
        z = np.atleast_2d(z)
        sign = np.sign(np.sum(ref * z)) or 1.0
        aligned.append(sign * z)
    return np.mean(aligned, 0)
