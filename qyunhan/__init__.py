"""Qyunhan - a program-resolution virtual cell with calibrated abstention.

The space is a closed-form shared-latent estimator; the hold is a relative-error
conformal layer ordered by a hardness head, which decides whether a claim is
delivered, abstained from, or returned as an experiment.
"""
from . import conformal, estimator, hardness, triage
from .conformal import nonconformity, quantile, radii, relative_error
from .estimator import consensus_latents, decode, fit_line, optimal_rank, project
from .hardness import HardnessHead, shape_features
from .triage import ACCEPT, ABSTAIN, LOW_CONFIDENCE, MEASURE, triage as classify

__version__ = "0.1.0"

__all__ = [
    "conformal", "estimator", "hardness", "triage",
    "fit_line", "project", "decode", "optimal_rank", "consensus_latents",
    "relative_error", "nonconformity", "quantile", "radii",
    "HardnessHead", "shape_features",
    "classify", "ACCEPT", "LOW_CONFIDENCE", "ABSTAIN", "MEASURE",
    "__version__",
]
