"""Three-state triage of a target, plus the measurement state for untestable ones.

The four outcomes are ordered by what the instrument is allowed to say:

    accept          two or more source lines agree; the top-three profile is
                    delivered as a hypothesis.
    low-confidence  the profile is delivered, flagged; source disagreement is
                    too large for it to be ranked against other targets.
    abstain         a single source line measured the perturbation; the profile
                    is not delivered.
    measure         no source line measured it; the correct output is an
                    experiment, not a number.
"""
from __future__ import annotations

import numpy as np

ACCEPT = "accept"
LOW_CONFIDENCE = "low-confidence"
ABSTAIN = "abstain"
MEASURE = "measure"

DEFAULT_MIN_AGREEMENT = 0.30


def triage(n_sources, agreement=None, radius=None,
           tau=0.5, min_agreement=DEFAULT_MIN_AGREEMENT):
    """Decide the state of one target.

    n_sources  : number of cell lines in which the perturbation was measured.
    agreement  : mean pairwise correlation of the source profiles, in [0, 1].
    radius     : relative-error conformal radius at the chosen level, if known.
    tau        : radius above which a delivered profile is flagged.
    """
    if n_sources is None or n_sources <= 0:
        return MEASURE
    if n_sources == 1:
        return ABSTAIN
    if agreement is not None and agreement < min_agreement:
        return LOW_CONFIDENCE
    if radius is not None and radius > tau:
        return LOW_CONFIDENCE
    return ACCEPT


def triage_table(rows, agreement=None, radius=None, **kw):
    """Apply :func:`triage` to a table of targets.

    rows: sequence of dicts with at least ``target`` and ``n_sources``.
    """
    agreement = agreement or {}
    radius = radius or {}
    out = []
    for r in rows:
        t = r["target"]
        out.append({
            "target": t,
            "n_sources": r["n_sources"],
            "state": triage(r["n_sources"], agreement.get(t), radius.get(t), **kw),
        })
    return out


def state_counts(states):
    """Counts per state, in the fixed display order."""
    order = (ACCEPT, LOW_CONFIDENCE, ABSTAIN, MEASURE)
    counts = {k: 0 for k in order}
    for s in states:
        counts[s] = counts.get(s, 0) + 1
    return counts
