---
title: Qyunhan
library_name: qyunhan
license: mit
tags:
  - virtual-cell
  - perturbation-response
  - conformal-prediction
  - selective-prediction
  - tuberculosis
---

# Qyunhan: a virtual cell that separates what it knows from what it doubts

A program-resolution virtual cell with calibrated abstention, for host-directed
target triage in tuberculosis.

Qyunhan has two halves. The **space** is a closed-form shared-latent estimator
that places four cell lines in one low-dimensional space and predicts a target
line's response from its source lines. The **hold** is a split-conformal layer
on *relative* error, ordered by a hardness head, which decides whether a claim
is delivered, flagged, abstained from, or returned as an experiment.

The two halves answer different questions, and the package keeps them apart:

| question | component | what it buys |
|---|---|---|
| what is the program profile? | `qyunhan.estimator` | a prediction, no statement of trust |
| how wrong might it be? | `qyunhan.conformal` | a per-row radius at a nominal level |
| which rows are hard? | `qyunhan.hardness` | an ordering, fitted without labels of the target line |
| what may be said? | `qyunhan.triage` | accept / low-confidence / abstain / measure |

## Install

```bash
pip install -e .          # numpy only; the analysis scripts add torch (see code/)
```

## Quickstart

```python
import numpy as np
from qyunhan import estimator, conformal, hardness, triage

# X[c] is (n_perturbations x n_genes) log fold change for cell line c.
models = {c: estimator.fit_line(X[c]) for c in ("k562", "rpe1", "hepg2", "jurkat")}

# Predict jurkat from the three lines in which the perturbation was measured.
z = estimator.consensus_latents({c: estimator.project(models[c], X[c][q])
                                 for c in ("k562", "rpe1", "hepg2")})
delta_hat = estimator.decode(models["jurkat"], z)

# Per-row radius on relative error, scaled by a label-free hardness proxy.
cal = conformal.nonconformity(y_cal, p_cal, s_cal)
r = conformal.radii(cal, s_test=conformal.magnitude_proxy(p_test), alpha=0.1)

# Three-state triage of one target.
state = triage.triage(n_sources=2, agreement=0.42, radius=r[0])
```

## Self-test

```bash
python -m qyunhan.selftest                     # synthetic, numpy only
python -m qyunhan.selftest --results PATH      # reproduces Tables 3 and 6
```

The synthetic mode checks the estimator against a known rank-4 signal, checks
that relative-error intervals attain their nominal marginal coverage, and checks
that the triage table reproduces the four states of Table 5. With `--results`
(or `QYUNHAN_RESULTS`) it recomputes every printed value of **Table 3**
(abstention ablation) and **Table 6** (source-count ablation and reliability-head
transfer) from the archived result files and compares them to three decimals;
all 42 values reproduce.

## Repository layout

```
qyunhan/        the package: estimator, conformal, hardness, triage, selftest
code/           analysis scripts kt109 - kt120, as run
docs/           reproduction notes and the data dictionary
```

The upstream estimation stages (kt99 - kt108) are consolidated into the
`qyunhan` package; their outputs are the `kt104_*`, `kt106_*` and `kt110_*`
files the self-test consumes.

## Data availability

- **Perturbation compendium.** Public. PIE essential compendium (Arc Institute);
  MSigDB Hallmark 2020 and KEGG 2021 gene sets are downloaded from MSigDB by the
  user and are not redistributed here.
- **Auxiliary resources.** DepMap CRISPRGeneEffect 26Q1, STRING v12,
  UniProtKB/Swiss-Prot 2025, and the ESM2-t30-150M checkpoint are public.
- **Patient data.** Not in this repository and not redistributable: the
  tuberculosis pseudobulk matrix is from the authors' cohort, held under ethics
  approval KY20250227, and is available from the lead contact under that approval.
- **Result and figure artefacts.** Archived result files, the processed
  intermediate arrays and the figure sources are supplied to editors and referees
  on request while the manuscript is under review.

## License

Code: MIT (see `LICENSE`). The manuscript text, figures and result artefacts are
not covered by that license.

## Contact

Jiangdong Wu — 1556874645@qq.com
