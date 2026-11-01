# Reproducing the analyses

## What is here

`code/` holds the analysis scripts exactly as they were run (`kt109`-`kt120`).
They read a processed snapshot of the PIE essential compendium from a directory
called `cov/`, and they write JSON/CSV into `runs/`.

```
cov/meta.json          gene order, context and perturbation vocabularies
cov/ctx_ids.npy        cell line index per row
cov/pert_ids.npy       perturbation index per row
cov/delta_p.npy        (rows x genes) log fold change, float32, memmapped
```

`cov/` is built from the public PIE essential compendium (Replogle et al.;
Nadig et al.) and is not redistributed with this repository.

## Stage map

| script | what it produces | used in |
|---|---|---|
| `kt109_tbpatients.py` | programme profile of the 15-donor tuberculosis cohort | Fig 2a, Section 2.9 |
| `kt110_sources.py` | source-count ablation; reliability-head transfer | Table 6 |
| `kt111_dump.py` | per-perturbation predicted and measured profiles | Fig 1a, Fig 3 |
| `kt112_clinical.py` | ten-candidate triage table | Table 5, Fig 2c |
| `kt113_dumpdeploy.py` | deployment dump | Fig 4a |
| `kt114_patient_anchor.py` | global shift and expression confound | Section 2.9 |
| `kt115_concordance.py` | per-candidate concordance with literature | Table 7 |
| `kt116_llm_prior.py` | frozen language-model sign prior | Fig 3i |
| `kt117_seq_prep.py`, `kt117_seq_gapfill.py`, `kt117_zeroshot.py` | ESM2 sequence-only predictor, leave-gene-out | Fig 2d, Fig 3d, Fig 4d |
| `kt118_alloc.py`, `kt118_decision.py`, `kt118b_decision_agg.py` | decision head vs profile head | Table 4, Fig 4a |
| `kt119_ceiling_match.py`, `kt120_ceiling_gain.py`, `kt120_seqrad.py` | matched-ceiling reference and sequence radii | Fig 3b, Fig 2d |

The upstream estimation stages (kt99-kt108) are consolidated into the `qyunhan`
package. Their outputs are `kt104_relconf.json`, `kt106_armablation.json` and
`kt110_sources.json`, which the self-test reads.

## Order of execution

```bash
python code/kt110_sources.py            # Table 6
python code/kt111_dump.py               # per-perturbation dump
python code/kt112_clinical.py           # Table 5
python code/kt113_dumpdeploy.py
python code/kt114_patient_anchor.py
python code/kt115_concordance.py
python code/kt116_llm_prior.py
python code/kt117_seq_prep.py && python code/kt117_seq_gapfill.py && python code/kt117_zeroshot.py
python code/kt118_alloc.py && python code/kt118_decision.py && python code/kt118b_decision_agg.py
python code/kt119_ceiling_match.py
python code/kt120_ceiling_gain.py && python code/kt120_seqrad.py
```

All scripts except `kt109` and `kt117_seq_prep` need a CUDA device. `kt109` reads
the cohort pseudobulk matrix, which is not redistributed.
