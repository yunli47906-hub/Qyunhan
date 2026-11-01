"""Self-test for the Qyunhan package.

Two modes.

    python -m qyunhan.selftest
        Synthetic smoke test.  Exercises the estimator, the conformal layer, the
        hardness head and the triage table on data with a known answer, and
        checks that the relative-error intervals attain their nominal coverage.
        Needs nothing but numpy.

    python -m qyunhan.selftest --results PATH
        Reproduces Table 3 and Table 6 of the paper from the archived result
        files (kt104_relconf.json, kt106_armablation.json, kt110_sources.json)
        and compares every printed value.  PATH may also be given through the
        QYUNHAN_RESULTS environment variable.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from . import conformal, estimator, hardness, triage

# --- values printed in the paper -------------------------------------------

TABLE3 = {
    # ordering:                    (AUC,   rowcorr@10%, @50%, @100%)
    "random":                     (0.460, 0.453, 0.460, 0.462),
    "cross-source disagreement":  (0.529, 0.597, 0.532, 0.462),
    "cross-arm disagreement":     (0.536, 0.603, 0.547, 0.462),
    "predicted magnitude (1/|p|)": (0.561, 0.730, 0.550, 0.462),
    "learned hardness head":      (0.576, 0.732, 0.576, 0.462),
    "oracle (true relative error)": (0.618, 0.797, 0.629, 0.462),
}

TABLE6_TOP = {  # s -> (rowcorr, dominant hit@3, hit@1)
    "1": (0.447, 0.267, 0.136),
    "2": (0.485, 0.299, 0.156),
    "3": (0.494, 0.309, 0.163),
}

TABLE6_BOTTOM = {  # held-out line -> (head, magnitude, oracle)
    "hepg2": (0.620, 0.623, 0.731),
    "jurkat": (0.569, 0.554, 0.668),
    "k562": (0.588, 0.584, 0.686),
    "rpe1": (0.627, 0.622, 0.735),
}

ATOL = 5e-4


def _load(results_dir, name):
    with open(os.path.join(results_dir, name), encoding="utf-8") as fh:
        return json.load(fh)


def _cmp(rows, label, got, want):
    ok = abs(got - want) <= ATOL
    rows.append((f"{label:<34}", f"{got:.4f}", f"{want:.3f}", "ok" if ok else "FAIL"))
    return ok


def reproduce_tables(results_dir):
    """Compare the archived results against every value printed in Tables 3 and 6."""
    rel = _load(results_dir, "kt104_relconf.json")
    arm = _load(results_dir, "kt106_armablation.json")
    src = _load(results_dir, "kt110_sources.json")

    r, prog = rel["res"], arm["res"]["prog"]
    idx = {c: i for i, c in enumerate(rel["coverages"])}
    i10, i50, i100 = idx[0.1], idx[0.5], idx[1.0]

    sources = {
        "random": (r["auc_random"], r["sel_corr_random"]),
        "cross-source disagreement": (r["auc_cv_src"], r["sel_corr_cv_src"]),
        "cross-arm disagreement": (r["auc_agree"], r["sel_corr_agree"]),
        "predicted magnitude (1/|p|)": (r["auc_mag"], r["sel_corr_mag"]),
        "learned hardness head": (prog["auc_head"], prog["sel_head"]),
        "oracle (true relative error)": (r["auc_oracle"], r["sel_corr_oracle"]),
    }

    rows, ok = [], True
    print("Table 3 - abstention ablation (Hallmark, programme arm)\n")
    print(f"{'ordering':<34} {'computed':>9} {'printed':>8}  result")
    print("-" * 62)
    for name, want in TABLE3.items():
        auc, curve = sources[name]
        ok &= _cmp(rows, name + " / AUC", auc, want[0])
        ok &= _cmp(rows, name + " / @10%", curve[i10], want[1])
        ok &= _cmp(rows, name + " / @50%", curve[i50], want[2])
        ok &= _cmp(rows, name + " / @100%", curve[i100], want[3])
    for line in rows:
        print(f"{line[0]:<34} {line[1]:>9} {line[2]:>8}  {line[3]}")

    rows = []
    print("\nTable 6, top - source-count ablation\n")
    print(f"{'cell':<34} {'computed':>9} {'printed':>8}  result")
    print("-" * 62)
    for s, want in TABLE6_TOP.items():
        got = src["source_ablation"][s]
        ok &= _cmp(rows, f"s={s} / rowcorr", got["rowcorr"], want[0])
        ok &= _cmp(rows, f"s={s} / dominant hit@3", got["hit3"], want[1])
        ok &= _cmp(rows, f"s={s} / hit@1", got["hit1"], want[2])
    for line in rows:
        print(f"{line[0]:<34} {line[1]:>9} {line[2]:>8}  {line[3]}")

    rows = []
    print("\nTable 6, bottom - reliability-head transfer\n")
    print(f"{'cell':<34} {'computed':>9} {'printed':>8}  result")
    print("-" * 62)
    for line_name, want in TABLE6_BOTTOM.items():
        got = src["head_transfer"][line_name]
        ok &= _cmp(rows, f"{line_name} / head", got["auc_transfer"], want[0])
        ok &= _cmp(rows, f"{line_name} / magnitude", got["auc_mag"], want[1])
        ok &= _cmp(rows, f"{line_name} / oracle", got["auc_oracle"], want[2])
    for line in rows:
        print(f"{line[0]:<34} {line[1]:>9} {line[2]:>8}  {line[3]}")

    return ok


def synthetic(seed=0):
    """Known-answer smoke test; returns True when every assertion holds."""
    rng = np.random.default_rng(seed)
    lines = ["k562", "rpe1", "hepg2", "jurkat"]
    n_pert, n_gene, n_prog, rank = 120, 60, 12, 4

    # One shared latent per perturbation, one decoder per line.
    Z = rng.normal(size=(n_pert, rank))
    models, X = {}, {}
    for c in lines:
        B = rng.normal(size=(n_gene, rank))
        models[c] = {"mu": rng.normal(size=n_gene), "B": B, "r": rank, "s": np.ones(rank)}
        X[c] = models[c]["mu"] + Z @ B.T + 0.02 * rng.normal(size=(n_pert, n_gene))

    # Estimator: exact recovery of the latent up to the fitted basis.
    fit = estimator.fit_line(X["k562"], r=rank)
    assert fit["B"].shape == (n_gene, rank), "decoder must be p x r"
    assert fit["mu"].shape == (n_gene,), "mean must be p"
    Zhat = estimator.project(fit, X["k562"])
    assert Zhat.shape == (n_pert, rank)
    # The latent is identified only up to rotation, so the checkable signature of
    # a rank-4 signal is the spectral gap at four, not agreement of the axes.
    s = fit["s"]
    assert s[rank - 1] > 10 * s[rank], "no spectral gap at the fitted rank"
    assert np.linalg.matrix_rank(Zhat) == rank
    Xhat = estimator.decode(fit, Zhat)
    recon = np.linalg.norm(Xhat - X["k562"]) / np.linalg.norm(X["k562"] - fit["mu"])
    assert recon < 0.05, "rank-r reconstruction carries more than the noise floor"

    # Rank selection returns something sane on a rank-r matrix.
    auto = estimator.fit_line(X["k562"])
    assert 1 <= auto["r"] <= n_gene

    # Consensus of identical sources is that source.
    zc = estimator.consensus_latents({c: Zhat for c in lines})
    assert np.allclose(zc, Zhat, atol=1e-8)

    # Conformal layer: relative-error intervals attain nominal marginal coverage
    # when the hardness proxy is constant.
    y = rng.normal(size=(400, n_prog))
    noise = rng.normal(size=y.shape) * rng.uniform(0.2, 0.8, size=(len(y), 1))
    p = y + noise
    cal, test = slice(0, 200), slice(200, None)
    scores = conformal.nonconformity(y[cal], p[cal])
    for alpha in (0.1, 0.2, 0.5):
        q = conformal.quantile(scores, alpha)
        covered = conformal.relative_error(y[test], p[test]) <= q
        assert covered.mean() >= 1 - alpha - 0.06, f"undercoverage at alpha={alpha}"
        assert covered.mean() <= 1 - alpha + 0.08, f"overcoverage at alpha={alpha}"

    # Hardness head: ranking by the true relative error is the oracle.
    err = conformal.relative_error(y, p)
    cov_grid = np.linspace(0.1, 1.0, 10)
    # Accepting the least-error rows first must accept a smaller mean error.
    assert conformal.risk_coverage_auc(err, err, cov_grid) < \
        conformal.risk_coverage_auc(err, -err, cov_grid)

    F = hardness.shape_features(p)
    assert F.shape == (len(p), 7)
    head = hardness.HardnessHead(lam=1.0).fit(F[cal], err[cal])
    assert head.predict(F[test]).shape == (len(y) - 200,)

    # Triage reproduces the four states of Table 5.
    assert triage.triage(0) == triage.MEASURE
    assert triage.triage(1, agreement=0.0) == triage.ABSTAIN
    assert triage.triage(2, agreement=0.20) == triage.LOW_CONFIDENCE
    assert triage.triage(2, agreement=0.36) == triage.ACCEPT
    assert triage.triage(2, agreement=0.90, radius=0.9, tau=0.5) == triage.LOW_CONFIDENCE
    counts = triage.state_counts([triage.ACCEPT, triage.ACCEPT, triage.LOW_CONFIDENCE,
                                  triage.ABSTAIN, triage.MEASURE])
    assert counts == {triage.ACCEPT: 2, triage.LOW_CONFIDENCE: 1,
                      triage.ABSTAIN: 1, triage.MEASURE: 1}
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default=os.environ.get("QYUNHAN_RESULTS"),
                    help="directory holding kt104/kt106/kt110 result files")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    print("Qyunhan self-test\n" + "=" * 62)
    synthetic(args.seed)
    print("synthetic checks: ok\n")

    if not args.results:
        print("Archived result files not supplied; skipping Tables 3 and 6.")
        print("Re-run with --results PATH (or set QYUNHAN_RESULTS) to reproduce them.")
        return 0

    ok = reproduce_tables(args.results)
    print("\n" + ("all printed values reproduced" if ok else "MISMATCH - see FAIL rows"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
