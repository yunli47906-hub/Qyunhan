"""kt119 (local): a ceiling measured the same way as the model.

Section 2.1 previously compared a model R2 on the four-line consensus with a
split-half reliability of a single line.  Those are different denominators (the
older kt37 artifact itself returns share_of_ceiling = 1.39 for k562).  Here the
ceiling is measured with the *same* task, target, space and CV geometry as the
model it bounds: predict a target line's 45-programme profile from the mean of
the other three lines with a fitted linear read-out.

Inputs (all local):
  runs/kt111_dump.json  obs_per_line[line]  (638 x 45 measured programme z)
  runs/kt100_program.json  per-line r2c of the gene and prog arms
Output:
  runs/kt119_ceiling_match.json
"""
import json, numpy as np

D = json.load(open("runs/kt111_dump.json"))
O = {k: np.array(v, float) for k, v in D["obs_per_line"].items()}
LINES = ["hepg2", "jurkat", "k562", "rpe1"]
K100 = json.load(open("runs/kt100_program.json"))
ALPHAS = [0.1, 1.0, 10.0]
NREP, NFOLD = 3, 5


def fit(X, Y, alpha):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    A = (X - mu) / sd
    ym = Y.mean(0)
    W = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (Y - ym))
    return mu, sd, ym, W


def cv_r2(X, Y, alpha, seed):
    n = len(Y)
    pm = np.random.RandomState(seed).permutation(n)
    P = np.zeros_like(Y)
    for f in range(NFOLD):
        te = pm[f::NFOLD]
        tr = np.setdiff1d(pm, te)
        mu, sd, ym, W = fit(X[tr], Y[tr], alpha)
        P[te] = ((X[te] - mu) / sd) @ W + ym
    mu = Y.mean(0)
    return 1 - ((Y - P) ** 2).sum() / ((Y - mu) ** 2).sum()


res = {}
for c in LINES:
    others = np.mean([O[l] for l in LINES if l != c], 0)
    grid = {al: float(np.mean([cv_r2(others, O[c], al, s) for s in range(NREP)]))
            for al in ALPHAS}
    al = max(grid, key=grid.get)
    ceil = grid[al]
    gene = K100[c]["gene"]["r2c"]
    prog = K100[c]["prog"]["r2c"]
    res[c] = {
        "ceiling_r2": ceil, "alpha": al,
        "gene_arm_r2": gene, "prog_arm_r2": prog,
        "gene_share": gene / ceil, "prog_share": prog / ceil,
        "alpha_grid": {str(k): v for k, v in grid.items()},
    }
    print(f"{c:7s} ceiling {ceil:.4f} | gene {gene:.4f} ({100*gene/ceil:.1f}%) "
          f"| prog {prog:.4f} ({100*prog/ceil:.1f}%)", flush=True)

# spread over replicate CV partitions (a bootstrap that resamples rows is invalid
# here: duplicated perturbations leak between folds and inflate R2 by ~0.04)
for c in LINES:
    others = np.mean([O[l] for l in LINES if l != c], 0)
    al = res[c]["alpha"]
    reps = np.array([cv_r2(others, O[c], al, s) for s in range(10)])
    res[c]["rep_mean"] = float(reps.mean())
    res[c]["rep_sd"] = float(reps.std(ddof=1))
    print(f"{c:7s} ceiling {reps.mean():.4f} +- {reps.std(ddof=1):.4f} "
          f"(10 CV partitions, min {reps.min():.4f} max {reps.max():.4f})", flush=True)

shares = [res[c]["prog_share"] for c in LINES]
res["_summary"] = {"prog_share_min": float(min(shares)), "prog_share_max": float(max(shares)),
                   "prog_share_mean": float(np.mean(shares)),
                   "n_genes": int(len(O["hepg2"]))}
print("prog-arm share of matched ceiling: %.3f-%.3f (mean %.3f)" %
      (min(shares), max(shares), np.mean(shares)))
json.dump(res, open("runs/kt119_ceiling_match.json", "w"), indent=1)
print("DONE")
