"""kt120 (local): the matched ceiling scored with the model's own metric.

kt119 computed the ceiling with the plain centred R2,  1 - SS(y-p)/SS(y-ybar).
kt100 scores the model with a fitted per-programme gain,
   g_k = sum(p y)/sum(p^2),   r2_k = 1 - sum((y - g p)^2)/sum((y - ybar)^2),
and averages r2_k over the 45 programmes.  The two denominators agree but the
model's metric carries one free scale per programme; to make the share an
apples-to-apples ratio the bound must be scored the same way.  This script
recomputes kt119's estimator under the kt100 metric (same rows, same 5-fold
geometry, 3 alpha candidates, 3 selection repeats, 10 partitions for dispersion).

Output: runs/kt120_ceiling_gain.json
"""
import json, numpy as np

D = json.load(open("runs/kt111_dump.json"))
O = {k: np.array(v, float) for k, v in D["obs_per_line"].items()}
LINES = ["hepg2", "jurkat", "k562", "rpe1"]
K100 = json.load(open("runs/kt100_program.json"))
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
NREP, NFOLD = 3, 5


def fit(X, Y, alpha):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    A = (X - mu) / sd
    ym = Y.mean(0)
    W = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (Y - ym))
    return mu, sd, ym, W


def predict(X, Y, alpha, seed, frac=None):
    """frac=None: fit on all training rows. frac=0.6: fit on the same 60% A
    subset that the latent model's arms are restricted to (matched budget)."""
    n = len(Y)
    rs = np.random.RandomState(seed)
    pm = rs.permutation(n)
    P = np.zeros_like(Y)
    for f in range(NFOLD):
        te = pm[f::NFOLD]
        tr = np.setdiff1d(pm, te)
        if frac is not None:
            tr = tr[rs.permutation(len(tr))[: int(round(frac * len(tr)))]]
        mu, sd, ym, W = fit(X[tr], Y[tr], alpha)
        P[te] = ((X[te] - mu) / sd) @ W + ym
    return P


def r2_plain(P, Y):
    return float(1 - ((Y - P) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


def r2_gain(P, Y):
    """kt100 metrics(): per-programme fitted gain, averaged over programmes."""
    g = (P * Y).sum(0) / np.maximum((P ** 2).sum(0), 1e-12)
    res = ((Y - g * P) ** 2).sum(0)
    den = ((Y - Y.mean(0)) ** 2).sum(0)
    return float(np.mean(1 - res / den))


def r2_gain_global(P, Y):
    g = (P * Y).sum() / (P ** 2).sum()
    return float(1 - ((Y - g * P) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


res = {}
for c in LINES:
    X = np.mean([O[l] for l in LINES if l != c], 0)
    Y = O[c]
    # alpha selected on the gain metric (the metric the share uses)
    grid = {al: float(np.mean([r2_gain(predict(X, Y, al, 100 + s, 0.6), Y) for s in range(NREP)]))
            for al in ALPHAS}
    al = max(grid, key=grid.get)
    P = predict(X, Y, al, 0, 0.6)
    gene, prog = K100[c]["gene"]["r2c"], K100[c]["prog"]["r2c"]
    res[c] = {
        "alpha": al,
        "ceiling_gain": grid[al],
        "ceiling_gain_global": r2_gain_global(P, Y),
        "ceiling_plain": r2_plain(P, Y),
        "gene_arm_r2": gene, "prog_arm_r2": prog,
        "gene_share": gene / grid[al], "prog_share": prog / grid[al],
        "alpha_grid": {str(k): v for k, v in grid.items()},
    }
    print(f"{c:7s} ceiling(gain) {grid[al]:.4f} [plain {r2_plain(P, Y):.4f}] | "
          f"gene {gene:.4f} ({100*gene/grid[al]:.1f}%) | "
          f"prog {prog:.4f} ({100*prog/grid[al]:.1f}%)", flush=True)

for c in LINES:
    X = np.mean([O[l] for l in LINES if l != c], 0)
    Y = O[c]
    al = res[c]["alpha"]
    reps = np.array([r2_gain(predict(X, Y, al, s, 0.6), Y) for s in range(10)])
    repsf = np.array([r2_gain(predict(X, Y, al, s), Y) for s in range(10)])
    res[c]["rep_mean_full"] = float(repsf.mean())
    res[c]["rep_sd_full"] = float(repsf.std(ddof=1))
    res[c]["rep_mean"] = float(reps.mean())
    res[c]["rep_sd"] = float(reps.std(ddof=1))
    print(f"{c:7s} ceiling A60 {reps.mean():.4f} +- {reps.std(ddof=1):.4f} | "
          f"full {repsf.mean():.4f} +- {repsf.std(ddof=1):.4f}", flush=True)

gs = [res[c]["gene_share"] for c in LINES]
ps = [res[c]["prog_share"] for c in LINES]
res["_summary"] = {"gene_share_min": float(min(gs)), "gene_share_max": float(max(gs)),
                   "prog_share_min": float(min(ps)), "prog_share_max": float(max(ps)),
                   "n_genes": int(len(O["hepg2"]))}
print("gene arm share %.3f-%.3f | prog arm share %.3f-%.3f"
      % (min(gs), max(gs), min(ps), max(ps)))
json.dump(res, open("runs/kt120_ceiling_gain.json", "w"), indent=1)
print("DONE")
