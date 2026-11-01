"""kt118 (local): decision-level abstention and the accounting of one experiment.

The reliability head in the model was trained on the *profile* error (predicted
row correlation).  But the decision a user makes is narrower: which programme
does this target move.  Are these the same notion of trust?

  (1) Profile head (rel, higher = more reliable): Spearman against the true
      row correlation, and AUC for predicting a top-3 miss.
  (2) Decision head: logistic model on prediction-shape features (peak height,
      peak margin, concentration, plus rel), cross-validated 5x5.
  (3) Coverage curve: among accepted targets, how often is the top-3 right?
  (4) Delivered accuracy: accept a fraction c, measure the rest exactly, so the
      answer handed to the user is right with probability
      c * hit3(accepted) + (1-c) * 1.  This is the per-experiment accounting.

Random-abstention control uses the same coverage; if the heads were not
selective the two curves would coincide.
"""
import json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

R = "runs/"
D = json.load(open(R + "kt111_dump.json"))
P = np.array(D["pred"], float)
Y = np.array(D["truth"], float)
REL = np.array(D["rel"], float)          # predicted row correlation (higher = better)
G = D["pert_names"]
n, K = Y.shape


def rowcorr(A, B):
    A = A - A.mean(1, keepdims=True)
    B = B - B.mean(1, keepdims=True)
    return (A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1)).clip(1e-9)


a = rowcorr(P, Y)
hit3 = np.array([len(set(np.argsort(-np.abs(P[i]))[:3]) & set(np.argsort(-np.abs(Y[i]))[:3])) >= 1
                 for i in range(n)])
if hit3.mean() < 0.5:
    hit3 = ~hit3
print(f"baseline: rowcorr mean {a.mean():.3f}, hit@3 {hit3.mean():.3f}")

sp = float(np.corrcoef(np.argsort(np.argsort(REL)), np.argsort(np.argsort(a)))[0, 1])
auc_prof = float(roc_auc_score(~hit3, -REL))
print(f"profile head: Spearman rel~rowcorr {sp:.3f}; AUC for top-3 miss {auc_prof:.3f}")

ap = np.sort(np.abs(P), 1)[:, ::-1]
feat = np.c_[ap[:, 0], ap[:, 0] - ap[:, 3], ap[:, 0] - ap[:, 2], ap[:, :3].mean(1),
             ap.mean(1), ap.std(1), np.linalg.norm(P, axis=1), REL]
X = (feat - feat.mean(0)) / (feat.std(0) + 1e-9)
y = (~hit3).astype(int)

SEEDS = range(5)
pscore = np.zeros((len(SEEDS), n))
aucs = []
for s in SEEDS:
    cv = StratifiedKFold(5, shuffle=True, random_state=s)
    for tr, te in cv.split(X, y):
        m = LogisticRegression(C=0.5, max_iter=3000).fit(X[tr], y[tr])
        pscore[s, te] = m.predict_proba(X[te])[:, 1]
    aucs.append(roc_auc_score(y, pscore[s]))
print(f"decision head AUC {np.mean(aucs):.3f} +- {np.std(aucs):.3f}")
ps = pscore.mean(0)

COV = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])
rows = []
for c in COV:
    for name, score in (("decision_head", -ps), ("profile_head", REL), ("random", None)):
        reps = 200 if score is None else 1
        acc_h, deliv = [], []
        for rep in range(reps):
            s = np.random.RandomState(rep).permutation(n) if score is None else np.argsort(-score)
            k = int(round(c * n))
            keep = s[:k]
            h = hit3[keep].mean()
            acc_h.append(h)
            deliv.append(c * h + (1 - c))
        rows.append({"coverage": c, "strategy": name,
                     "hit3_accepted": float(np.mean(acc_h)),
                     "delivered_accuracy": float(np.mean(deliv)),
                     "measured_per_100": round(100 * (1 - c), 1)})
COVT = pd.DataFrame(rows)
print(COVT.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# ---- cost-optimal coverage ------------------------------------------------
# per target: accept -> wrong with prob 1-h(c) at cost C; measure -> cost c.
dh = COVT[COVT.strategy == "decision_head"].set_index("coverage")
cost_rows = []
for r in (1, 2, 5, 10, 20, 50, 100):
    loss = {c: c * (1 - dh.loc[c, "hit3_accepted"]) * r + (1 - c) for c in COV}
    best = min(loss, key=loss.get)
    cost_rows.append({"C_over_c": r, "optimal_coverage": best,
                      "optimal_measure_per_100": round(100 * (1 - best), 1),
                      "expected_loss": float(loss[best]),
                      "loss_accept_all": float(loss[1.0]),
                      "loss_measure_all": 1.0})
CST = pd.DataFrame(cost_rows)
print("\nCOST-OPTIMAL COVERAGE")
print(CST.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# where does the decision head put its risk?
r10 = np.argsort(-ps)[: int(round(.1 * n))]
print(f"top-10% risk targets: {len(r10)}, actual hit@3 {hit3[r10].mean():.3f}, "
      f"mean rowcorr {a[r10].mean():.3f}")
print(f"rest: hit@3 {hit3[np.setdiff1d(np.arange(n), r10)].mean():.3f}")

out = {
    "n": n, "baseline_rowcorr": float(a.mean()), "baseline_hit3": float(hit3.mean()),
    "profile_head": {"spearman_rowcorr": sp, "auc_top3_miss": auc_prof},
    "decision_head": {"auc_mean": float(np.mean(aucs)), "auc_sd": float(np.std(aucs)),
                      "features": ["max|z|", "margin_1_4", "margin_1_3", "mean_top3",
                                   "mean_abs", "std", "norm", "profile_head"]},
    "coverage": COVT.to_dict(orient="records"),
    "cost_optimal": CST.to_dict(orient="records"),
    "deploy_90": {"accepted_hit3": float(COVT[(COVT.coverage == .9) &
                                              (COVT.strategy == "decision_head")].hit3_accepted.iloc[0]),
                  "delivered": float(COVT[(COVT.coverage == .9) &
                                          (COVT.strategy == "decision_head")].delivered_accuracy.iloc[0])},
}
json.dump(out, open(R + "kt118_alloc.json", "w"), indent=1)
pd.DataFrame({"gene": G, "rowcorr": a, "hit3": hit3, "profile_head": REL,
              "decision_risk": ps}).to_csv(R + "kt118_per_gene.csv", index=False)
print("DONE")
