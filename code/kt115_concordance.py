"""kt115 (local): literature concordance of the model's predicted programme
profiles for the ten host-directed candidates with an unambiguous expectation.

The question is narrow: without any label, does the model put the programme
that prior biology says should move in the top of the profile, and with the
right sign?  The observed-mode rows (MTOR, HSPA9) are counted separately
because they are in-split for those targets.
"""
import json, numpy as np, pandas as pd
from scipy import stats

RUN = r"C:/Users/1/Desktop/P16/runs/"
dep = json.load(open(RUN + "kt113_deploy.json"))
progs = dep["programs"]
pidx = {p: i for i, p in enumerate(progs)}
ORIG = np.array(progs)
EXPECT = {
    "HMGCR": ("Cholesterol Homeostasis", +1), "MTOR": ("mTORC1 Signaling", -1),
    "PTEN": ("mTORC1 Signaling", +1), "MFN2": ("mTORC1 Signaling", +1),
    "ATF4": ("Unfolded Protein Response", -1), "TFRC": ("IL-6/JAK/STAT3 Signaling", +1),
    "MYC": ("Myc Targets V1", -1), "FASN": ("Fatty Acid Metabolism", -1),
    "HSPA9": ("Oxidative Phosphorylation", -1), "GSK3B": ("Wnt-beta Catenin Signaling", +1),
}
rows = []
for tgt, (prog, sgn) in EXPECT.items():
    per = dep["cand"].get(tgt)
    if not per:
        continue
    pr = [r["profile"] for r in per.values() if "profile" in r]
    ob = [r["observed"] for r in per.values() if "observed" in r]
    mode = "predicted" if pr else "observed"
    v = np.mean(pr, 0) if pr else np.mean(ob, 0)
    rank = int(np.where(ORIG[np.argsort(-np.abs(v))] == prog)[0][0]) + 1
    # within-line rank and sign consistency
    line_rk, line_sg = [], []
    for c, r in per.items():
        vv = r.get("profile", r.get("observed"))
        if vv is None:
            continue
        vv = np.array(vv)
        o = ORIG[np.argsort(-np.abs(vv))]
        line_rk.append(int(np.where(o == prog)[0][0]) + 1)
        line_sg.append(np.sign(vv[pidx[prog]]))
    rows.append({"target": tgt, "expected": prog, "mode": mode,
                 "n_lines": len(per), "consensus_rank": rank,
                 "consensus_z": float(v[pidx[prog]]),
                 "sign_ok": bool(np.sign(v[pidx[prog]]) == sgn),
                 "line_ranks": line_rk, "line_signs_ok": [bool(s == sgn) for s in line_sg],
                 "frac_lines_sign_ok": float(np.mean([s == sgn for s in line_sg])),
                 "z_by_line": {c: float(np.array(r.get("profile", r.get("observed")))[pidx[prog]])
                               for c, r in per.items() if r.get("profile", r.get("observed")) is not None}})

D = pd.DataFrame(rows).sort_values("consensus_rank")
pd.set_option("display.width", 220)
print(D[["target", "expected", "mode", "n_lines", "consensus_rank", "consensus_z", "sign_ok", "line_ranks"]].to_string(index=False))

pred = D[D["mode"] == "predicted"]
allr = D


def binom(k, n, p, alt="greater"):
    return float(stats.binomtest(k, n, p, alternative=alt).pvalue)


res = {}
for name, d in (("predicted_only", pred), ("all_ten", allr)):
    n = len(d)
    res[name] = {
        "n": n,
        "top3": int((d.consensus_rank <= 3).sum()), "top3_p": binom(int((d.consensus_rank <= 3).sum()), n, 3 / 45),
        "top5": int((d.consensus_rank <= 5).sum()), "top5_p": binom(int((d.consensus_rank <= 5).sum()), n, 5 / 45),
        "top10": int((d.consensus_rank <= 10).sum()), "top10_p": binom(int((d.consensus_rank <= 10).sum()), n, 10 / 45),
        "median_rank": float(d.consensus_rank.median()),
        "mean_rank": float(d.consensus_rank.mean()),
        "sign_ok": int(d.sign_ok.sum()), "sign_p": binom(int(d.sign_ok.sum()), n, 0.5),
        "frac_lines_sign_ok": float(d.frac_lines_sign_ok.mean()),
    }
    print(name, json.dumps(res[name], indent=1))

# hit rate restricted to programmes that are not in the top of every profile
# (i.e. a null baseline for arbitrary targets) -- sanity: random target->programme
rng = np.random.RandomState(0)
null_rank = []
for _ in range(20000):
    rs = []
    for _, r in allr.iterrows():
        rr = rng.choice(np.arange(1, 46), 1)[0]
        rs.append(rr)
    null_rank.append(np.mean(np.array(rs) <= 3))
print(f"random target->programme top3 rate: {np.mean(null_rank):.3f} (theory 3/45={3/45:.3f})")

json.dump({"rows": rows, "summary": res}, open(RUN + "kt115_concordance.json", "w"), indent=1)
D.to_csv(RUN + "kt115_concordance.csv", index=False)
print("DONE")
