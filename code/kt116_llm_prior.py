"""kt116 (local): LLM prior baseline against measured programme profiles.

The question: an LLM that knows only the gene symbol (and the four cell lines
and the 45 Hallmark names) will happily name the programmes a knockdown should
move.  Is VERDICT's measured-data-derived profile worth anything *on top of*
that?  Three comparisons on the same rows, against the same measured truth:

  (1) LLM-only       -- the frozen top-3 (programme, direction) list.
  (2) VERDICT-only   -- the model's predicted 45-vector.
  (3) fused          -- normalize(pred) + lambda * normalize(llm prior),
                        lambda chosen by 5-fold cross-validation over genes.

Plus two null baselines so neither arm gets credit for guessing the modal
answer: a constant "most frequent dominant programme" predictor and the single
most common LLM triple applied to every gene.

Truth = runs/kt111_dump.json (measured Hallmark profile per perturbation,
averaged over the lines in which it was measured).
"""
import json, numpy as np, pandas as pd
from scipy import stats

RUN = "runs/"
PRI = json.load(open(RUN + "kt116_llm_priors.json"))
D = json.load(open(RUN + "kt111_dump.json"))
progs = D["programs"]
pidx = {p: i for i, p in enumerate(progs)}
P = np.array(D["pred"], float)
Y = np.array(D["truth"], float)
names = D["pert_names"]
gidx = {g: i for i, g in enumerate(names)}

# ---- 0. vocabulary check --------------------------------------------------
bad = []
for g, lst in PRI["priors"].items():
    for p, s in lst:
        if p not in pidx:
            bad.append((g, p))
        if int(s) not in (-1, 1):
            bad.append((g, "sign", s))
assert not bad, f"prior vocabulary mismatch: {bad}"

GENES = [g for g in PRI["priors"] if g in gidx]
print(f"{len(GENES)} of {len(PRI['priors'])} prior genes have a measured profile")

CLIN = ["HSPA9", "ATF4", "HMGCR", "FASN", "GCLC", "GSK3B", "HSPA8"]
CLIN = [g for g in CLIN if g in GENES]


def prior_vec(g):
    u = np.zeros(45)
    for p, s in PRI["priors"][g]:
        u[pidx[p]] = s
    return u


def rank_of(i, order):
    return int(np.where(order == i)[0][0]) + 1


rows = []
for g in GENES:
    i = gidx[g]
    y, p = Y[i], P[i]
    torder = np.argsort(-np.abs(y))
    porder = np.argsort(-np.abs(p))
    u = prior_vec(g)
    lup = [pidx[pp] for pp, _ in PRI["priors"][g]]
    lsign = [int(s) for _, s in PRI["priors"][g]]
    rank1 = rank_of(lup[0], torder)
    rankmin = min(rank_of(x, torder) for x in lup)
    # VERDICT: its own top-1 and the overlap of its top-3 with truth top-3
    vtop3 = set(porder[:3].tolist())
    ttop3 = set(torder[:3].tolist())
    # sign of y is relative to the perturbation average (the programme score is
    # z-scored across perturbations), so it is NOT a vs-control direction and
    # is kept for completeness only.
    pct = np.argsort(np.argsort(np.abs(y))) / (len(y) - 1)
    rows.append({
        "target": g,
        "clinical": g in CLIN,
        "truth_top1": progs[torder[0]],
        "llm_top1": progs[lup[0]],
        "verdict_top1": progs[porder[0]],
        "llm_rank1": rank1,
        "llm_best3_rank": rankmin,
        "llm_hit1": rank1 == 1,
        "llm_hit3": rankmin <= 3,
        "llm_hit5": rankmin <= 5,
        "llm_hit10": rankmin <= 10,
        "llm_absz_named": float(np.mean(np.abs(y[lup]))),
        "llm_absz_all": float(np.mean(np.abs(y))),
        "llm_pct_named": float(np.mean(pct[lup])),
        "llm_dir1": bool(np.sign(y[lup[0]]) == lsign[0]),
        "llm_dir3": float(np.mean([np.sign(y[j]) == s for j, s in zip(lup, lsign)])),
        "v_hit1": bool(torder[0] in vtop3) and progs[porder[0]] == progs[torder[0]],
        "v_hit3": len(vtop3 & ttop3),
        "v_hit5": len(set(porder[:5].tolist()) & set(torder[:5].tolist())),
        "v_dir3": float(np.mean([np.sign(p[j]) == np.sign(y[j]) for j in porder[:3]])),
        "rowcorr_pred": float(np.corrcoef(p, y)[0, 1]),
        "rowcorr_llm": float(np.corrcoef(u, y)[0, 1]),
        "jaccard": len(vtop3 & set(lup)) / len(vtop3 | set(lup)),
        "u": u, "p": p, "y": y,
    })

T = pd.DataFrame(rows)


def binom(k, n, p, alt="greater"):
    return float(stats.binomtest(int(k), int(n), p, alternative=alt).pvalue)


def summarize(d, tag):
    n = len(d)
    return {
        "tag": tag, "n": n,
        "llm_hit1": int(d.llm_hit1.sum()), "llm_hit1_p": binom(d.llm_hit1.sum(), n, 1 / 45),
        "llm_hit3": int(d.llm_hit3.sum()), "llm_hit3_p": binom(d.llm_hit3.sum(), n, 3 / 45),
        "llm_hit5": int(d.llm_hit5.sum()), "llm_hit5_p": binom(d.llm_hit5.sum(), n, 5 / 45),
        "llm_median_rank1": float(d.llm_rank1.median()),
        "llm_pct_named": float(d.llm_pct_named.mean()),
        "llm_absz_named": float(d.llm_absz_named.mean()),
        "llm_absz_all": float(d.llm_absz_all.mean()),
        "llm_dir3_vs_centred": float(d.llm_dir3.mean()),
        "v_hit1": int(d.v_hit1.sum()),
        "v_hit3_ge1": int((d.v_hit3 >= 1).sum()),
        "v_hit3_exact": float(np.mean(d.v_hit3 == 3)),
        "v_mean_overlap3": float(d.v_hit3.mean()),
        "v_dir3": float(d.v_dir3.mean()),
        "rowcorr_pred": float(d.rowcorr_pred.mean()),
        "rowcorr_llm": float(d.rowcorr_llm.mean()),
        "jaccard_llm_v": float(d.jaccard.mean()),
    }


S_all = summarize(T, "all")
S_clin = summarize(T[T.clinical], "clinical")
print(json.dumps(S_all, indent=1))
print(json.dumps(S_clin, indent=1))

# test: VERDICT top-3 overlap vs LLM hit@3, paired over genes
n = len(T)
b = int(((T.v_hit3 >= 1) & ~T.llm_hit3).sum())
c = int((~((T.v_hit3 >= 1)) & T.llm_hit3).sum())
mcn = float(stats.binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0
print(f"VERDICT-hits-LLM-misses {b}, LLM-hits-VERDICT-misses {c}, exact McNemar p={mcn:.4g}")

# ---- null baselines --------------------------------------------------------
mode_prog = T.truth_top1.value_counts().idxmax()
mode_rate = float((T.truth_top1 == mode_prog).mean())
# most common LLM triple
from collections import Counter
trip = Counter(tuple(sorted(pp for pp, _ in PRI["priors"][g])) for g in GENES)
common_triple, common_n = trip.most_common(1)[0]
cidx = [pidx[x] for x in common_triple]
const_hit3 = float(np.mean([len(set(cidx) & set(np.argsort(-np.abs(Y[gidx[g]]))[:3].tolist())) >= 1
                            for g in GENES]))
print(f"constant-dominant null: '{mode_prog}' is truth-top1 in {mode_rate:.1%}; "
      f"most common LLM triple {common_triple} recovers truth top-3 in {const_hit3:.1%}")

# generic-template baseline scored row-by-row (same definition as llm_hit3)
T["tmpl_hit3"] = [len(set(cidx) & set(np.argsort(-np.abs(Y[gidx[g]]))[:3].tolist())) >= 1
                  for g in GENES]
S_tmpl = {"const_triple_hit3": int(T.tmpl_hit3.sum()), "n": n,
          "const_triple_hit3_rate": float(T.tmpl_hit3.mean())}
print(json.dumps(S_tmpl, indent=1))

# LLM names programmes of larger magnitude?  (sign-free, fair to the centring)
d_abs = (T.llm_absz_named - T.llm_absz_all).values
rngm = np.random.RandomState(1)
bsa = np.array([d_abs[rngm.randint(0, n, n)].mean() for _ in range(10000)])
mag = {"mean_absz_named": float(T.llm_absz_named.mean()),
       "mean_absz_all": float(T.llm_absz_all.mean()),
       "delta": float(d_abs.mean()),
       "ci95": [float(np.percentile(bsa, 2.5)), float(np.percentile(bsa, 97.5))],
       "p_boot_one_sided": float(np.mean(bsa <= 0)),
       "mean_pct_rank_named": float(T.llm_pct_named.mean())}
print("MAGNITUDE", json.dumps(mag, indent=1))

# random vs clinical hit rates
rnd = T[~T.clinical]
S_rnd = {"n": len(rnd), "llm_hit1": int(rnd.llm_hit1.sum()), "llm_hit3": int(rnd.llm_hit3.sum()),
         "llm_hit3_rate": float(rnd.llm_hit3.mean()),
         "llm_hit3_p": binom(rnd.llm_hit3.sum(), len(rnd), 3 / 45),
         "v_hit3_ge1": int((rnd.v_hit3 >= 1).sum()),
         "v_hit3_rate": float((rnd.v_hit3 >= 1).mean())}
print("RANDOM-80", json.dumps(S_rnd, indent=1))

# ---- fusion with cross-fitted lambda --------------------------------------
rng = np.random.RandomState(0)
folds = np.array_split(rng.permutation(n), 5)
LAM = np.linspace(0, 1.0, 21)


def unit(v):
    nv = np.linalg.norm(v)
    return v / nv if nv > 1e-12 else v


def score(d, lam):
    rc, h3 = [], []
    for _, r in d.iterrows():
        f = unit(r.p) + lam * unit(r.u)
        rc.append(np.corrcoef(f, r.y)[0, 1])
        h3.append(len(set(np.argsort(-np.abs(f))[:3].tolist()) &
                     set(np.argsort(-np.abs(r.y))[:3].tolist())) >= 1)
    return float(np.mean(rc)), float(np.mean(h3))


oof_rc = np.zeros(n); oof_h3 = np.zeros(n); chosen = []
for k in range(5):
    te = folds[k]; tr = np.setdiff1d(np.arange(n), te)
    best, bl = -9, 0.0
    for lam in LAM:
        s, _ = score(T.iloc[tr], lam)
        if s > best:
            best, bl = s, lam
    chosen.append(bl)
    for j in te:
        f = unit(T.iloc[j].p) + bl * unit(T.iloc[j].u)
        oof_rc[j] = np.corrcoef(f, T.iloc[j].y)[0, 1]
        oof_h3[j] = len(set(np.argsort(-np.abs(f))[:3].tolist()) &
                        set(np.argsort(-np.abs(T.iloc[j].y))[:3].tolist())) >= 1
rc0, h30 = score(T, 0.0)

d_rc = oof_rc - T.rowcorr_pred.values
bs = np.array([d_rc[rng.randint(0, n, n)].mean() for _ in range(10000)])
fus = {
    "lambda_chosen": [float(x) for x in chosen],
    "lambda_mode": float(stats.mode(np.round(chosen, 2), keepdims=False).mode),
    "oof_rowcorr_lambda0": float(rc0), "oof_rowcorr_fused": float(oof_rc.mean()),
    "delta_rowcorr": float(oof_rc.mean() - rc0),
    "delta_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
    "delta_p_boot": float(np.mean(bs <= 0)),
    "oof_hit3_lambda0": float(h30), "oof_hit3_fused": float(oof_h3.mean()),
}
print("FUSION", json.dumps(fus, indent=1))

# per-programme direction table (LLM vs truth) for the clinical subset
det = []
for g in CLIN:
    i = gidx[g]
    y = Y[i]
    for pp, s in PRI["priors"][g]:
        j = pidx[pp]
        det.append({"target": g, "programme": pp, "llm_sign": s,
                    "truth_z": round(float(y[j]), 3),
                    "measured_sign": int(np.sign(y[j])), "ok": int(np.sign(y[j]) == s),
                    "verdict_z": round(float(P[i, j]), 3)})
DET = pd.DataFrame(det)
print(DET.to_string(index=False))

c1 = Counter(PRI["priors"][g][0][0] for g in GENES)
t1 = Counter(progs[np.argsort(-np.abs(Y[gidx[g]]))[0]] for g in GENES)
t3 = Counter()
for g in GENES:
    for j in np.argsort(-np.abs(Y[gidx[g]]))[:3]:
        t3[progs[j]] += 1
div = {"llm_top1_distinct": len(c1), "llm_top1_top5": c1.most_common(5),
       "truth_top1_distinct": len(t1), "truth_top1_top5": t1.most_common(5),
       "truth_top3_distinct_members": len(t3), "truth_top3_top5": t3.most_common(5)}
print("DIVERSITY", json.dumps(div, indent=1))

out = {
    "protocol": PRI["protocol"],
    "n_compared": n, "n_excluded_no_truth": sorted(set(PRI["priors"]) - set(GENES)),
    "all": S_all, "clinical": S_clin,
    "mcnemar": {"verdict_only": b, "llm_only": c, "p": mcn},
    "null": {"modal_programme": mode_prog, "modal_rate": mode_rate,
             "common_llm_triple": list(common_triple), "const_hit3": const_hit3,
             "const_triple_hit3_rate": float(T.tmpl_hit3.mean())},
    "random80": S_rnd,
    "diversity": div,
    "magnitude": mag,
    "fusion": fus,
    "per_gene": T.drop(columns=["u", "p", "y"]).to_dict(orient="records"),
    "clinical_detail": det,
}
json.dump(out, open(RUN + "kt116_llm_baseline.json", "w"), indent=1)
T.drop(columns=["u", "p", "y"]).to_csv(RUN + "kt116_llm_baseline.csv", index=False)
DET.to_csv(RUN + "kt116_clinical_directions.csv", index=False)
print("DONE")
