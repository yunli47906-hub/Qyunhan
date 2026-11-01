"""kt112: the clinical chain.

For each TB host-directed candidate with an unambiguous literature expectation,
compare the model's predicted program profile with that expectation and with
the program's effect in 15 TB patients (5 DR / 5 DS / 5 HC).
"""
import json, numpy as np, pandas as pd
from scipy import stats

RUN = r"C:/Users/1/Desktop/P16/runs/"
dep = json.load(open(RUN + "kt113_deploy.json"))
tb = pd.read_csv(RUN + "kt109_tb_programs.csv").set_index("program")
progs = dep["programs"]
pidx = {p: i for i, p in enumerate(progs)}

EXPECT = {
    "HMGCR": ("Cholesterol Homeostasis", +1, "statin target; SREBP feedback"),
    "MTOR": ("mTORC1 Signaling", -1, "rapamycin target"),
    "PTEN": ("mTORC1 Signaling", +1, "PI3K-AKT brake loss"),
    "MFN2": ("mTORC1 Signaling", +1, "MAM / mitochondrial dynamics"),
    "ATF4": ("Unfolded Protein Response", -1, "integrated stress response"),
    "TFRC": ("IL-6/JAK/STAT3 Signaling", +1, "iron acquisition / inflammatory axis"),
    "MYC": ("Myc Targets V1", -1, "MYC-driven inflammation"),
    "FASN": ("Fatty Acid Metabolism", -1, "lipid synthesis / foamy macrophage"),
    "HSPA9": ("Oxidative Phosphorylation", -1, "mitochondrial chaperone"),
    "GSK3B": ("Wnt-beta Catenin Signaling", +1, "GSK3beta degradation of beta-catenin"),
}

rows, details = [], {}
for tgt, (prog, sgn, why) in EXPECT.items():
    per = dep["cand"].get(tgt)
    if not per:
        continue
    preds = [r["profile"] for r in per.values() if "profile" in r]
    obss = [r["observed"] for r in per.values() if "observed" in r]
    mode = "predicted" if preds else "observed"
    if not preds and not obss:
        continue
    pv = np.mean(preds, 0) if preds else np.mean(obss, 0)
    ov = np.mean(obss, 0) if obss else None
    ags = [r.get("agree", 0.0) for r in per.values() if "profile" in r]
    nsrc = max(len(r["measured_in"]) for r in per.values())
    top = np.argsort(-np.abs(pv))[:3]
    in_top3 = prog in [progs[i] for i in top]
    z = float(pv[pidx[prog]])
    sign_ok = bool(in_top3 and np.sign(z) == sgn)
    rows.append({
        "target": tgt, "expected": prog, "why": why, "n_sources": nsrc, "mode": mode,
        "source_agreement": float(np.mean(ags)) if ags else 0.0,
        "pred_z": z, "in_top3": bool(in_top3), "recovered": sign_ok,
        "top3": "; ".join(f"{progs[i]} {pv[i]:+.1f}" for i in top),
        "top3_names": [progs[i] for i in top],
        "d_TB_vs_HC": float(tb.loc[prog, "d_TB_vs_HC"]),
        "q_TB_vs_HC": float(tb.loc[prog, "q_TB_vs_HC"]),
        "d_DR_vs_DS": float(tb.loc[prog, "d_DR_vs_DS"]),
        "q_DR_vs_DS": float(tb.loc[prog, "q_DR_vs_DS"]),
    })
    details[tgt] = {"pred": np.round(pv, 3).tolist(), "obs": None if ov is None else np.round(ov, 3).tolist()}

C = pd.DataFrame(rows)
C.to_csv(RUN + "kt112_clinical_chain.csv", index=False)
pd.set_option("display.width", 200)
print(C[["target", "expected", "mode", "n_sources", "pred_z", "in_top3", "recovered",
         "d_TB_vs_HC", "d_DR_vs_DS", "q_DR_vs_DS"]].to_string(index=False))

# --- are the model-proposed programs more TB-dysregulated than the other programs?
rng = np.random.RandomState(0)
prop = list(dict.fromkeys([p for _, r in C.iterrows() for p in r.top3_names]))
prop = [p for p in prop if p in tb.index]
others = [p for p in tb.index if p not in prop]
for col in ("d_TB_vs_HC", "d_DR_vs_DS"):
    a = np.abs(tb.loc[prop, col].values)
    b = np.abs(tb.loc[others, col].values)
    null = np.array([np.abs(tb.loc[rng.choice(list(tb.index), len(prop), replace=False), col].values).mean()
                     for _ in range(5000)])
    print(f"{col}: proposed |d| mean {a.mean():.2f} (n={len(a)}) vs all-programs {np.abs(tb[col]).mean():.2f}, "
          f"perm p={np.mean(null >= a.mean()):.3f}")

k, N = int(C.recovered.sum()), len(C)
pbin = stats.binomtest(k, N, 3 / 45, alternative="greater").pvalue
print(f"\nrecovered {k}/{N}; chance 3/45 = {3/45:.3f}; binomial p = {pbin:.4f}")

# how often does the expected program at least appear in the top-5 by |z|?
top5 = 0
for tgt, (prog, sgn, _) in EXPECT.items():
    per = dep["cand"].get(tgt)
    if not per:
        continue
    pr = [r["profile"] for r in per.values() if "profile" in r]
    if not pr:
        continue
    pv = np.mean(pr, 0)
    t5 = [progs[i] for i in np.argsort(-np.abs(pv))[:5]]
    top5 += prog in t5
print(f"expected program in predicted top-5: {top5}/{N}")

json.dump({"n_testable": N, "recovered": k, "chance": 3 / 45, "binom_p": float(pbin),
           "rows": rows}, open(RUN + "kt112_clinical_chain.json", "w"), indent=1)
np.save(RUN + "kt112_profiles.npy", details, allow_pickle=True)
print("DONE")
