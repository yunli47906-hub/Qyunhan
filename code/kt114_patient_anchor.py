"""kt114 (local): make the patient anchor defensible.

Two things the raw programme table does not tell us:

  (1) Is there a global shift?  Every programme is lower in TB than in HC and
      higher in DR than in DS.  If the direction is shared across programmes,
      the per-programme effect sizes are partly measuring sample-level scale
      (RNA content / composition), not programme-specific biology.

  (2) Is |d| tracking expression level?  Low-expression programmes have
      inflated variance in log2(CPM+1) space, so they can manufacture large
      standardised effects.

We quantify both, then report the well-expressed subset only.
"""
import json, numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import pyreadr

R = pyreadr.read_r(r"C:/Users/1/Desktop/P11/L1_Outdata/p11_L1.pseudobulk_trusted.rds")[None]
R = R[~R.index.duplicated()]
raw = R.to_numpy(float)
cpm = raw / raw.sum(0, keepdims=True) * 1e6
L = np.log2(cpm + 1)
Z = (L - L.mean(1, keepdims=True)) / (L.std(1, keepdims=True) + 1e-9)
gidx = {g: i for i, g in enumerate(R.index)}
grp = np.array([c[:2] for c in R.columns])

GS = json.load(open(r"C:/Users/1/Desktop/P16/runs/genesets_MSigDB_Hallmark_2020.json"))
progs = {}
for k, v in GS.items():
    ix = [gidx[g] for g in v if g in gidx]
    if len(ix) >= 10:
        progs[k.replace("HALLMARK_", "")] = ix

T = pd.read_csv(r"C:/Users/1/Desktop/P16/runs/kt109_tb_programs.csv")

# ---- (1) global shift ------------------------------------------------------
dTB = T.d_TB_vs_HC.values
dDR = T.d_DR_vs_DS.values
rng = np.random.RandomState(0)
S = pd.DataFrame({k: Z[ix].mean(0) for k, ix in progs.items()}, index=R.columns)


def mean_d_draw(labels):
    tb = labels != "HC"
    hc = labels == "HC"
    dr = labels == "DR"
    ds = labels == "DS"
    out_tb, out_dr = [], []
    for k in S.columns:
        a, b = S.loc[tb, k].values, S.loc[hc, k].values
        out_tb.append((a.mean() - b.mean()) / np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)))
        a, b = S.loc[dr, k].values, S.loc[ds, k].values
        out_dr.append((a.mean() - b.mean()) / np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)))
    return np.mean(out_tb), np.mean(out_dr)


lab = grp.copy()
null_tb, null_dr = [], []
for _ in range(5000):
    rng.shuffle(lab)
    m1, m2 = mean_d_draw(lab)
    null_tb.append(m1)
    null_dr.append(m2)
null_tb, null_dr = np.array(null_tb), np.array(null_dr)
glob = {"mean_d_TB": float(dTB.mean()), "frac_negative_TB": float((dTB < 0).mean()),
        "perm_p_TB": float(np.mean(null_tb <= dTB.mean())),
        "mean_d_DR": float(dDR.mean()), "frac_positive_DR": float((dDR > 0).mean()),
        "perm_p_DR": float(np.mean(null_dr >= dDR.mean())),
        "null_sd_TB": float(null_tb.std()), "null_sd_DR": float(null_dr.std())}
print("GLOBAL SHIFT", json.dumps(glob, indent=1))

# ---- (2) expression-level confound ----------------------------------------
T["logexpr"] = [float(L[progs[p]].mean()) for p in T.program]
rho_tb = stats.spearmanr(T.logexpr, T.d_TB_vs_HC.abs())
rho_dr = stats.spearmanr(T.logexpr, T.d_DR_vs_DS.abs())
print(f"|d| vs expression: TB rho={rho_tb.statistic:+.3f} p={rho_tb.pvalue:.3f}; "
      f"DR rho={rho_dr.statistic:+.3f} p={rho_dr.pvalue:.3f}")

# well-expressed: programme mean log2(CPM+1) above the median of all programmes
thr = float(T.logexpr.median())
well = T[T.logexpr >= thr].copy()
well["q_TB"] = multipletests(well.p_TB_vs_HC, method="fdr_bh")[1]
well["q_DR"] = multipletests(well.p_DR_vs_DS, method="fdr_bh")[1]
print(f"\nwell-expressed programmes (mean log2CPM >= {thr:.2f}): {len(well)}/{len(T)}")
cols = ["program", "logexpr", "d_TB_vs_HC", "p_TB_vs_HC", "q_TB", "d_DR_vs_DS", "p_DR_vs_DS", "q_DR"]
print(well.sort_values("d_TB_vs_HC")[cols].to_string(index=False))

# repeat the expression-confound test inside the well-expressed subset
r2t = stats.spearmanr(well.logexpr, well.d_TB_vs_HC.abs())
r2d = stats.spearmanr(well.logexpr, well.d_DR_vs_DS.abs())
print(f"\nwithin well-expressed: |d| vs expression TB rho={r2t.statistic:+.3f} p={r2t.pvalue:.3f}; "
      f"DR rho={r2d.statistic:+.3f} p={r2d.pvalue:.3f}")

# ---- model-proposed programmes are more dysregulated? ---------------------
C = pd.read_csv(r"C:/Users/1/Desktop/P16/runs/kt112_clinical_chain.csv")
prop = list(dict.fromkeys([p for s in C.top3_names for p in eval(s)]))
prop = [p for p in prop if p in T.program.values]
others = [p for p in T.program if p not in prop]
perm = {}
for col in ("d_TB_vs_HC", "d_DR_vs_DS"):
    a = T.set_index("program").loc[prop, col].abs().values
    b = T.set_index("program").loc[others, col].abs().values
    rng2 = np.random.RandomState(0)
    idx = np.arange(len(T))
    null = np.array([T[col].abs().values[rng2.choice(idx, len(prop), replace=False)].mean() for _ in range(5000)])
    perm[col] = {"proposed_mean_absd": float(a.mean()), "others": float(b.mean()),
                 "all": float(T[col].abs().mean()), "p": float((null >= a.mean()).mean())}
print("\nMODEL-PROPOSED PROGRAMMES", json.dumps(perm, indent=1))

json.dump({"global_shift": glob,
           "expr_confound": {"TB_rho": float(rho_tb.statistic), "TB_p": float(rho_tb.pvalue),
                             "DR_rho": float(rho_dr.statistic), "DR_p": float(rho_dr.pvalue)},
           "well_expressed_threshold": thr, "n_well": int(len(well)),
           "well_rows": well[cols].round(4).to_dict(orient="records"),
           "well_expr_confound": {"TB_rho": float(r2t.statistic), "TB_p": float(r2t.pvalue),
                                  "DR_rho": float(r2d.statistic), "DR_p": float(r2d.pvalue)},
           "model_proposed": perm,
           "n_proposed": len(prop), "proposed": prop},
          open(r"C:/Users/1/Desktop/P16/runs/kt114_patient_anchor.json", "w"), indent=1)
well[cols].to_csv(r"C:/Users/1/Desktop/P16/runs/kt114_well_expressed.csv", index=False)
print("DONE")
