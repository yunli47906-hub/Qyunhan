"""kt109 (local): do the Hallmark programs used as the output space carry
real tuberculosis patient signal?

Input : P11 pseudobulk, 8,385 genes x 15 donors (HC/DS/DR, n=5 each).
Output: per-program DR/DS/HC scores, effect sizes, and an alignment test
        between the triage targets' predicted top-3 programs and the set of
        programs that are dysregulated in patients.

Program scoring: log2(CPM+1) -> gene-wise z across 15 donors -> mean z of
member genes (the Z-score method; no rank-based ssGSEA dependency).
"""
import json, numpy as np, pandas as pd
from scipy import stats
import pyreadr

R = pyreadr.read_r(r"C:/Users/1/Desktop/P11/L1_Outdata/p11_L1.pseudobulk_trusted.rds")[None]
R.index.name = "gene"
R = R[~R.index.duplicated()]
cnt = R.to_numpy(float)
cpm = cnt / cnt.sum(0, keepdims=True) * 1e6
L = np.log2(cpm + 1)
Z = (L - L.mean(1, keepdims=True)) / (L.std(1, keepdims=True) + 1e-9)
gidx = {g: i for i, g in enumerate(R.index)}

GS = json.load(open(r"C:/Users/1/Desktop/P16/runs/genesets_MSigDB_Hallmark_2020.json"))
progs = {}
for k, v in GS.items():
    ix = [gidx[g] for g in v if g in gidx]
    if len(ix) >= 10:
        progs[k.replace("HALLMARK_", "")] = ix
print(f"donors={R.shape[1]} genes={R.shape[0]} programs={len(progs)}")

S = pd.DataFrame({k: Z[ix].mean(0) for k, ix in progs.items()}, index=R.columns)
grp = np.array([c[:2] for c in R.columns])
print("groups:", {g: int((grp == g).sum()) for g in np.unique(grp)})


def d(a, b):
    return (a.mean() - b.mean()) / np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))


rows = []
for k in S.columns:
    tb = S.loc[grp != "HC", k]
    hc = S.loc[grp == "HC", k]
    dr = S.loc[grp == "DR", k]
    ds = S.loc[grp == "DS", k]
    p_tb = stats.mannwhitneyu(tb, hc, alternative="two-sided").pvalue
    p_dr = stats.mannwhitneyu(dr, ds, alternative="two-sided").pvalue
    rows.append({"program": k, "d_TB_vs_HC": d(tb.values, hc.values), "p_TB_vs_HC": p_tb,
                 "d_DR_vs_DS": d(dr.values, ds.values), "p_DR_vs_DS": p_dr,
                 "mean_HC": hc.mean(), "mean_DS": ds.mean(), "mean_DR": dr.mean()})
T = pd.DataFrame(rows)
T["q_TB_vs_HC"] = np.minimum(1, T.p_TB_vs_HC * len(T) / (np.arange(len(T)) + 1)[::-1][np.argsort(T.p_TB_vs_HC.values)] * 0 + T.p_TB_vs_HC * len(T))
# BH
from statsmodels.stats.multitest import multipletests
T["q_TB_vs_HC"] = multipletests(T.p_TB_vs_HC, method="fdr_bh")[1]
T["q_DR_vs_DS"] = multipletests(T.p_DR_vs_DS, method="fdr_bh")[1]
T = T.sort_values("d_TB_vs_HC")
T.to_csv(r"C:/Users/1/Desktop/P16/runs/kt109_tb_programs.csv", index=False)
S.T.to_csv(r"C:/Users/1/Desktop/P16/runs/kt109_tb_scores.csv")

print("\nTop TB-up programs (TB vs HC):")
for _, r in T.tail(8).iloc[::-1].iterrows():
    print(f"  {r.program:38s} d={r.d_TB_vs_HC:+.2f} p={r.p_TB_vs_HC:.3f} q={r.q_TB_vs_HC:.3f}")
print("\nTop TB-down programs:")
for _, r in T.head(8).iterrows():
    print(f"  {r.program:38s} d={r.d_TB_vs_HC:+.2f} p={r.p_TB_vs_HC:.3f} q={r.q_TB_vs_HC:.3f}")
print("\nDR vs DS:")
for _, r in T.reindex(T.d_DR_vs_DS.abs().sort_values(ascending=False).index).head(6).iterrows():
    print(f"  {r.program:38s} d={r.d_DR_vs_DS:+.2f} p={r.p_DR_vs_DS:.3f} q={r.q_DR_vs_DS:.3f}")

# ---- alignment: predicted top-3 programs of accepted triage targets vs patient dysregulation
hdt = json.load(open(r"C:/Users/1/Desktop/P16/runs/kt108_hdt.json"))
ptop = {}
for name, rows_ in hdt["phase2"].items():
    for r in rows_:
        ptop.setdefault(name, []).extend([x["program"].replace("HALLMARK_", "") for x in r["top_programs"][:3]])

dys = set(T.loc[T.q_TB_vs_HC < 0.25, "program"])
print(f"\nprograms dysregulated in patients (BH q<0.25): {len(dys)}/{len(T)}")

rng = np.random.RandomState(0)
allp = list(T.program)
res = {}
for name, tp in ptop.items():
    hit = sum(p in dys for p in tp)
    obs = hit / len(tp)
    null = np.array([np.mean([rng.choice(allp) in dys for _ in tp]) for _ in range(4000)])
    res[name] = {"top3": tp, "observed": obs, "null_mean": float(null.mean()),
                 "p": float((null >= obs).mean())}
    print(f"  {name:8s} {obs:.2f} vs null {null.mean():.2f}  p={res[name]['p']:.3f}  {tp}")

json.dump({"n_donors": int(R.shape[1]), "programs": len(T),
           "dysregulated_q25": sorted(dys), "alignment": res},
          open(r"C:/Users/1/Desktop/P16/runs/kt109_tb_alignment.json", "w"), indent=1)
print("DONE")
