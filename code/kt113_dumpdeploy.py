"""kt113 (GPU): dump raw deployment profiles (kt108 Phase 2) including the
full 45-program z-profile per predicted line, which kt108's summary dropped."""
import json, numpy as np, torch


class _CatCol:
    def __getitem__(self, xs):
        return torch.cat(xs, 1)


torch.c_ = _CatCol()
dev = "cuda"
torch.manual_seed(0)

CV = "cov"
m = json.load(open(f"{CV}/meta.json"))
genes = m["genes"]
if isinstance(genes, dict):
    genes = [g for g, _ in sorted(genes.items(), key=lambda kv: kv[1])]
genes = list(genes)
names = [None] * len(m["context_to_id"])
for k, v in m["context_to_id"].items():
    names[v] = k
names = sorted(names)
pn = [None] * len(m["pert_to_id"])
for k, v in m["pert_to_id"].items():
    pn[v] = k
ctx = np.load(f"{CV}/ctx_ids.npy").ravel()
pid = np.load(f"{CV}/pert_ids.npy").ravel().astype(int)
Dm = np.load(f"{CV}/delta_p.npy", mmap_mode="r")
obs = {c: {int(pid[r]): r for r in np.where(ctx == i)[0]} for i, c in enumerate(names)}
keep = sorted(set.intersection(*[set(obs[c]) for c in names]))
n = len(keep)
X = {c: torch.tensor(np.asarray(Dm[[obs[c][q] for q in keep]], np.float32), device=dev) for c in names}

GS = json.load(open("genesets_MSigDB_Hallmark_2020.json"))
gi = {g: i for i, g in enumerate(genes)}
pnames, idxs = [], []
for k, v in GS.items():
    ix = np.array(sorted({gi[g] for g in v if g in gi}))
    if len(ix) >= 10:
        pnames.append(k.replace("HALLMARK_", ""))
        idxs.append(ix)

CAND = ["HSPA9", "MTOR", "ATF4", "MFN2", "HSPA8", "TFRC", "MYC", "PTEN", "HMGCR", "FASN",
        "DDIT3", "BAX", "NFE2L2", "SIRT1", "NFKB1", "HIF1A", "AKT1", "XBP1", "ATF6",
        "CASP3", "MAP1LC3B", "SQSTM1", "GPX4", "SLC7A11", "STAT1", "IRF1", "BECN1",
        "ATG5", "ATG7", "SLC2A1", "LDHA", "PKM", "RB1", "MAPK1", "MAPK3", "PIK3CA",
        "RICTOR", "FOXO3", "SREBF1", "LDLR", "ACACA", "UCP2", "KEAP1", "TSC1", "TSC2",
        "RPTOR", "EIF2AK3", "ERN1", "EIF2AK1", "PPP1R15A", "GCLC", "GCLM", "TXNRD1",
        "NQO1", "HMOX1", "PTGS2", "ALOX5", "VDR", "NR1H4", "PPARG", "PPARA", "SIRT3",
        "FOXO1", "CTNNB1", "GSK3B", "TNFAIP3", "IL10", "IFNG", "TNF", "IL1B", "NLRP3",
        "CARD9", "TLR2", "TLR4", "MYD88", "IRAK4", "TICAM1", "RELA", "NFKBIA", "CHUK"]
CAND = list(dict.fromkeys(CAND))
pidx = {p: i for i, p in enumerate(pn)}

out = {}
hit_list = []
for tgt in names:
    srcs = [c for c in names if c != tgt]
    mu = {c: X[c].mean(0) for c in names}
    Xc = {c: X[c] - mu[c] for c in names}
    Sm = torch.cat([Xc[c] for c in names], 1)
    U, sv, _ = torch.linalg.svd(Sm, full_matrices=False)
    beta = min(*Sm.shape) / max(*Sm.shape)
    om = 0.56 * beta ** 3 - 0.95 * beta ** 2 + 1.82 * beta + 1.43
    r = max(2, min(int((sv > om * sv.median()).sum()), n - 1))
    Z = U[:, :r]
    B = {c: Z.T @ Xc[c] for c in names}
    Pinv = {c: torch.linalg.pinv(B[c].T.double()).float() for c in srcs}
    tr_raw = np.stack([Xc[tgt].cpu().numpy()[:, ix].mean(1) for ix in idxs], 1)
    tr_mu, tr_sd = tr_raw.mean(0), tr_raw.std(0) + 1e-8
    for name in CAND:
        qi = pidx.get(name)
        if qi is None:
            continue
        have = [c for c in srcs if qi in obs[c]]
        if not have:
            continue
        rec = {"measured_in": have, "r": int(r)}
        if qi in obs[tgt]:
            xo = torch.tensor(np.asarray(Dm[obs[tgt][qi]], np.float32), device=dev) - mu[tgt]
            ro = xo.cpu().numpy()
            rec["observed"] = ((np.array([ro[ix].mean() for ix in idxs]) - tr_mu) / tr_sd).tolist()
        else:
            zs = []
            for c in have:
                xq = torch.tensor(np.asarray(Dm[obs[c][qi]], np.float32), device=dev) - mu[c]
                zs.append(Pinv[c] @ xq)
            zq = torch.stack(zs).mean(0)
            dp = (zq @ B[tgt]).cpu().numpy()
            raw = np.array([dp[ix].mean() for ix in idxs])
            rec["profile"] = ((raw - tr_mu) / tr_sd).tolist()
            if len(zs) > 1:
                Zc = torch.stack(zs)
                nz = Zc / Zc.norm(dim=1, keepdim=True).clamp_min(1e-8)
                rec["agree"] = float((nz @ nz.T).triu(1).sum() / max(1, len(zs) * (len(zs) - 1) / 2))
            else:
                rec["agree"] = 0.0
        out.setdefault(name, {})[tgt] = rec
    hit_list.append(tgt)

json.dump({"programs": pnames, "n": n, "targets": hit_list, "cand": out},
          open("runs/kt113_deploy.json", "w"))
print("targets:", len(hit_list), "candidates:", len(out))
print("DONE")
