"""kt117 (GPU): sequence-only zero-shot prediction of the 45-programme profile.

Question: for a gene that was never perturbed in any of the four lines, does
protein sequence (ESM2) carry the measured Hallmark programme signature?

Target  Y[i] = kt111 truth (line-averaged, z-scored programme score).
Feature X[i] = ESM2 mean-pooled last-layer embedding of the canonical UniProt
               sequence.  Nothing else -- no expression, no network, no prior.

Evaluation is leave-gene-out (5 folds x 3 repeats): the held-out gene is never
in training, which is the honest proxy for a TB candidate absent from the
compendium.  Comparators on the same rows:
  * zero profile            (chance)
  * 1-nearest-neighbour in embedding space
  * frozen LLM prior vector (kt116)
  * VERDICT measured model  (kt111 pred; upper reference, uses measured lines)

Abstention: split-conformal on the relative error e = ||y-yhat||/||y||, with a
difficulty score from the embedding geometry, so unseen genes get a calibrated
interval alongside the point estimate.
"""
import json, os, time, numpy as np, torch
from transformers import EsmModel, EsmTokenizer

dev = "cuda"
torch.set_grad_enabled(False)
t0 = time.time()
R = "runs/"

SEQ = json.load(open("seq/gene_seq.json"))
D = json.load(open(R + "kt111_dump.json"))
progs, names = D["programs"], D["pert_names"]
Yall = np.array(D["truth"], float)
Pverdict = np.array(D["pred"], float)
K = Yall.shape[1]
pidx = {p: i for i, p in enumerate(progs)}

G = [g for g in names if g in SEQ]
gi = {g: i for i, g in enumerate(G)}
Y = np.array([Yall[names.index(g)] for g in G])
PV = np.array([Pverdict[names.index(g)] for g in G])
n = len(G)
print(f"{n}/{len(names)} compendium genes have a sequence", flush=True)


# ---- 1. ESM2 embeddings ---------------------------------------------------
MDIR = "models/esm2_t30_150M_UR50D"
tok = EsmTokenizer.from_pretrained(MDIR, local_files_only=True)
model = EsmModel.from_pretrained(MDIR, local_files_only=True).to(dev).eval()
MAXLEN = 1022
emb = np.zeros((n, model.config.hidden_size), np.float32)
embmax = np.zeros_like(emb)
order = sorted(range(n), key=lambda i: len(SEQ[G[i]]["seq"]))
B = 8
for s in range(0, n, B):
    idx = order[s:s + B]
    seqs = [SEQ[G[i]]["seq"][:MAXLEN] for i in idx]
    enc = tok(seqs, return_tensors="pt", padding=True, truncation=True).to(dev)
    out = model(**enc).last_hidden_state
    mask = enc["attention_mask"].bool()
    for j, i in enumerate(idx):
        m = mask[j]
        v = out[j][m]                      # includes special tokens
        emb[i] = v[1:-1].mean(0).float().cpu().numpy()
        embmax[i] = v.max(0).values.float().cpu().numpy()
    if s % 80 == 0:
        print(f"  embed {s}/{n} {time.time()-t0:.0f}s", flush=True)
del model
torch.cuda.empty_cache()
np.save(R + "kt117_esm_mean.npy", emb)
np.save(R + "kt117_esm_max.npy", embmax)
print("embeddings", emb.shape, f"{time.time()-t0:.0f}s", flush=True)

X = np.c_[emb, embmax]
X = (X - X.mean(0)) / (X.std(0) + 1e-8)
Xd = torch.tensor(X, device=dev)

PRI = json.load(open(R + "kt116_llm_priors.json"))["priors"]
U = np.zeros((n, K))
for g in G:
    for p, s in PRI.get(g, []):
        U[gi[g], pidx[p]] = s

Yt = torch.tensor(Y, device=dev)


def rowcorr(A, Bm):
    A = A - A.mean(-1, keepdim=True)
    Bm = Bm - Bm.mean(-1, keepdim=True)
    return (A * Bm).sum(-1) / (A.norm(dim=-1) * Bm.norm(dim=-1)).clamp_min(1e-9)


def topk_hit(P, Yq, k=3):
    pt = torch.topk(P.abs(), k, 1).indices
    yt = torch.topk(Yq.abs(), k, 1).indices
    return torch.tensor([len(set(a.tolist()) & set(b.tolist())) for a, b in zip(pt, yt)])


ALPHAS = [1e-1, 1e0, 10.0, 1e2, 1e3, 1e4]
NREP, NFOLD = 3, 5


def fit_ridge(Xtr, Ytr, Xte, alpha):
    mu = Xtr.mean(0)
    A = (Xtr - mu).double()
    Bx = (Xte - mu).double()
    W = torch.linalg.solve(A.T @ A + alpha * torch.eye(A.shape[1], device=dev, dtype=A.dtype),
                           A.T @ Ytr.double())
    return (Bx @ W).float()


oof = np.zeros((NREP, n, K), np.float32)
oof_knn = np.zeros((n, K), np.float32)
knn_done = False
for rep in range(NREP):
    g = torch.Generator().manual_seed(7700 + rep)
    pm = torch.randperm(n, generator=g)
    for f in range(NFOLD):
        te = pm[f::NFOLD]
        tr = torch.tensor([i for i in range(n) if i not in set(te.tolist())], device=dev)
        trn = tr[torch.randperm(len(tr), generator=g)]
        nA = int(round(.75 * len(trn)))
        A_, Bv = trn[:nA], trn[nA:]
        # inner alpha selection on a held-out part of the training genes
        best, ba = -9.0, None
        for al in ALPHAS:
            Pb = fit_ridge(Xd[A_], Yt[A_], Xd[Bv], al)
            sc = float(rowcorr(Pb, Yt[Bv]).mean())
            if sc > best:
                best, ba = sc, al
        p = fit_ridge(Xd[tr], Yt[tr], Xd[te], ba)
        oof[rep, te.cpu().numpy()] = p.cpu().numpy()
    print(f"rep {rep} done {time.time()-t0:.0f}s", flush=True)

# 1-NN in raw embedding space (cosine), leave-one-out
E = torch.tensor(emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9), device=dev)
for i in range(n):
    sim = E @ E[i]
    sim[i] = -2
    j = int(sim.argmax())
    oof_knn[i] = Y[j]

Oof = oof.mean(0)


def metrics(P, tag):
    rc = rowcorr(torch.tensor(P, device=dev), Yt).cpu().numpy()
    h3 = topk_hit(torch.tensor(P, device=dev), Yt, 3).numpy()
    h1 = topk_hit(torch.tensor(P, device=dev), Yt, 1).numpy()
    h5 = topk_hit(torch.tensor(P, device=dev), Yt, 5).numpy()
    d = {
        "tag": tag, "n": int(n),
        "rowcorr_mean": float(rc.mean()), "rowcorr_median": float(np.median(rc)),
        "hit3_ge1": float((h3 >= 1).mean()), "hit3_mean": float(h3.mean()),
        "hit3_exact": float((h3 == 3).mean()),
        "hit1_exact": float((h1 == 1).mean()), "hit5_ge1": float((h5 >= 1).mean()),
        "rowcorr_gt_0.2": float((rc > .2).mean()),
        "rowcorr_gt_0.4": float((rc > .4).mean()),
    }
    print(json.dumps(d, indent=1), flush=True)
    return d, rc


S_seq, rc_seq = metrics(Oof, "seq-ridge")
S_knn, rc_knn = metrics(oof_knn, "seq-1nn")
S_zero, rc_zero = metrics(np.zeros_like(Y), "zero")
S_llm, rc_llm = metrics(U, "llm")
S_v, rc_v = metrics(PV, "verdict")
S_rep = metrics(oof[0], "seq-rep0")
per_rep = [float(rowcorr(torch.tensor(oof[r], device=dev), Yt).mean()) for r in range(NREP)]
print("rowcorr per repeat", [round(x, 4) for x in per_rep], flush=True)

# paired bootstrap of seq - llm
rng = np.random.RandomState(0)
d_ = rc_seq - rc_llm
bs = np.array([d_[rng.randint(0, n, n)].mean() for _ in range(10000)])
boot = {"delta_rowcorr_seq_minus_llm": float(d_.mean()),
        "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
        "p_le0": float(np.mean(bs <= 0))}
print(json.dumps(boot, indent=1), flush=True)

# ---- 2. conformal abstention on relative error ---------------------------
def rel_err(P, T):
    return (np.linalg.norm(T - P, axis=1) / np.linalg.norm(T, axis=1).clip(1e-9))


# difficulty: cosine distance to nearest training genes + predicted norm
dist_all = 1 - (E.cpu().numpy() @ E.cpu().numpy().T)
np.fill_diagonal(dist_all, np.inf)
hard = np.c_[np.sort(dist_all, 1)[:, :5].mean(1), np.linalg.norm(Oof, axis=1)]
hard = np.c_[hard, hard[:, 0] * hard[:, 1]]
Hc = (hard - hard.mean(0)) / (hard.std(0) + 1e-8)
e = rel_err(Oof, Y)

cov_rows = []
for rep in range(NREP):
    g = np.random.RandomState(99 + rep)
    pm = g.permutation(n)
    for f in range(5):
        te = pm[f::5]
        cal = np.setdiff1d(pm, te)
        # fit difficulty model on calibration half, use the other half to score
        hc = np.c_[Hc[cal], np.ones(len(cal))]
        A = hc.T @ hc + 1e-2 * np.eye(hc.shape[1])
        w = np.linalg.solve(A, hc.T @ e[cal])
        ec = hc @ w
        q = np.quantile(ec, 0.9)
        sc = np.c_[Hc[te], np.ones(len(te))] @ w
        cov_rows.append({
            "covered": float((e[te] <= sc + q).mean()),
            "width": float((sc + q).mean()),
            "e_te": float(e[te].mean()),
        })
cov = {k: float(np.mean([c[k] for c in cov_rows])) for k in ("covered", "width", "e_te")}
print("CONFORMAL", json.dumps(cov, indent=1), flush=True)

# ---- 3. TB candidates ----------------------------------------------------
TB = ["DDIT3", "BAX", "NFE2L2", "SIRT1", "NFKB1", "HIF1A", "AKT1", "XBP1", "ATF6",
      "CASP3", "MAP1LC3B", "SQSTM1", "GPX4", "SLC7A11", "STAT1", "IRF1", "BECN1",
      "ATG5", "ATG7", "SLC2A1", "LDHA", "PKM", "RB1", "MAPK1", "MAPK3", "PIK3CA",
      "RICTOR", "FOXO3", "SREBF1", "LDLR", "ACACA", "UCP2", "TSC1", "TSC2",
      "RPTOR", "EIF2AK3", "ERN1", "EIF2AK1", "PPP1R15A", "GCLM", "TXNRD1",
      "NQO1", "HMOX1", "PTGS2", "ALOX5", "VDR", "NR1H4", "PPARG", "PPARA",
      "SIRT3", "FOXO1", "CTNNB1", "TNFAIP3", "IL10", "IFNG", "TNF", "IL1B",
      "NLRP3", "CARD9", "TLR2", "TLR4", "MYD88", "IRAK4", "TICAM1", "RELA",
      "NFKBIA", "CHUK"]
TBu = [g for g in dict.fromkeys(TB) if g in SEQ and g not in gi]

model = EsmModel.from_pretrained(MDIR, local_files_only=True).to(dev).eval()
newm = np.zeros((len(TBu), emb.shape[1]), np.float32)
newmax = np.zeros_like(newm)
for s in range(0, len(TBu), B):
    chunk = TBu[s:s + B]
    enc = tok([SEQ[g]["seq"][:MAXLEN] for g in chunk], return_tensors="pt",
              padding=True, truncation=True).to(dev)
    out = model(**enc).last_hidden_state
    mask = enc["attention_mask"].bool()
    for j in range(len(chunk)):
        v = out[j][mask[j]]
        newm[s + j] = v[1:-1].mean(0).float().cpu().numpy()
        newmax[s + j] = v.max(0).values.float().cpu().numpy()
Xn = np.c_[newm, newmax]
Xraw = np.c_[emb, embmax]
mu = Xraw.mean(0)
sd = Xraw.std(0) + 1e-8
Xtr_full = torch.tensor((Xraw - mu) / sd, device=dev)
Xn_s = torch.tensor((Xn - mu) / sd, device=dev)
best, ba = -9.0, None
g = torch.Generator().manual_seed(1)
pm = torch.randperm(n, generator=g)
te = pm[:n // 5]
tr = pm[n // 5:]
for al in ALPHAS:
    sc = float(rowcorr(fit_ridge(Xtr_full[tr], Yt[tr], Xtr_full[te], al), Yt[te]).mean())
    if sc > best:
        best, ba = sc, al
Pt = fit_ridge(Xtr_full, Yt, Xn_s, ba).cpu().numpy()

# per-candidate conformal width using the seq-ridge oof residuals
hc = np.c_[Hc, np.ones(n)]
w = np.linalg.solve(hc.T @ hc + 1e-2 * np.eye(hc.shape[1]), hc.T @ e)
ec = hc @ w
q90 = float(np.quantile(ec, .9))
En = newm / (np.linalg.norm(newm, axis=1, keepdims=True) + 1e-9)
dnew = 1 - (torch.tensor(En, device=dev) @ E.T).cpu().numpy()   # (n_TB, n_compendium)
hnew = np.c_[np.sort(dnew, 1)[:, :5].mean(1), np.linalg.norm(Pt, axis=1)]
hnew = np.c_[hnew, hnew[:, 0] * hnew[:, 1]]
hnew = (hnew - hard.mean(0)) / (hard.std(0) + 1e-8)
se = np.c_[hnew, np.ones(len(TBu))] @ w
cand = {}
for i, gname in enumerate(TBu):
    top = np.argsort(-np.abs(Pt[i]))[:5]
    cand[gname] = {
        "acc": SEQ[gname]["acc"],
        "top5": [{"programme": progs[j], "z": round(float(Pt[i, j]), 3)} for j in top],
        "mean_absz": round(float(np.abs(Pt[i]).mean()), 3),
        "rel_width": round(float(se[i] + q90), 3),
        "state": "prior" if se[i] + q90 < np.median(e) + 0.5 else "low-confidence",
    }
print("TB candidates", len(cand), flush=True)

out = {
    "protocol": {
        "features": "ESM2-t30-150M mean+max pooled last hidden state (1280-d)",
        "target": "kt111 line-averaged Hallmark programme z-profile",
        "cv": f"leave-gene-out, {NREP} reps x {NFOLD} folds",
        "alpha": ALPHAS, "n_genes": int(n),
        "excluded_no_sequence": sorted(set(names) - set(G)),
    },
    "metrics": {"seq_ridge": S_seq, "seq_1nn": S_knn, "zero": S_zero,
                "llm_prior": S_llm, "verdict": S_v},
    "per_repeat_rowcorr": per_rep,
    "bootstrap": boot,
    "conformal": cov,
    "candidates": cand,
    "per_gene": [{"gene": G[i], "rowcorr_seq": round(float(rc_seq[i]), 4),
                  "rowcorr_knn": round(float(rc_knn[i]), 4),
                  "rowcorr_llm": round(float(rc_llm[i]), 4),
                  "rel_err": round(float(e[i]), 4)} for i in range(n)],
}
json.dump(out, open(R + "kt117_zeroshot.json", "w"), indent=1)
np.save(R + "kt117_oof.npy", Oof)
print("DONE", time.time() - t0, flush=True)
