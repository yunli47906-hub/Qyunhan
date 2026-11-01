"""kt111 (GPU): dump per-perturbation program-level predictions and truth.

Reproduces the kt108 Phase-1 geometry exactly (5 reps x 5 folds, per target
line) but writes every held-out record, so downstream figures and clinical
tables can use per-perturbation values instead of aggregates.
"""
import json, time, numpy as np, torch


class _CatCol:
    def __getitem__(self, xs):
        return torch.cat(xs, 1)


torch.c_ = _CatCol()
dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True
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
Xnp = {c: X[c].cpu().numpy() for c in names}

GS = json.load(open("genesets_MSigDB_Hallmark_2020.json"))
gi = {g: i for i, g in enumerate(genes)}
pnames, idxs = [], []
for k, v in GS.items():
    ix = np.array(sorted({gi[g] for g in v if g in gi}))
    if len(ix) >= 10:
        pnames.append(k.replace("HALLMARK_", ""))
        idxs.append(ix)
K = len(idxs)


def scores_np(M):
    S = np.stack([M[:, ix].mean(1) for ix in idxs], 1)
    return (S - S.mean(0)) / (S.std(0) + 1e-8)


S = {c: torch.tensor(scores_np(Xnp[c]), device=dev) for c in names}


def gd(sv, nn_, p, cap=256):
    beta = min(nn_, p) / max(nn_, p)
    om = 0.56 * beta ** 3 - 0.95 * beta ** 2 + 1.82 * beta + 1.43
    return max(2, min(int((sv > om * sv.median()).sum()), cap))


def rowcorr_t(P, Y):
    Pc = P - P.mean(1, keepdim=True)
    Yc = Y - Y.mean(1, keepdim=True)
    return (Pc * Yc).sum(1) / ((Pc.pow(2).sum(1) * Yc.pow(2).sum(1)).sqrt().clamp_min(1e-12))


def feats(idx, p, g, srcs):
    srcs_ = torch.stack([S[c][idx] for c in srcs])
    mu_, sd_ = srcs_.mean(0), srcs_.std(0)
    nm = p.pow(2).sum(1).sqrt()
    cv = sd_.pow(2).sum(1).sqrt() / mu_.pow(2).sum(1).sqrt().clamp_min(1e-8)
    ag = (p - g).pow(2).sum(1).sqrt() / (nm + g.pow(2).sum(1).sqrt()).clamp_min(1e-8)
    return torch.stack([nm.log1p(), cv, ag, p.abs().mean(1), (p.abs() > 1).float().mean(1),
                        p.abs().max(1).values, sd_.pow(2).sum(1).sqrt(),
                        (mu_ - p).pow(2).sum(1).sqrt().log1p()], 1)


NREP, NFOLD = 5, 5
REC = {q: [] for q in range(n)}
t0 = time.time()
for tgt in names:
    srcs = [c for c in names if c != tgt]
    for rep in range(NREP):
        g = torch.Generator(device=dev)
        g.manual_seed(13000 + rep)
        pm = torch.randperm(n, generator=g, device=dev)
        folds = [pm[i::NFOLD] for i in range(NFOLD)]
        for kf in range(NFOLD):
            te = folds[kf]
            tr = torch.cat([folds[j] for j in range(NFOLD) if j != kf])
            pcx = torch.randperm(tr.numel(), generator=g, device=dev)
            nA = int(round(.6 * tr.numel()))
            A, Bs_ = tr[pcx[:nA]], tr[pcx[nA:]]
            yte, ycal = S[tgt][te], S[tgt][Bs_]

            mus = {c: X[c][A].mean(0) for c in names}
            XcA = {c: X[c][A] - mus[c] for c in names}
            XcB = {c: X[c][Bs_] - mus[c] for c in names}
            XcE = {c: X[c][te] - mus[c] for c in names}
            Sm = torch.cat([XcA[c] for c in names], 1)
            U, sv, _ = torch.linalg.svd(Sm, full_matrices=False)
            r = max(2, min(gd(sv, *Sm.shape), nA - 1))
            Z = U[:, :r]
            B = {c: Z.T @ XcA[c] for c in names}
            Bt = B[tgt]
            zevA = sum((XcA[c] @ B[c].T) for c in srcs) / len(srcs)
            zevB = sum((XcB[c] @ B[c].T) for c in srcs) / len(srcs)
            zevE = sum((XcE[c] @ B[c].T) for c in srcs) / len(srcs)
            WA = torch.linalg.pinv(torch.c_[torch.ones(len(A), 1, device=dev), zevA].double()) @ (XcA[tgt] @ Bt.T).double()
            dcal = ((torch.c_[torch.ones(len(Bs_), 1, device=dev), zevB].double() @ WA) @ Bt.double()).float()
            dte = ((torch.c_[torch.ones(len(te), 1, device=dev), zevE].double() @ WA) @ Bt.double()).float()

            F = torch.cat([S[c][A] for c in srcs], 1)
            W = torch.linalg.solve(F.double().T @ F.double() + 10.0 * torch.eye(F.shape[1], dtype=torch.float64, device=dev),
                                   F.double().T @ S[tgt][A].double()).float()
            pp_cal = torch.cat([S[c][Bs_] for c in srcs], 1) @ W
            pp_te = torch.cat([S[c][te] for c in srcs], 1) @ W
            gp_cal = torch.tensor(scores_np(dcal.cpu().numpy()), device=dev)
            gp_te = torch.tensor(scores_np(dte.cpu().numpy()), device=dev)

            FC = feats(Bs_, pp_cal, gp_cal, srcs)
            FE = feats(te, pp_te, gp_te, srcs)
            fmu, fsd = FC.mean(0), FC.std(0) + 1e-8
            FCs = torch.c_[(FC - fmu) / fsd, torch.ones(len(FC), 1, device=dev)]
            FEs = torch.c_[(FE - fmu) / fsd, torch.ones(len(FE), 1, device=dev)]
            yt = rowcorr_t(pp_cal, ycal)
            w = torch.linalg.solve(FCs.double().T @ FCs.double() + 1e-2 * torch.eye(FCs.shape[1], dtype=torch.float64, device=dev),
                                   FCs.double().T @ yt.double()).float()
            rel = (FEs @ w).cpu().numpy()
            P = pp_te.cpu().numpy()
            Y = yte.cpu().numpy()
            for j, qi in enumerate(te.cpu().numpy()):
                REC[int(qi)].append((tgt, P[j], float(rel[j]), Y[j]))
    print(f"phase1 {tgt} {time.time()-t0:.0f}s", flush=True)

Q = [q for q in range(n) if len(REC[q]) == len(names) * NREP]
pred = np.zeros((len(Q), K), np.float32)
truth = np.zeros((len(Q), K), np.float32)
rel = np.zeros(len(Q), np.float32)
perline = {c: np.zeros((len(Q), K), np.float32) for c in names}
for i, q in enumerate(Q):
    byline = {c: [] for c in names}
    for (c, p, rl, y) in REC[q]:
        byline[c].append((p, rl, y))
    pl = {c: np.mean([x[0] for x in byline[c]], 0) for c in names}
    pred[i] = np.mean([pl[c] for c in names], 0)
    truth[i] = np.mean([np.mean([x[2] for x in byline[c]], 0) for c in names], 0)
    rel[i] = float(np.mean([np.mean([x[1] for x in byline[c]]) for c in names]))
    for c in names:
        perline[c][i] = pl[c]

# observed per-line program scores (ground truth, no CV), for all 638
obsS = {c: scores_np(Xnp[c]) for c in names}
json.dump({"programs": pnames, "pert_names": [pn[q] for q in Q],
           "pred": np.round(pred, 4).tolist(), "truth": np.round(truth, 4).tolist(),
           "rel": np.round(rel, 4).tolist(),
           "obs_per_line": {c: np.round(obsS[c], 4).tolist() for c in names}},
          open("runs/kt111_dump.json", "w"))
print(f"dumped {len(Q)} perturbations -> runs/kt111_dump.json", flush=True)
print("DONE", flush=True)
