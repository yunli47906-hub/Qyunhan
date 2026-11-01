"""kt118 (GPU): decision head vs profile head for Fig 4d.

Same folds/seeds as kt108 (5 reps x 5 folds, within-fold A .6 / Bs_ .4, te).
Two heads fitted on the SAME per-fold calibration half:
  profile head : ridge on 8 shape features -> rowcorr(pp_cal, ycal)   [kt108 head]
  decision head: L2 logistic on the manuscript feature set (peak, top1-top2 margin,
                 top2-top3 margin, mean top3, mean|.|, sd|.|, profile norm, profile
                 head value) -> label 1 if predicted top3 overlaps true top3
Reported: hit3_ge1 curves over a 20-point coverage grid for decision / profile /
random, delivered accuracy c*h+(1-c), AUC against the realised miss, and the
riskiest-decile hit rate.
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
        pnames.append(k)
        idxs.append(ix)
K = len(idxs)
print(f"n={n} programs={K}", flush=True)


def scores_np(M):
    S = np.stack([M[:, ix].mean(1) for ix in idxs], 1)
    return (S - S.mean(0)) / (S.std(0) + 1e-8)


S = {c: torch.tensor(scores_np(X[c].cpu().numpy()), device=dev) for c in names}


def gd_from_sv(sv, nn_, p, cap=256):
    beta = min(nn_, p) / max(nn_, p)
    om = 0.56 * beta ** 3 - 0.95 * beta ** 2 + 1.82 * beta + 1.43
    return max(2, min(int((sv > om * sv.median()).sum()), cap))


def rowcorr_t(P, Y):
    Pc = P - P.mean(1, keepdim=True)
    Yc = Y - Y.mean(1, keepdim=True)
    return (Pc * Yc).sum(1) / ((Pc.pow(2).sum(1) * Yc.pow(2).sum(1)).sqrt().clamp_min(1e-12))


def feats_one(idx, p, g, srcs, Sp):
    srcs_ = torch.stack([Sp[c][idx] for c in srcs])
    mu_, sd_ = srcs_.mean(0), srcs_.std(0)
    nm = p.pow(2).sum(1).sqrt()
    cv = sd_.pow(2).sum(1).sqrt() / mu_.pow(2).sum(1).sqrt().clamp_min(1e-8)
    ag = (p - g).pow(2).sum(1).sqrt() / (nm + g.pow(2).sum(1).sqrt()).clamp_min(1e-8)
    return torch.stack([nm.log1p(), cv, ag, p.abs().mean(1), (p.abs() > 1).float().mean(1),
                        p.abs().max(1).values, sd_.pow(2).sum(1).sqrt(),
                        (mu_ - p).pow(2).sum(1).sqrt().log1p()], 1)


def shape_feats(P, rel):
    A = np.abs(P)
    srt = -np.sort(-A, 1)
    return np.c_[srt[:, 0], srt[:, 0] - srt[:, 1], srt[:, 1] - srt[:, 2],
                 srt[:, :3].mean(1), A.mean(1), A.std(1), np.linalg.norm(P, axis=1), rel]


def hit3_ge1(P, Y):
    out = np.zeros(len(P))
    for i in range(len(P)):
        pt = set(np.argsort(-np.abs(P[i]))[:3].tolist())
        yt = set(np.argsort(-np.abs(Y[i]))[:3].tolist())
        out[i] = float(len(pt & yt) >= 1)
    return out


def logit_fit(F, y, l2=1.0, it=25):
    F = np.c_[F, np.ones(len(F))]
    w = np.zeros(F.shape[1])
    for _ in range(it):
        z = F @ w
        p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        W = np.clip(p * (1 - p), 1e-6, None)
        H = F.T @ (W[:, None] * F) + l2 * np.eye(F.shape[1])
        g = F.T @ (y - p) - l2 * w
        w = w + np.linalg.solve(H, g)
    return w


def auc(score, label):
    o = np.argsort(score)
    r = np.empty(len(score))
    r[o] = np.arange(1, len(score) + 1)
    pos = label == 1
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


NREP, NFOLD = 5, 5
REC = {q: [] for q in range(n)}
t0 = time.time()
for tgt in names:
    srcs = [c for c in names if c != tgt]
    Sp = S
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
            yte, ycal = Sp[tgt][te], Sp[tgt][Bs_]

            mus = {c: X[c][A].mean(0) for c in names}
            XcA = {c: X[c][A] - mus[c] for c in names}
            XcB = {c: X[c][Bs_] - mus[c] for c in names}
            XcE = {c: X[c][te] - mus[c] for c in names}
            Sm = torch.cat([XcA[c] for c in names], 1)
            U, sv, _ = torch.linalg.svd(Sm, full_matrices=False)
            r = max(2, min(gd_from_sv(sv, *Sm.shape), nA - 1))
            Z = U[:, :r]
            B = {c: Z.T @ XcA[c] for c in names}
            Bt = B[tgt]
            zevA = sum((XcA[c] @ B[c].T) for c in srcs) / len(srcs)
            zevB = sum((XcB[c] @ B[c].T) for c in srcs) / len(srcs)
            zevE = sum((XcE[c] @ B[c].T) for c in srcs) / len(srcs)
            WA = torch.linalg.pinv(torch.c_[torch.ones(len(A), 1, device=dev), zevA].double()) @ (XcA[tgt] @ Bt.T).double()
            dcal = ((torch.c_[torch.ones(len(Bs_), 1, device=dev), zevB].double() @ WA) @ Bt.double()).float()
            dte = ((torch.c_[torch.ones(len(te), 1, device=dev), zevE].double() @ WA) @ Bt.double()).float()

            F = torch.cat([Sp[c][A] for c in srcs], 1)
            W = torch.linalg.solve(F.double().T @ F.double() + 10.0 * torch.eye(F.shape[1], dtype=torch.float64, device=dev),
                                   F.double().T @ Sp[tgt][A].double()).float()
            pp_cal = torch.cat([Sp[c][Bs_] for c in srcs], 1) @ W
            pp_te = torch.cat([Sp[c][te] for c in srcs], 1) @ W
            gp_cal = torch.tensor(scores_np(dcal.cpu().numpy()), device=dev)
            gp_te = torch.tensor(scores_np(dte.cpu().numpy()), device=dev)

            FC = feats_one(Bs_, pp_cal, gp_cal, srcs, Sp)
            FE = feats_one(te, pp_te, gp_te, srcs, Sp)
            fmu, fsd = FC.mean(0), FC.std(0) + 1e-8
            FCs = torch.c_[(FC - fmu) / fsd, torch.ones(len(FC), 1, device=dev)]
            FEs = torch.c_[(FE - fmu) / fsd, torch.ones(len(FE), 1, device=dev)]
            yt = rowcorr_t(pp_cal, ycal)
            w = torch.linalg.solve(FCs.double().T @ FCs.double() + 1e-2 * torch.eye(FCs.shape[1], dtype=torch.float64, device=dev),
                                   FCs.double().T @ yt.double()).float()
            rel_cal = (FCs @ w).cpu().numpy()
            rel = (FEs @ w).cpu().numpy()

            Pc = pp_cal.cpu().numpy()
            Yc = ycal.cpu().numpy()
            lab = hit3_ge1(Pc, Yc)
            Xc = shape_feats(Pc, rel_cal)
            mu2, sd2 = Xc.mean(0), Xc.std(0) + 1e-8
            lw = logit_fit((Xc - mu2) / sd2, lab)
            Xe = shape_feats(pp_te.cpu().numpy(), rel)
            dec = 1 / (1 + np.exp(-np.clip(np.c_[(Xe - mu2) / sd2, np.ones(len(Xe))] @ lw, -30, 30)))

            P = pp_te.cpu().numpy()
            Y = yte.cpu().numpy()
            for j, qi in enumerate(te.cpu().numpy()):
                REC[int(qi)].append((tgt, P[j], float(rel[j]), float(dec[j]), Y[j]))
    print(f"phase1 {tgt} {time.time()-t0:.0f}s", flush=True)

Q = [q for q in range(n) if len(REC[q]) == len(names) * NREP]
print(f"records {len(Q)}/{n}", flush=True)
prof, truth, relh, dech = {}, {}, {}, {}
for q in Q:
    byline = {c: [] for c in names}
    for (c, p, rl, dc, y) in REC[q]:
        byline[c].append((p, rl, dc, y))
    pl = {c: np.mean([x[0] for x in byline[c]], 0) for c in names}
    prof[q] = np.mean([pl[c] for c in names], 0)
    truth[q] = np.mean([np.mean([x[3] for x in byline[c]], 0) for c in names], 0)
    relh[q] = float(np.mean([np.mean([x[1] for x in byline[c]]) for c in names]))
    dech[q] = float(np.mean([np.mean([x[2] for x in byline[c]]) for c in names]))

Pm = np.array([prof[q] for q in Q])
Ym = np.array([truth[q] for q in Q])
REL = np.array([relh[q] for q in Q])
DEC = np.array([dech[q] for q in Q])
H = hit3_ge1(Pm, Ym)
rcs = np.array([np.corrcoef(Pm[i], Ym[i])[0, 1] for i in range(len(Q))])
print(f"full: hit3_ge1 {H.mean():.4f}  rowcorr {rcs.mean():.4f}  n={len(Q)}", flush=True)
print(f"AUC decision {auc(DEC, H):.4f}  AUC profile {auc(REL, H):.4f}", flush=True)

COVS = np.round(np.arange(0.05, 1.0001, 0.05), 2)
rng = np.random.RandomState(0)
ORD = {"decision": np.argsort(-DEC), "profile": np.argsort(-REL),
       "random": rng.permutation(len(Q))}
out = {"n": int(len(Q)), "full": {"hit3_ge1": float(H.mean()), "rowcorr": float(rcs.mean())},
       "auc": {"decision": auc(DEC, H), "profile": auc(REL, H)},
       "coverage": COVS.tolist(), "curves": {}}
for tag, o in ORD.items():
    out["curves"][tag] = {"hit3": [], "delivered": []}
    for c in COVS:
        sel = o[: max(1, int(round(c * len(o))))]
        h = float(H[sel].mean())
        out["curves"][tag]["hit3"].append(h)
        out["curves"][tag]["delivered"].append(float(c * h + (1 - c)))
k10 = int(round(0.1 * len(Q)))
out["riskiest_decile_hit3"] = float(H[ORD["decision"][-k10:]].mean())
out["best_decile_hit3"] = float(H[ORD["decision"][:k10]].mean())
out["per_target"] = [{"iq": int(q), "hit3": float(H[i]), "rel": float(REL[i]),
                      "dec": float(DEC[i]), "rowcorr": float(rcs[i])}
                     for i, q in enumerate(Q)]
json.dump(out, open("runs/kt118_decision.json", "w"), indent=1, default=float)
print(json.dumps({k: out[k] for k in ("full", "auc", "riskiest_decile_hit3", "best_decile_hit3")}, indent=1), flush=True)
for tag in ORD:
    print(tag, "hit3", [round(x, 3) for x in out["curves"][tag]["hit3"]], flush=True)
print("DONE", flush=True)
