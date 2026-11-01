"""kt110 (GPU): two deployment-critical ablations.

(A) Source-count ablation.  Deployment targets are often measured in only 2-3
    of the 4 lines.  Restrict the source set to s = 1, 2, 3 lines and measure
    Hallmark program-level consensus rowcorr and hit@3 for the target line.

(B) Reliability-head transfer.  Fit the hardness head on rows from three target
    lines and evaluate its risk-coverage AUC on the held-out fourth line.  If it
    transfers, reliability is a property of the perturbation, not of the line,
    which is what the label-free deployment claim requires.
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
        pnames.append(k.replace("HALLMARK_", ""))
        idxs.append(ix)
K = len(idxs)


def scores_np(M):
    S = np.stack([M[:, ix].mean(1) for ix in idxs], 1)
    return (S - S.mean(0)) / (S.std(0) + 1e-8)


S = {c: torch.tensor(scores_np(X[c].cpu().numpy()), device=dev) for c in names}
print(f"n={n} programs={K}", flush=True)


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


NREP, NFOLD = 3, 5
COVS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
t0 = time.time()

# ---------------- (A) source-count ablation ----------------
import itertools
ACC = {}
for s in (1, 2, 3):
    ACC[s] = []
for tgt in names:
    others = [c for c in names if c != tgt]
    for s in (1, 2, 3):
        for subset in itertools.combinations(others, s):
            srcs = list(subset)
            P_, Y_ = [], []
            for rep in range(NREP):
                g = torch.Generator(device=dev)
                g.manual_seed(13000 + rep)
                pm = torch.randperm(n, generator=g, device=dev)
                folds = [pm[i::NFOLD] for i in range(NFOLD)]
                for kf in range(NFOLD):
                    te = folds[kf]
                    tr = torch.cat([folds[j] for j in range(NFOLD) if j != kf])
                    # ridge directly on source program scores (prog arm)
                    F = torch.cat([S[c][tr] for c in srcs], 1)
                    W = torch.linalg.solve(
                        F.double().T @ F.double() + 10.0 * torch.eye(F.shape[1], dtype=torch.float64, device=dev),
                        F.double().T @ S[tgt][tr].double()).float()
                    pte = torch.cat([S[c][te] for c in srcs], 1) @ W
                    P_.append(pte.cpu().numpy())
                    Y_.append(S[tgt][te].cpu().numpy())
            P = np.concatenate(P_, 0)
            Y = np.concatenate(Y_, 0)
            rc = rowcorr_t(torch.tensor(P), torch.tensor(Y)).numpy()
            o = np.argsort(-P, 1)
            t = np.argmax(Y, 1)
            h3 = float(np.mean([t[i] in o[i, :3] for i in range(len(t))]))
            h1 = float(np.mean([t[i] == o[i, 0] for i in range(len(t))]))
            ACC[s].append((float(rc.mean()), h3, h1))
    print(f"src-ablation {tgt} {time.time()-t0:.0f}s", flush=True)

resA = {}
for s in (1, 2, 3):
    a = np.array(ACC[s])
    resA[s] = {"rowcorr": float(a[:, 0].mean()), "rowcorr_sd": float(a[:, 0].std()),
               "hit3": float(a[:, 1].mean()), "hit1": float(a[:, 2].mean()),
               "nsets": int(len(a))}
    print(f"s={s}: rowcorr {resA[s]['rowcorr']:.3f}  hit3 {resA[s]['hit3']:.3f}  "
          f"hit1 {resA[s]['hit1']:.3f}  ({len(a)} target-subsets)", flush=True)

# ---------------- (B) head transfer across lines ----------------
PERLINE = {}
for tgt in names:
    srcs = [c for c in names if c != tgt]
    ords, bases, ps, ys = [], [], [], []
    for rep in range(NREP):
        g = torch.Generator(device=dev)
        g.manual_seed(13000 + rep)
        pm = torch.randperm(n, generator=g, device=dev)
        folds = [pm[i::NFOLD] for i in range(NFOLD)]
        for kf in range(NFOLD):
            te = folds[kf]
            tr = torch.cat([folds[j] for j in range(NFOLD) if j != kf])
            F = torch.cat([S[c][tr] for c in srcs], 1)
            W = torch.linalg.solve(
                F.double().T @ F.double() + 10.0 * torch.eye(F.shape[1], dtype=torch.float64, device=dev),
                F.double().T @ S[tgt][tr].double()).float()
            pte = torch.cat([S[c][te] for c in srcs], 1) @ W
            ords.append(pte.cpu().numpy())
            bases.append(pte.pow(2).sum(1).sqrt().cpu().numpy())
            ps.append(pte.cpu().numpy())
            ys.append(S[tgt][te].cpu().numpy())
    PERLINE[tgt] = {k: np.concatenate(v, 0) for k, v in
                    (("p", ps), ("y", ys), ("ord", ords), ("base", bases))}

# head features are not stored per row here; use the analytic magnitude proxy and
# an oracle for reference, and fit a cross-line head on stored (p, y) rows.
FEAT = {}
for c in names:
    P, Y = PERLINE[c]["p"], PERLINE[c]["y"]
    feats_c = [np.linalg.norm(P, axis=1, keepdims=True),
               np.abs(P).mean(1, keepdims=True), np.abs(P).max(1, keepdims=True),
               (np.abs(P) > 1).mean(1, keepdims=True), np.abs(P).std(1, keepdims=True)]
    FEAT[c] = np.concatenate(feats_c, 1)
    PERLINE[c]["rc"] = rowcorr_t(torch.tensor(P), torch.tensor(Y)).numpy()


def auc_from_order(rc, o):
    return float(np.mean([rc[o[: max(1, int(round(c * len(o))))]].mean() for c in COVS]))


resB = {}
for hold in names:
    tr_lines = [c for c in names if c != hold]
    Ftr = np.concatenate([np.c_[np.ones(len(FEAT[c])), FEAT[c]] for c in tr_lines], 0)
    ytr = np.concatenate([PERLINE[c]["rc"] for c in tr_lines], 0)
    w = np.linalg.solve(Ftr.T @ Ftr + 1e-2 * np.eye(Ftr.shape[1]), Ftr.T @ ytr)
    Fho = np.c_[np.ones(len(FEAT[hold])), FEAT[hold]]
    o = np.argsort(-(Fho @ w))
    rc = PERLINE[hold]["rc"]
    o_mag = np.argsort(-PERLINE[hold]["base"])
    o_orc = np.argsort(-rc)
    resB[hold] = {"auc_transfer": auc_from_order(rc, o),
                  "auc_mag": auc_from_order(rc, o_mag),
                  "auc_oracle": auc_from_order(rc, o_orc),
                  "rowcorr_full": float(rc.mean())}
    print(f"head->{hold}: transfer {resB[hold]['auc_transfer']:.3f}  "
          f"mag {resB[hold]['auc_mag']:.3f}  oracle {resB[hold]['auc_oracle']:.3f}", flush=True)

json.dump({"source_ablation": resA, "head_transfer": resB},
          open("runs/kt110_sources.json", "w"), indent=1, default=float)
print("DONE", flush=True)
