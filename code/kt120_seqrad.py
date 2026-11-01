"""kt120 (GPU-lite): conformal radius vs nominal level for the sequence-only channel.

Fig 7c. Same quantities as kt117's abstention block (relative error, difficulty
score from embedding geometry, split-conformal with a linear difficulty model),
but swept over the nominal level instead of evaluated at 0.9 only.

Inputs are kt117's saved artefacts, so nothing is re-embedded:
  kt117_oof.npy        leave-gene-out ESM2 ridge predictions (n x 45)
  kt117_esm_mean.npy   mean-pooled ESM2 embeddings (n x 1280)
  kt111_dump.json      targets (truth) in pert_names order
  seq/gene_seq.json    which compendium genes have a sequence
"""
import json, numpy as np

R = "runs/"
SEQ = json.load(open("seq/gene_seq.json"))
D = json.load(open(R + "kt111_dump.json"))
names = D["pert_names"]
Yall = np.array(D["truth"], float)
G = [g for g in names if g in SEQ]
Y = np.array([Yall[names.index(g)] for g in G])
n = len(G)

Oof = np.load(R + "kt117_oof.npy")
emb = np.load(R + "kt117_esm_mean.npy")
assert Oof.shape == Y.shape, (Oof.shape, Y.shape)
print(f"n={n} K={Y.shape[1]}", flush=True)


def rel_err(P, T):
    return np.linalg.norm(T - P, axis=1) / np.linalg.norm(T, axis=1).clip(1e-9)


E = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
dist_all = 1 - E @ E.T
np.fill_diagonal(dist_all, np.inf)
hard = np.c_[np.sort(dist_all, 1)[:, :5].mean(1), np.linalg.norm(Oof, axis=1)]
hard = np.c_[hard, hard[:, 0] * hard[:, 1]]
Hc = (hard - hard.mean(0)) / (hard.std(0) + 1e-8)
e = rel_err(Oof, Y)
print(f"mean relative error {e.mean():.4f}", flush=True)

NREP, NFOLD = 5, 5
NOM = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
rows = {a: {"covered": [], "width": []} for a in NOM}
for rep in range(NREP):
    g = np.random.RandomState(99 + rep)
    pm = g.permutation(n)
    for f in range(NFOLD):
        te = pm[f::5]
        cal = np.setdiff1d(pm, te)
        hc = np.c_[Hc[cal], np.ones(len(cal))]
        A = hc.T @ hc + 1e-2 * np.eye(hc.shape[1])
        w = np.linalg.solve(A, hc.T @ e[cal])
        ec = hc @ w
        sc = np.c_[Hc[te], np.ones(len(te))] @ w
        for a in NOM:
            q = np.quantile(ec, a)
            rows[a]["covered"].append(float((e[te] <= sc + q).mean()))
            rows[a]["width"].append(float((sc + q).mean()))
out = {"n": int(n), "rel_err_mean": float(e.mean()),
       "nominal": NOM,
       "covered": [float(np.mean(rows[a]["covered"])) for a in NOM],
       "width": [float(np.mean(rows[a]["width"])) for a in NOM],
       "profile_norm_mean": float(np.linalg.norm(Y, axis=1).mean())}
json.dump(out, open(R + "kt120_seqrad.json", "w"), indent=1)
print(json.dumps(out, indent=1), flush=True)
print("DONE", flush=True)
