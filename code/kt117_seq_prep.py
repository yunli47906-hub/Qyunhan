"""kt117 step 1 (local): map gene symbols to canonical human protein sequences.

Source: UniProtKB/Swiss-Prot reviewed human FASTA (GN= field).  One entry per
gene, preferring PE=1 and then the longest isoform.
"""
import gzip, json, re

FA = "seq/uniprot_human_reviewed.fasta.gz"
GN = re.compile(r"GN=(\S+)")
PE = re.compile(r"PE=(\d)")

best = {}


def _store(best, name, acc, seq, pe):
    if not name or not seq:
        return
    cur = best.get(name)
    key = (pe, -len(seq))
    if cur is None or key < cur[0]:
        best[name] = (key, acc, seq)


with gzip.open(FA, "rt", encoding="utf-8", errors="replace") as f:
    acc = name = None
    chunks = []
    for line in f:
        if line.startswith(">"):
            if acc:
                _store(best, name, acc, "".join(chunks), pe)
            hdr = line[1:].strip()
            parts = hdr.split("|")
            acc = parts[1] if len(parts) > 1 else hdr.split()[0]
            g = GN.search(hdr)
            name = g.group(1).upper() if g else None
            p = PE.search(hdr)
            pe = int(p.group(1)) if p else 9
            chunks = []
        else:
            chunks.append(line.strip())
    if acc:
        _store(best, name, acc, "".join(chunks), pe)


D = json.load(open("runs/kt111_dump.json"))
comp = set(D["pert_names"])
PRI = json.load(open("runs/kt116_llm_priors.json"))
llm = set(PRI["priors"])
TB = ["HSPA9", "MTOR", "ATF4", "MFN2", "HSPA8", "TFRC", "MYC", "PTEN", "HMGCR", "FASN",
      "DDIT3", "BAX", "NFE2L2", "SIRT1", "NFKB1", "HIF1A", "AKT1", "XBP1", "ATF6",
      "CASP3", "MAP1LC3B", "SQSTM1", "GPX4", "SLC7A11", "STAT1", "IRF1", "BECN1",
      "ATG5", "ATG7", "SLC2A1", "LDHA", "PKM", "RB1", "MAPK1", "MAPK3", "PIK3CA",
      "RICTOR", "FOXO3", "SREBF1", "LDLR", "ACACA", "UCP2", "KEAP1", "TSC1", "TSC2",
      "RPTOR", "EIF2AK3", "ERN1", "EIF2AK1", "PPP1R15A", "GCLC", "GCLM", "TXNRD1",
      "NQO1", "HMOX1", "PTGS2", "ALOX5", "VDR", "NR1H4", "PPARG", "PPARA", "SIRT3",
      "FOXO1", "CTNNB1", "GSK3B", "TNFAIP3", "IL10", "IFNG", "TNF", "IL1B", "NLRP3",
      "CARD9", "TLR2", "TLR4", "MYD88", "IRAK4", "TICAM1", "RELA", "NFKBIA", "CHUK"]
kept = {g: {"acc": v[1], "seq": v[2]} for g, v in best.items()
        if g in comp or g in llm or g in set(TB)}
json.dump(kept, open("seq/gene_seq.json", "w"))

for tag, want in (("compendium", comp), ("llm_prior", llm), ("tb_cand", set(TB))):
    have = sorted(want & set(kept))
    miss = sorted(want - set(kept))
    print(f"{tag}: {len(have)}/{len(want)} mapped")
    if miss:
        print(f"  missing ({len(miss)}): {miss}")
L = [len(v["seq"]) for v in kept.values()]
print(f"kept {len(kept)} genes, length median {sorted(L)[len(L)//2]}, max {max(L)}")
