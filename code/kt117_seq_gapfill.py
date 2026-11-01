"""kt117 step 1b (local): resolve gene symbols the FASTA GN= field missed.

UniProtKB search by symbol synonyms (gene: matches primary and alternative
names), reviewed human only, one canonical sequence per query.
"""
import json, time, urllib.parse, urllib.request

D = json.load(open("seq/gene_seq.json"))
C = json.load(open("runs/kt111_dump.json"))
PRI = json.load(open("runs/kt116_llm_priors.json"))
want = sorted((set(C["pert_names"]) | set(PRI["priors"])) - set(D))

UNI = ("https://rest.uniprot.org/uniprotkb/search?query="
       "{q}&fields=accession,sequence,gene_primary&format=tsv&size=5")


def fetch(sym):
    q = urllib.parse.quote(f"(gene:{sym}) AND (organism_id:9606) AND (reviewed:true)")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(UNI.format(q=q), timeout=30) as r:
                txt = r.read().decode("utf-8", "replace")
            break
        except Exception:
            time.sleep(3)
    else:
        return None
    rows = [x.split("\t") for x in txt.strip().split("\n")[1:] if x.count("\t") >= 1]
    rows = [r for r in rows if len(r) >= 2 and r[1].strip()]
    if not rows:
        return None
    rows.sort(key=lambda r: -len(r[1]))
    acc, seq = rows[0][0], rows[0][1].strip()
    return {"acc": acc, "seq": seq, "query": sym}


added, failed = {}, []
for s in want:
    r = fetch(s)
    if r:
        added[s] = {"acc": r["acc"], "seq": r["seq"]}
    else:
        failed.append(s)
    time.sleep(0.25)
D.update(added)
json.dump(D, open("seq/gene_seq.json", "w"))
comp = set(C["pert_names"])
print(f"resolved {len(added)}/{len(want)}; compendium coverage "
      f"{len(comp & set(D))}/{len(comp)}")
print("added:", {k: v["acc"] for k, v in added.items()})
print("failed:", failed)
