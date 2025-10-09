import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from goatools.obo_parser import GODag
from transformers import AutoTokenizer, AutoModel


def load_terms_from_csv(csv_path: str, col: str = "GOID"):
    df = pd.read_csv(csv_path)
    assert col in df.columns, f"{csv_path} missing: {col}"
    return sorted(list(set(df[col].astype(str).tolist())))

def download_go_obo(file_path="go.obo"):
    import requests
    if not os.path.exists(file_path):
        print("Downloading go.obo ...")
        url = "http://purl.obolibrary.org/obo/go.obo"
        r = requests.get(url)
        r.raise_for_status()
        with open(file_path, "wb") as f:
            f.write(r.content)
    return file_path

def safe_get_definition(term):
    definition = getattr(term, "definition", None)
    if definition is None:
        definition = getattr(term, "defn", None)
    if definition is None and hasattr(term, "termdef"):
        definition = term.termdef
    return definition or ""

def build_term_texts(terms, godag, add_neighbors=True, neighbor_k=5):
    rows, id2text = [], {}
    for gid in tqdm(terms, desc="Compose GO texts"):
        if gid not in godag:
            continue
        t = godag[gid]
        if t.is_obsolete:
            continue
        name = t.name or ""
        ns = t.namespace or ""
        definition = safe_get_definition(t)
        syns = [str(s) for s in getattr(t, "synonyms", [])]

        neighbors = []
        if add_neighbors:
            parents = [p for p in t.parents if not p.is_obsolete][:neighbor_k]
            children = [c for c in t.children if not c.is_obsolete][:neighbor_k]
            neighbors = [p.name for p in parents if p.name] + [c.name for c in children if c.name]

        text_parts = [
            f"Name: {name}",
            f"Definition: {definition}" if definition else "",
            f"Synonyms: {', '.join(syns)}" if syns else "",
            f"Neighbors: {', '.join(neighbors)}" if neighbors else ""
        ]
        text = "\n".join([p for p in text_parts if p])
        rows.append({"GOID": gid, "Name": name, "Namespace": ns, "Text": text})
        id2text[gid] = text
    return pd.DataFrame(rows), id2text

def filter_by_namespace(df, ns):
    if ns.upper() == "ALL":
        return df
    mapping = {"BP": "biological_process", "CC": "cellular_component", "MF": "molecular_function"}
    return df[df["Namespace"] == mapping[ns.upper()]].reset_index(drop=True)

class HFEncoder:
    def __init__(self, model_name="dmis-lab/biobert-base-cased-v1.1", max_len=256, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.max_len = max_len

    @torch.no_grad()
    def encode(self, texts, batch_size=16):
        self.model.eval()
        out_vecs = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encode"):
            batch = texts[i:i+batch_size]
            enc = self.tok(batch, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt").to(self.device)
            out = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)
            vec = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            out_vecs.append(F.normalize(vec, p=2, dim=1).cpu())
        return torch.cat(out_vecs, dim=0)

def build_topk_edges(emb, ids, topk=15):
    N = emb.size(0)
    sims = emb @ emb.T
    edges = []
    for i in range(N):
        sims[i, i] = -1e9
        vals, idx = torch.topk(sims[i], k=min(topk, N-1))
        for v, j in zip(vals.tolist(), idx.tolist()):
            edges.append((ids[i], ids[j], float(v)))
    df = pd.DataFrame(edges, columns=["source", "target", "score"])
    df["key"] = df.apply(lambda r: tuple(sorted([r["source"], r["target"]])), axis=1)
    df = df.sort_values("score", ascending=False).drop_duplicates("key").drop(columns=["key"])
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="bipartite_*.csv")
    ap.add_argument("--obo", default="go.obo")
    ap.add_argument("--namespace", default="BP", help="BP/CC/MF/ALL")
    ap.add_argument("--model_name", default="dmis-lab/biobert-base-cased-v1.1")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--topk", type=int, default=15)
    ap.add_argument("--outdir", default="process")
    args = ap.parse_args()

    suffix = args.namespace.upper()
    os.makedirs(args.outdir, exist_ok=True)

    terms = load_terms_from_csv(args.csv)
    download_go_obo(args.obo)
    godag = GODag(args.obo)
    df, _ = build_term_texts(terms, godag)
    df = filter_by_namespace(df, args.namespace)
    if df.empty:
        raise ValueError(f"No terms for {suffix}")
    df.to_excel(os.path.join(args.outdir, f"GO_{suffix}.norm.xlsx"), index=False)

    encoder = HFEncoder(args.model_name, args.max_len)
    emb = encoder.encode(df["Text"].tolist(), args.batch_size)
    torch.save(emb, os.path.join(args.outdir, f"emb_go_text_{suffix}.pt"))

    edges = build_topk_edges(emb, df["GOID"].tolist(), args.topk)
    edges.to_csv(os.path.join(args.outdir, f"go_edges_topk_undirected_{suffix}.csv"), index=False)

    print("Finished building", suffix)

if __name__ == "__main__":
    main()
