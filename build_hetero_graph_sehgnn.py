import os
import re
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
from collections import defaultdict

DATA_DIR = "data/HG"
PROC_DIR = "process"
os.makedirs(PROC_DIR, exist_ok=True)


def read_csv_safe(path, **kw):
    return pd.read_csv(path, **kw)

def clean_mesh_id(x: str) -> str:
    if pd.isna(x):
        return None
    x = str(x).strip()
    x = re.sub(r'^(MESH:)', '', x, flags=re.IGNORECASE)
    return x

def as_str(x):
    return None if pd.isna(x) else str(x).strip()

def to_int_or_str(x):
    if pd.isna(x): 
        return None
    try:
        return int(str(x).replace(',', '').strip())
    except:
        return str(x).strip()

def build_index(items):
    uniq = list(dict.fromkeys(items)) 
    id2idx = {k:i for i,k in enumerate(uniq)}
    idx2id = uniq
    return id2idx, idx2id

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def save_csr(mtx, path):
    sp.save_npz(path, mtx.tocsr())

def load_embeddings_csv(path, key_name):
    if not os.path.exists(path):
        return {}
    df = read_csv_safe(path, header=None)
    embs = {}
    for i, row in df.iterrows():
        key = str(row.iloc[0]).strip()
        vec = row.iloc[1:].astype(float).values
        embs[key] = vec
    return embs

def ensure_feat_matrix(keys, emb_dict, dim=None):
    if dim is None:
        for v in emb_dict.values():
            dim = len(v)
            break
        if dim is None:
            dim = 64 
    feats = np.zeros((len(keys), dim), dtype=np.float32)
    miss = 0
    for i, k in enumerate(keys):
        if k in emb_dict:
            v = emb_dict[k]
            if len(v) != dim:
                w = np.zeros(dim, dtype=np.float32)
                n = min(dim, len(v))
                w[:n] = v[:n]
                feats[i] = w
            else:
                feats[i] = v
        else:
            miss += 1
    if miss > 0:
        print(f"[warn] {miss}/{len(keys)} nodes missing embeddings; filled zeros.")
    return feats

def make_bipartite_adj(left_ids, right_ids, pairs, weight=1.0):
    rows, cols, data = [], [], []
    for a, b in pairs:
        if a in left_ids and b in right_ids:
            rows.append(left_ids[a]); cols.append(right_ids[b]); data.append(weight)
    if len(rows) == 0:
        shape = (len(left_ids), len(right_ids))
        return sp.csr_matrix(shape, dtype=np.float32)
    mat = sp.coo_matrix((data, (rows, cols)),
                        shape=(len(left_ids), len(right_ids)),
                        dtype=np.float32)
    return mat.tocsr()

def make_undirected_adj(ids, edges, weight=1.0):
    rows, cols, data = [], [], []
    for e in edges:
        if len(e) == 3:
            u, v, s = e
            w = float(s) if s is not None else weight
        else:
            u, v = e[:2]
            w = weight
        if (u in ids) and (v in ids) and (u != v):
            ui, vi = ids[u], ids[v]
            rows.extend([ui, vi]); cols.extend([vi, ui]); data.extend([w, w])
    if len(rows) == 0:
        n = len(ids)
        return sp.csr_matrix((n, n), dtype=np.float32)
    mat = sp.coo_matrix((data, (rows, cols)),
                        shape=(len(ids), len(ids)),
                        dtype=np.float32)
    mat = mat.tocsr()
    mat.setdiag(0.0)
    mat.eliminate_zeros()
    return mat


dd_csv = os.path.join(DATA_DIR, "Drug_Disease.csv")
df_dd = read_csv_safe(dd_csv, sep=",")
df_dd["Drug"] = df_dd["Drug"].map(as_str)
df_dd["MESH ID"] = df_dd["MESH ID"].map(clean_mesh_id)
pairs_drug_disease = [(d, m) for d, m in zip(df_dd["Drug"], df_dd["MESH ID"]) if (d and m)]

dm_xlsx = os.path.join(DATA_DIR, "Drug_Microbe.xlsx")
df_dm = pd.read_excel(dm_xlsx)
df_dm["Drug"] = df_dm["Drug"].map(as_str)
df_dm["Tax ID"] = df_dm["Tax ID"].map(lambda x: as_str(x).replace(',', '') if as_str(x) else None)
pairs_drug_microbe = [(d, t) for d, t in zip(df_dm["Drug"], df_dm["Tax ID"]) if (d and t)]

md_xlsx = os.path.join(DATA_DIR, "Microbe_Disease.xlsx")
df_md = pd.read_excel(md_xlsx)
df_md["MESH ID"] = df_md["MESH ID"].map(clean_mesh_id)
df_md["Tax ID"] = df_md["Tax ID"].map(lambda x: as_str(x).replace(',', '') if as_str(x) else None)
pairs_microbe_disease = [(t, m) for t, m in zip(df_md["Tax ID"], df_md["MESH ID"]) if (t and m)]

mg_xlsx = os.path.join(DATA_DIR, "Microbe_Gene.xlsx")
df_mg = pd.read_excel(mg_xlsx)
df_mg["Tax ID"] = df_mg["Tax ID"].map(lambda x: as_str(x).replace(',', '') if as_str(x) else None)
df_mg["Entrez ID"] = df_mg["Entrez ID"].map(to_int_or_str)
pairs_microbe_gene = [(t, g) for t, g in zip(df_mg["Tax ID"], df_mg["Entrez ID"]) if (t and g is not None)]

drug_sim_csv = os.path.join(PROC_DIR, "drug_edges_topk_undirected.csv")
df_drug_sim = read_csv_safe(drug_sim_csv)
df_drug_sim["source"] = df_drug_sim["source"].map(as_str)
df_drug_sim["target"] = df_drug_sim["target"].map(as_str)
edges_drug_drug = [(s, t, df_drug_sim.loc[i, "score"])
                   for i,(s,t) in enumerate(zip(df_drug_sim["source"], df_drug_sim["target"]))
                   if (s and t)]

gene_sim_csv = os.path.join(PROC_DIR, "gene_edges_topk_undirected_cosine.csv")
df_gene_sim = read_csv_safe(gene_sim_csv)
df_gene_sim["source"] = df_gene_sim["source"].map(to_int_or_str)
df_gene_sim["target"] = df_gene_sim["target"].map(to_int_or_str)
edges_gene_gene = [(s, t, df_gene_sim.loc[i, "score"])
                   for i,(s,t) in enumerate(zip(df_gene_sim["source"], df_gene_sim["target"]))
                   if (s is not None and t is not None)]

disease_sim_csv = os.path.join(PROC_DIR, "disease_edges_topk.csv")
df_dis_sim = read_csv_safe(disease_sim_csv)
df_dis_sim["source"] = df_dis_sim["source"].map(clean_mesh_id)
df_dis_sim["target"] = df_dis_sim["target"].map(clean_mesh_id)
edges_dis_dis = [(s, t, df_dis_sim.loc[i, "score"])
                 for i,(s,t) in enumerate(zip(df_dis_sim["source"], df_dis_sim["target"]))
                 if (s and t)]

microbe_sim_csv = os.path.join(PROC_DIR, "microbe_edges_topk_undirected.csv")
df_mic_sim = read_csv_safe(microbe_sim_csv)
df_mic_sim["source"] = df_mic_sim["source"].map(as_str)
df_mic_sim["target"] = df_mic_sim["target"].map(as_str)

def infer_taxid_from_name(s):
    if not s: return None
    m = re.search(r'(\d+)\s*$', s.strip())
    return m.group(1) if m else None

name2tax = {}
for n, t in zip(df_dm["Microbe"].map(as_str), df_dm["Tax ID"]):
    if n and t and n not in name2tax:
        name2tax[n] = t
for n, t in zip(df_md["Microbe"].map(as_str), df_md["Tax ID"]):
    if n and t and n not in name2tax:
        name2tax[n] = t

def map_microbe_name_to_id(name):
    if name in name2tax:
        return name2tax[name]
    tid = infer_taxid_from_name(name)
    return tid if tid else name 

edges_mic_mic = []
for i,(s,t) in enumerate(zip(df_mic_sim["source"], df_mic_sim["target"])):
    if not (s and t): 
        continue
    sid = map_microbe_name_to_id(s)
    tid = map_microbe_name_to_id(t)
    edges_mic_mic.append((sid, tid, df_mic_sim.loc[i, "score"]))

drug_nodes = set()
gene_nodes = set()
disease_nodes = set()
microbe_nodes = set()

drug_nodes |= {d for d,_ in pairs_drug_disease}
disease_nodes |= {m for _,m in pairs_drug_disease}

drug_nodes |= {d for d,_ in pairs_drug_microbe}
microbe_nodes |= {m for _,m in pairs_drug_microbe}

microbe_nodes |= {m for m,_ in pairs_microbe_disease}
disease_nodes |= {d for _,d in pairs_microbe_disease}

microbe_nodes |= {m for m,_ in pairs_microbe_gene}
gene_nodes    |= {g for _,g in pairs_microbe_gene}

drug_nodes |= {u for u,_,_ in edges_drug_drug} | {v for _,v,_ in edges_drug_drug}
gene_nodes |= {u for u,_,_ in edges_gene_gene} | {v for _,v,_ in edges_gene_gene}
disease_nodes |= {u for u,_,_ in edges_dis_dis} | {v for _,v,_ in edges_dis_dis}
microbe_nodes |= {u for u,_,_ in edges_mic_mic} | {v for _,v,_ in edges_mic_mic}

emb_drug_csv    = os.path.join(PROC_DIR, "emb_drug_gat_gae.csv")
emb_gene_csv    = os.path.join(PROC_DIR, "emb_gene_gat_gae_BIP.csv")
emb_dis_csv     = os.path.join(PROC_DIR, "emb_disease_gat_gae.csv")
emb_microbe_csv = os.path.join(PROC_DIR, "emb_microbe_gat_gae.csv")

emb_drug = load_embeddings_csv(emb_drug_csv, key_name="Drug")
emb_gene = load_embeddings_csv(emb_gene_csv, key_name="GeneID")
emb_dis  = load_embeddings_csv(emb_dis_csv,  key_name="DiseaseName")
emb_mic  = load_embeddings_csv(emb_microbe_csv, key_name="MicrobeName")

drug_nodes    |= set(emb_drug.keys())
gene_nodes    |= set(emb_gene.keys())
name2mesh_counts = defaultdict(lambda: defaultdict(int))
for name, mid in zip(df_dd["Disease"].map(as_str), df_dd["MESH ID"]):
    if name and mid:
        name2mesh_counts[name][mid] += 1
for name, mid in zip(df_md["Disease"].map(as_str), df_md["MESH ID"]):
    if name and mid:
        name2mesh_counts[name][mid] += 1

mapped_emb_dis = {}
for name, vec in emb_dis.items():
    mesh_map = name2mesh_counts.get(name, {})
    if len(mesh_map) == 1:
        mesh_id = next(iter(mesh_map.keys()))
        mapped_emb_dis[mesh_id] = vec
        disease_nodes.add(mesh_id)
    elif len(mesh_map) > 1:
        mesh_id = max(mesh_map.items(), key=lambda kv: kv[1])[0]
        mapped_emb_dis[mesh_id] = vec
        disease_nodes.add(mesh_id)
    else:
        mapped_emb_dis[name] = vec
        disease_nodes.add(name)
emb_dis = mapped_emb_dis

microbe_nodes |= set(map_microbe_name_to_id(n) for n in emb_mic.keys())

gene_nodes = {str(x) for x in gene_nodes if x is not None}
emb_gene = {str(k):v for k,v in emb_gene.items()}

drug_id2idx, drug_idx2id       = build_index(sorted(drug_nodes))
gene_id2idx, gene_idx2id       = build_index(sorted(gene_nodes, key=lambda x: (len(str(x)), str(x))))
disease_id2idx, disease_idx2id = build_index(sorted(disease_nodes))
microbe_id2idx, microbe_idx2id = build_index(sorted(microbe_nodes))

save_json(drug_idx2id,    os.path.join(PROC_DIR, "drug_id_map.json"))
save_json(gene_idx2id,    os.path.join(PROC_DIR, "gene_id_map.json"))
save_json(disease_idx2id, os.path.join(PROC_DIR, "disease_id_map.json"))
save_json(microbe_idx2id, os.path.join(PROC_DIR, "microbe_id_map.json"))

print(f"[nodes] Drug={len(drug_idx2id)}, Gene={len(gene_idx2id)}, Disease={len(disease_idx2id)}, Microbe={len(microbe_idx2id)}")

drug_feats    = ensure_feat_matrix(drug_idx2id, emb_drug)
gene_feats    = ensure_feat_matrix(gene_idx2id, emb_gene)
disease_feats = ensure_feat_matrix(disease_idx2id, emb_dis)
emb_mic_mapped = {}
for name, vec in emb_mic.items():
    mid = map_microbe_name_to_id(name)
    emb_mic_mapped[mid] = vec
microbe_feats = ensure_feat_matrix(microbe_idx2id, emb_mic_mapped)

np.save(os.path.join(PROC_DIR, "drug_feats.npy"),    drug_feats)
np.save(os.path.join(PROC_DIR, "gene_feats.npy"),    gene_feats)
np.save(os.path.join(PROC_DIR, "disease_feats.npy"), disease_feats)
np.save(os.path.join(PROC_DIR, "microbe_feats.npy"), microbe_feats)
print("[ok] saved feature matrices.")

A_DrugDrug     = make_undirected_adj(drug_id2idx,    edges_drug_drug)
A_GeneGene     = make_undirected_adj(gene_id2idx,    edges_gene_gene)
A_DiseaseDisease = make_undirected_adj(disease_id2idx, edges_dis_dis)
A_MicrobeMicrobe = make_undirected_adj(microbe_id2idx, edges_mic_mic)

save_csr(A_DrugDrug,       os.path.join(PROC_DIR, "adj_Drug-Drug.npz"))
save_csr(A_GeneGene,       os.path.join(PROC_DIR, "adj_Gene-Gene.npz"))
save_csr(A_DiseaseDisease, os.path.join(PROC_DIR, "adj_Disease-Disease.npz"))
save_csr(A_MicrobeMicrobe, os.path.join(PROC_DIR, "adj_Microbe-Microbe.npz"))

A_DrugDisease = make_bipartite_adj(drug_id2idx, disease_id2idx, pairs_drug_disease)
A_DiseaseDrug = A_DrugDisease.T.tocsr()
save_csr(A_DrugDisease, os.path.join(PROC_DIR, "adj_Drug-Disease.npz"))
save_csr(A_DiseaseDrug, os.path.join(PROC_DIR, "adj_Disease-Drug.npz"))

A_DrugMicrobe = make_bipartite_adj(drug_id2idx, microbe_id2idx, pairs_drug_microbe)
A_MicrobeDrug = A_DrugMicrobe.T.tocsr()
save_csr(A_DrugMicrobe, os.path.join(PROC_DIR, "adj_Drug-Microbe.npz"))
save_csr(A_MicrobeDrug, os.path.join(PROC_DIR, "adj_Microbe-Drug.npz"))

A_MicrobeDisease = make_bipartite_adj(microbe_id2idx, disease_id2idx, pairs_microbe_disease)
A_DiseaseMicrobe = A_MicrobeDisease.T.tocsr()
save_csr(A_MicrobeDisease, os.path.join(PROC_DIR, "adj_Microbe-Disease.npz"))
save_csr(A_DiseaseMicrobe, os.path.join(PROC_DIR, "adj_Disease-Microbe.npz"))

A_MicrobeGene = make_bipartite_adj(microbe_id2idx, gene_id2idx, pairs_microbe_gene)
A_GeneMicrobe = A_MicrobeGene.T.tocsr()
save_csr(A_MicrobeGene, os.path.join(PROC_DIR, "adj_Microbe-Gene.npz"))
save_csr(A_GeneMicrobe, os.path.join(PROC_DIR, "adj_Gene-Microbe.npz"))

print("[ok] saved adjacency matrices.")
raw_features = {
    "Drug":    drug_feats,     # shape [N_Drug,  D_d]
    "Gene":    gene_feats,     # shape [N_Gene,  D_g]
    "Disease": disease_feats,  # shape [N_Dis,   D_s]
    "Microbe": microbe_feats,  # shape [N_Mic,   D_m]
}

adj_dict = {
    "Drug-Drug":       A_DrugDrug,
    "Gene-Gene":       A_GeneGene,
    "Disease-Disease": A_DiseaseDisease,
    "Microbe-Microbe": A_MicrobeMicrobe,

    "Drug-Disease":    A_DrugDisease,
    "Disease-Drug":    A_DiseaseDrug,

    "Drug-Microbe":    A_DrugMicrobe,
    "Microbe-Drug":    A_MicrobeDrug,

    "Microbe-Disease": A_MicrobeDisease,
    "Disease-Microbe": A_DiseaseMicrobe,

    "Microbe-Gene":    A_MicrobeGene,
    "Gene-Microbe":    A_GeneMicrobe,
}

def aggregate_features_by_metapath(adj_dict, raw_features, metapath):
    types = metapath
    assert len(types) >= 1
    X = raw_features[types[0]]
    for i in range(len(types)-1):
        src, dst = types[i], types[i+1]
        A = adj_dict.get(f"{src}-{dst}", None)
        if A is None:
            raise ValueError(f"Missing adjacency for {src}-{dst}")
        X = A.dot(raw_features[dst])  
        deg = np.asarray(A.sum(1)).reshape(-1)
        deg[deg == 0] = 1.0
        X = X / deg[:, None]
    return X

features_by_metapath_for_drug = {
    "Drug": raw_features["Drug"],  
    "Dg-Dis-Dg": aggregate_features_by_metapath(adj_dict, raw_features, ["Drug","Disease","Drug"]),
    "Dg-Mic-Dg": aggregate_features_by_metapath(adj_dict, raw_features, ["Drug","Microbe","Drug"]),
}

np.savez_compressed(
    os.path.join(PROC_DIR, "features_by_metapath_for_drug.npz"),
    **{k:v.astype(np.float32) for k,v in features_by_metapath_for_drug.items()}
)