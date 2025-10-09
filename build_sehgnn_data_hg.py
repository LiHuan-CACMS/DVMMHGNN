import os
import re
import json
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
import scipy.sparse as sp

DATA_DIR = "data/HG"
PROC_DIR = "process"
FILE_DD = os.path.join(DATA_DIR, "Drug_Disease.csv")
FILE_DM = os.path.join(DATA_DIR, "Drug_Microbe.xlsx")
FILE_MD = os.path.join(DATA_DIR, "Microbe_Disease.xlsx")
FILE_MG = os.path.join(DATA_DIR, "Microbe_Gene.xlsx")
FILE_GP = os.path.join(DATA_DIR, "Gene_Pathway.xlsx")
FILE_PD = os.path.join(DATA_DIR, "Pathway_Disease.xlsx")


HOMO_EDGE_PATTERNS = {
    "Drug":    r"drug_edges_topk.*\.csv",
    "Gene":    r"gene_edges_topk.*\.csv",
    "Disease": r"disease_edges_topk.*\.csv",
    "Microbe": r"microbe_edges_topk.*\.csv",
}

EMB_FILES = {
    "Drug":    os.path.join(PROC_DIR, "emb_drug_gat_gae.csv"),
    "Gene":    os.path.join(PROC_DIR, "emb_gene_gat_gae_BIP.csv"),
    "Disease": os.path.join(PROC_DIR, "emb_disease_gat_gae.csv"),
    "Microbe": os.path.join(PROC_DIR, "emb_microbe_gat_gae.csv"),
}
BUILD_PATHWAY_FEATURES = "onehot"
PATHWAY_EMB_FILE = ""  

def _norm_str(x: str) -> str:
    if pd.isna(x):
        return ""
    x = str(x).strip()
    x = re.sub(r"\s+", " ", x)
    return x

def _read_table(path: str) -> pd.DataFrame:
    if path.endswith(".xlsx") or path.endswith(".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)

def _find_file_by_regex(dir_path: str, pattern: str) -> Optional[str]:
    regex = re.compile(pattern, re.IGNORECASE)
    for fn in os.listdir(dir_path):
        if regex.match(fn):
            return os.path.join(dir_path, fn)
    return None

def _make_indexer(items: List[str]) -> Tuple[Dict[str, int], List[str]]:
    uniq = sorted(set(items))
    return {k: i for i, k in enumerate(uniq)}, uniq

def _build_bipartite_adj(
    df: pd.DataFrame,
    left_col: str,
    right_col: str,
    left_map: Dict[str,int],
    right_map: Dict[str,int],
) -> sp.csr_matrix:
    rows, cols, data = [], [], []
    for _, row in df.iterrows():
        l = _norm_str(row[left_col])
        r = _norm_str(row[right_col])
        if l in left_map and r in right_map:
            rows.append(left_map[l]); cols.append(right_map[r]); data.append(1.0)
    if len(rows) == 0:
        return sp.csr_matrix((len(left_map), len(right_map)))
    mat = sp.coo_matrix((np.array(data, float), (np.array(rows), np.array(cols))),
                        shape=(len(left_map), len(right_map)))
    return mat.tocsr()

def _build_homo_adj_from_edgecsv(
    path: str,
    node_map: Dict[str,int],
    col_source="source",
    col_target="target",
    col_score="score",
    symmetrize=True
) -> sp.csr_matrix:
    df = pd.read_csv(path)
    rows, cols, data = [], [], []
    for _, row in df.iterrows():
        s = _norm_str(row[col_source])
        t = _norm_str(row[col_target])
        if s in node_map and t in node_map:
            rows.append(node_map[s]); cols.append(node_map[t])
            val = float(row[col_score]) if col_score in df.columns else 1.0
            data.append(val)
    n = len(node_map)
    if len(rows) == 0:
        return sp.csr_matrix((n, n))
    mat = sp.coo_matrix((np.array(data, float), (np.array(rows), np.array(cols))), shape=(n, n)).tocsr()
    if symmetrize:
        mat = mat.maximum(mat.T) 
    return mat

def _read_embeddings_csv(path: str) -> Dict[str, np.ndarray]:
    emb = {}
    df = pd.read_csv(path, header=None)
    for _, row in df.iterrows():
        key = _norm_str(row.iloc[0])
        vec = row.iloc[1:].astype(float).to_numpy()
        emb[key] = vec
    return emb

dd = _read_table(FILE_DD) if os.path.exists(FILE_DD) else pd.DataFrame()
dm = _read_table(FILE_DM) if os.path.exists(FILE_DM) else pd.DataFrame()
md = _read_table(FILE_MD) if os.path.exists(FILE_MD) else pd.DataFrame()
mg = _read_table(FILE_MG) if os.path.exists(FILE_MG) else pd.DataFrame()
gp = _read_table(FILE_GP) if os.path.exists(FILE_GP) else pd.DataFrame()
pd_df = _read_table(FILE_PD) if os.path.exists(FILE_PD) else pd.DataFrame()

if not dd.empty:
    if "MESH ID" in dd.columns:
        dd["DiseaseID"] = dd["MESH ID"].apply(_norm_str)
    elif "MESH_ID" in dd.columns:
        dd["DiseaseID"] = dd["MESH_ID"].apply(_norm_str)
    else:
        dd["DiseaseID"] = dd["Disease"].apply(_norm_str)
    dd["DrugID"] = dd["Drug"].apply(_norm_str)

if not dm.empty:
    dm["DrugID"] = dm["Drug"].apply(_norm_str)
    dm["MicrobeID"] = dm["Microbe"].apply(_norm_str)

if not md.empty:
    md["MicrobeID"] = md["Microbe"].apply(_norm_str)
    if "MESH ID" in md.columns:
        md["DiseaseID"] = md["MESH ID"].apply(_norm_str)
    else:
        md["DiseaseID"] = md["Disease"].apply(_norm_str)

if not mg.empty:
    mg["MicrobeID"] = mg["Microbe"].apply(_norm_str)
    if "Entrez ID" in mg.columns:
        mg["GeneID"] = mg["Entrez ID"].astype(str).apply(_norm_str)
    elif "EntrezID" in mg.columns:
        mg["GeneID"] = mg["EntrezID"].astype(str).apply(_norm_str)
    elif "GeneID" in mg.columns:
        mg["GeneID"] = mg["GeneID"].astype(str).apply(_norm_str)
    else:
        mg["GeneID"] = mg["Gene"].astype(str).apply(_norm_str)

if not gp.empty:
    if "Entrez ID" in gp.columns:
        gp["GeneID"] = gp["Entrez ID"].astype(str).apply(_norm_str)
    elif "EntrezID" in gp.columns:
        gp["GeneID"] = gp["EntrezID"].astype(str).apply(_norm_str)
    elif "GeneID" in gp.columns:
        gp["GeneID"] = gp["GeneID"].astype(str).apply(_norm_str)
    else:
        gp["GeneID"] = gp["Gene"].astype(str).apply(_norm_str)
    if "Pathway" in gp.columns:
        gp["PathwayID"] = gp["Pathway"].astype(str).apply(_norm_str)
    elif "Pathway ID" in gp.columns:
        gp["PathwayID"] = gp["Pathway ID"].astype(str).apply(_norm_str)
    else:
        raise ValueError("Gene_Pathway.xlsx error")

if not pd_df.empty:
    if "Pathway" in pd_df.columns:
        pd_df["PathwayID"] = pd_df["Pathway"].astype(str).apply(_norm_str)
    elif "Pathway ID" in pd_df.columns:
        pd_df["PathwayID"] = pd_df["Pathway ID"].astype(str).apply(_norm_str)
    else:
        raise ValueError("Pathway_Disease.xlsx error")
    if "MESH ID" in pd_df.columns:
        pd_df["DiseaseID"] = pd_df["MESH ID"].apply(_norm_str)
    else:
        pd_df["DiseaseID"] = pd_df["Disease"].apply(_norm_str)

all_drugs = []
all_diseases = []
all_microbes = []
all_genes = []
all_pathways = []

if not dd.empty:
    all_diseases += dd["DiseaseID"].tolist()
    all_drugs += dd["DrugID"].tolist()
if not dm.empty:
    all_drugs += dm["DrugID"].tolist()
    all_microbes += dm["MicrobeID"].tolist()
if not md.empty:
    all_microbes += md["MicrobeID"].tolist()
    all_diseases += md["DiseaseID"].tolist()
if not mg.empty:
    all_microbes += mg["MicrobeID"].tolist()
    all_genes += mg["GeneID"].tolist()
if not gp.empty:
    all_genes += gp["GeneID"].tolist()
    all_pathways += gp["PathwayID"].tolist()
if not pd_df.empty:
    all_pathways += pd_df["PathwayID"].tolist()
    all_diseases += pd_df["DiseaseID"].tolist()

drug2idx, drugs = _make_indexer(all_drugs)
disease2idx, diseases = _make_indexer(all_diseases)
microbe2idx, microbes = _make_indexer(all_microbes)
gene2idx, genes = _make_indexer(all_genes)
pathway2idx, pathways = _make_indexer(all_pathways)

print(f"  #Drug={len(drugs)}  #Disease={len(diseases)}  #Microbe={len(microbes)}  #Gene={len(genes)}  #Pathway={len(pathways)}")
adj_dict: Dict[str, sp.csr_matrix] = {}

def add_bi(name_lr: str, name_rl: str, mat_lr: sp.csr_matrix):
    adj_dict[name_lr] = mat_lr.tocsr()
    adj_dict[name_rl] = mat_lr.T.tocsr()
if not dd.empty:
    A = _build_bipartite_adj(dd, "DrugID", "DiseaseID", drug2idx, disease2idx)
    add_bi("Drug-Disease", "Disease-Drug", A)

if not dm.empty:
    A = _build_bipartite_adj(dm, "DrugID", "MicrobeID", drug2idx, microbe2idx)
    add_bi("Drug-Microbe", "Microbe-Drug", A)

if not md.empty:
    A = _build_bipartite_adj(md, "MicrobeID", "DiseaseID", microbe2idx, disease2idx)
    add_bi("Microbe-Disease", "Disease-Microbe", A)

if not mg.empty:
    A = _build_bipartite_adj(mg, "MicrobeID", "GeneID", microbe2idx, gene2idx)
    add_bi("Microbe-Gene", "Gene-Microbe", A)

if not gp.empty:
    A = _build_bipartite_adj(gp, "GeneID", "PathwayID", gene2idx, pathway2idx)
    add_bi("Gene-Pathway", "Pathway-Gene", A)

if not pd_df.empty:
    A = _build_bipartite_adj(pd_df, "PathwayID", "DiseaseID", pathway2idx, disease2idx)
    add_bi("Pathway-Disease", "Disease-Pathway", A)

for ntype, patt in HOMO_EDGE_PATTERNS.items():
    f = _find_file_by_regex(PROC_DIR, patt)
    if not f: 
        continue
    if ntype == "Drug":
        M = _build_homo_adj_from_edgecsv(f, drug2idx)
        adj_dict["Drug-Drug"] = M
    elif ntype == "Gene":
        M = _build_homo_adj_from_edgecsv(f, gene2idx)
        adj_dict["Gene-Gene"] = M
    elif ntype == "Disease":
        M = _build_homo_adj_from_edgecsv(f, disease2idx)
        adj_dict["Disease-Disease"] = M
    elif ntype == "Microbe":
        M = _build_homo_adj_from_edgecsv(f, microbe2idx)
        adj_dict["Microbe-Microbe"] = M
    elif ntype == "Pathway":
        M = _build_homo_adj_from_edgecsv(f, pathway2idx)
        adj_dict["Pathway-Pathway"] = M

raw_features: Dict[str, np.ndarray] = {}

for ntype in ["Drug", "Gene", "Disease", "Microbe"]:
    path = EMB_FILES.get(ntype, "")
    if not path or not os.path.exists(path):
        if ntype == "Drug":     N = len(drugs)
        elif ntype == "Gene":   N = len(genes)
        elif ntype == "Disease":N = len(diseases)
        else:                   N = len(microbes)
        raw_features[ntype] = np.eye(N, dtype=np.float32)
        continue

    emb = _read_embeddings_csv(path)
    if ntype == "Drug":
        idx2name = drugs; name2idx = drug2idx
    elif ntype == "Gene":
        idx2name = genes; name2idx = gene2idx
    elif ntype == "Disease":
        idx2name = diseases; name2idx = disease2idx
    else:
        idx2name = microbes; name2idx = microbe2idx

    dim = None
    feats = []
    miss = 0
    for name in idx2name:
        if name in emb:
            v = emb[name]
        else:
            if ntype == "Gene" and name not in emb:
                try:
                    key_alt = str(int(float(name)))
                    v = emb.get(key_alt, None)
                except:
                    v = None
            else:
                v = None
        if v is None:
            miss += 1
            if dim is None:
                if len(emb) > 0:
                    dim = len(next(iter(emb.values())))
                else:
                    dim = 64
            v = np.zeros(dim, dtype=np.float32)
        else:
            dim = len(v) if dim is None else dim
            v = v.astype(np.float32)
        feats.append(v)
    feats = np.vstack(feats)
    raw_features[ntype] = feats


if BUILD_PATHWAY_FEATURES == "from_file" and PATHWAY_EMB_FILE and os.path.exists(PATHWAY_EMB_FILE):
    p_emb = _read_embeddings_csv(PATHWAY_EMB_FILE)
    dim = len(next(iter(p_emb.values()))) if len(p_emb) else 64
    feats = []
    miss = 0
    for p in pathways:
        v = p_emb.get(p, None)
        if v is None:
            miss += 1
            v = np.zeros(dim, dtype=np.float32)
        feats.append(v.astype(np.float32))
    raw_features["Pathway"] = np.vstack(feats)
else:
    Np = len(pathways)
    raw_features["Pathway"] = np.eye(Np, dtype=np.float32)
    print(f"  Pathway(one-hot): shape={raw_features['Pathway'].shape}")


for k, v in raw_features.items():
    print(f"  features[{k}]: {v.shape}")


TARGET = "Disease"
metapaths = [
    [TARGET, "Drug", TARGET],
    [TARGET, "Microbe", TARGET],
    [TARGET, "Pathway", TARGET],
    [TARGET, "Gene", "Pathway", TARGET],
]
features_by_metapath: Dict[str, np.ndarray] = {}
features_by_metapath[TARGET] = raw_features[TARGET]
def _simple_metapath_aggregate(adj_dict, raw_feats, path: List[str]) -> np.ndarray:
    assert len(path) >= 2
    feat = raw_feats[path[-1]] 
    for i in range(len(path) - 1, 0, -1):
        t = path[i-1] + "-" + path[i]
        if t not in adj_dict:
            raise KeyError(f"adj_dict : {list(adj_dict.keys())[:8]} ...")
        A = adj_dict[t].tocsr()               
        deg = np.asarray(A.sum(axis=1)).reshape(-1) + 1e-8
        feat = A @ feat                        
        feat = feat / deg[:, None]             
    return feat.astype(np.float32)

features_by_metapath["D-Drug-D"]           = _simple_metapath_aggregate(adj_dict, raw_features, [TARGET, "Drug", TARGET])
features_by_metapath["D-Microbe-D"]        = _simple_metapath_aggregate(adj_dict, raw_features, [TARGET, "Microbe", TARGET])
features_by_metapath["D-Pathway-D"]        = _simple_metapath_aggregate(adj_dict, raw_features, [TARGET, "Pathway", TARGET])
features_by_metapath["D-Gene-Pathway-D"]   = _simple_metapath_aggregate(adj_dict, raw_features, [TARGET, "Gene", "Pathway", TARGET])

for k, v in features_by_metapath.items():
    print(f"   - {k}: {v.shape}")