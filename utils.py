import os, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score
)
from sklearn.model_selection import KFold
from tqdm import tqdm
from sehgnn.sehgnn_model import SeHGNN
from sklearn.metrics.pairwise import cosine_similarity
from configs import * 
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _csr_bin_power_sum(A: sp.csr_matrix, hops: int) -> sp.csr_matrix:
    X = A.copy().astype(np.int64)
    total = X.copy()
    cur = X
    for _ in range(2, hops + 1):
        cur = cur.dot(A).astype(np.int64)
        if cur.shape != A.shape:
            break
        total = (total + cur).tocsr()
    return total

def _cosine_sim_bipartite(F_left: np.ndarray, F_right: np.ndarray, chunk: int = 2048) -> np.ndarray:
    Fl = F_left / (np.linalg.norm(F_left, axis=1, keepdims=True) + 1e-9)
    Fr = F_right / (np.linalg.norm(F_right, axis=1, keepdims=True) + 1e-9)
    nL, nR = Fl.shape[0], Fr.shape[0]
    S = np.empty((nL, nR), dtype=np.float32)
    for i in range(0, nL, chunk):
        jmax = min(i + chunk, nL)
        S[i:jmax] = Fl[i:jmax].dot(Fr.T)
    return S

def augment_bipartite_by_reachability_and_similarity(A, F_left, F_right,
        hops=2, p_thr=1, sim_thr=0.6, clear_weak=False):
    A = A.tocsr().astype(np.int8)
    try:
        D = _csr_bin_power_sum(A, hops=hops).toarray()
    except Exception:
        D = A.toarray().astype(np.int64)
    S = _cosine_sim_bipartite(F_left=F_left, F_right=F_right)
    keep_mask = (D >= p_thr) & (S >= sim_thr)
    A_new = A.copy().toarray()
    if clear_weak:
        A_new = np.where(keep_mask, 1, 0).astype(np.int8)
    else:
        A_new = np.where(keep_mask, 1, A_new).astype(np.int8)
    return sp.csr_matrix(A_new)

def _kmeans_labels(X: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    return km.fit_predict(X)

def _sbm_block_probs(A: sp.csr_matrix, zl: np.ndarray, zr: np.ndarray, kL: int, kR: int) -> np.ndarray:
    A = A.tocsr()
    W = np.zeros((kL, kR), dtype=np.float64)
    cnt = np.zeros((kL, kR), dtype=np.int64)
    rows, cols = A.nonzero()
    for i, j in zip(rows, cols):
        W[zl[i], zr[j]] += 1.0
    for a in range(kL):
        na = np.sum(zl == a)
        for b in range(kR):
            nb = np.sum(zr == b)
            denom = na * nb
            cnt[a, b] = denom if denom > 0 else 1
    W = W / cnt
    return W

def _mix_graphon_block(W1: np.ndarray, W2: np.ndarray, lam: float) -> np.ndarray:
    return lam * W1 + (1 - lam) * W2

def _sample_edges_from_blocks(zl, zr, W, num_samples: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    kL, kR = W.shape
    nL, nR = len(zl), len(zr)
    mass = W / (W.sum() + 1e-12)
    flat = mass.ravel()
    if flat.sum() <= 0:
        flat = np.ones_like(flat) / flat.size
    draw = np.random.multinomial(num_samples, flat).reshape(W.shape)

    edges = []
    for a in range(kL):
        La = np.where(zl == a)[0]
        if len(La) == 0: continue
        for b in range(kR):
            Rb = np.where(zr == b)[0]
            if len(Rb) == 0 or draw[a, b] == 0: continue
            m = draw[a, b]*2
            li = rng.choice(La, size=min(m, len(La)), replace=(len(La) < m))
            rj = rng.choice(Rb, size=min(m, len(Rb)), replace=(len(Rb) < m))
            m2 = min(len(li), len(rj))
            if m2 == 0: continue
            li, rj = li[:m2], rj[:m2]
            pairs = list(zip(li, rj))
            probs = np.full(len(pairs), W[a, b])
            keep = rng.binomial(1, probs).astype(bool)
            sel = [pairs[idx] for idx, flag in enumerate(keep) if flag]
            edges.extend(sel[:draw[a, b]])
    if len(edges) == 0:
        return np.empty((0,2), dtype=np.int64)
    return np.array(edges, dtype=np.int64)

def run_graphon_augmentation(A_dd, A_dmd, drug_feats, disease_feats,
        k, lam_drug, lam_dis, ratio, seed=42):
    zl = _kmeans_labels(drug_feats, k=k, seed=seed)
    zr = _kmeans_labels(disease_feats, k=k, seed=seed)
    W_dd  = _sbm_block_probs(A_dd,  zl, zr, k, k)
    W_dmd = _sbm_block_probs(A_dmd, zl, zr, k, k)
    lam = 0.5*(lam_drug+lam_dis)
    W_mix = _mix_graphon_block(W_dd, W_dmd, lam=lam)
    n_pos_now = A_dd.nnz + A_dmd.nnz
    num_syn = int(max(1, ratio * n_pos_now))
    return _sample_edges_from_blocks(zl, zr, W_mix, num_samples=num_syn, seed=seed)

def add_edges_to_adj(A, edges_ij, shape=None):
    if edges_ij is None or len(edges_ij)==0: return A
    rows, cols = edges_ij[:,0], edges_ij[:,1]
    if shape is None: shape = A.shape
    data = np.ones(len(rows), dtype=np.int8)
    A_syn = sp.csr_matrix((data, (rows, cols)), shape=shape)
    return (A + A_syn).sign().tocsr()

def _new_model(feat_dict):
    in_dims = {k: v.shape[1] for k, v in feat_dict.items()}

    if MODEL_TYPE == "SeHGNN":
        kwargs = dict(
            metapath_dims=in_dims,
            hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM, num_classes=NUM_CLASSES
        )
        try:
            return SeHGNN(**kwargs, num_layers=NUM_GNN_LAYERS, dropout=DROPOUT).to(device)
        except TypeError:
            return SeHGNN(**kwargs).to(device)

    elif MODEL_TYPE == "HGT":
        from torch_geometric.nn import HGTConv
        class HGTWrapper(nn.Module):
            def __init__(self, in_dims, hidden_dim, out_dim, num_layers=2, heads=4):
                super().__init__()
                self.proj = nn.ModuleDict({
                    k: nn.Linear(v, hidden_dim) for k, v in in_dims.items()
                })
                self.convs = nn.ModuleList([
                    HGTConv(in_channels=hidden_dim, out_channels=hidden_dim,
                            metadata=(["Drug","Disease"], [("Drug","link","Disease")]),
                            heads=heads) for _ in range(num_layers)
                ])
                self.fc_out = nn.Linear(hidden_dim, out_dim)

            def forward(self, feats):
                h = {k: self.proj[k](x) for k, x in feats.items()}
                return self.fc_out(torch.cat(list(h.values()), dim=1))
        return HGTWrapper(in_dims, HIDDEN_DIM, OUT_DIM).to(device)

    elif MODEL_TYPE == "GAT":
        from torch_geometric.nn import GATConv
        class GATWrapper(nn.Module):
            def __init__(self, in_dims, hidden_dim, out_dim, heads=4):
                super().__init__()
                self.proj = nn.ModuleDict({
                    k: nn.Linear(v, hidden_dim) for k, v in in_dims.items()
                })
                self.gat = GATConv(hidden_dim, hidden_dim, heads=heads)
                self.fc_out = nn.Linear(hidden_dim*heads, out_dim)
            def forward(self, feats):
                h = {k: self.proj[k](x) for k, x in feats.items()}
                return self.fc_out(torch.cat(list(h.values()), dim=1))
        return GATWrapper(in_dims, HIDDEN_DIM, OUT_DIM).to(device)

    else:
        raise ValueError(f"Unknown MODEL_TYPE={MODEL_TYPE}")

def _row_normalize_csr(A: sp.csr_matrix) -> sp.csr_matrix:
    deg = np.asarray(A.sum(1)).ravel()
    deg[deg == 0.0] = 1.0
    inv = 1.0 / deg
    D_inv = sp.diags(inv, format="csr")
    return D_inv.dot(A)

def precompute_metapath_features(adj_dict, feats, metapaths):
    results = {}
    for mp in metapaths:
        start_type = mp[0]
        M = None
        for i in range(len(mp) - 1):
            src, dst = mp[i], mp[i+1]
            key = f"{src}-{dst}"
            A = _row_normalize_csr(adj_dict[key])
            M = A if M is None else M.dot(A)
        X_start = feats[start_type]
        X_meta = M.dot(X_start)
        results["-".join(mp)] = X_meta
    return results

def build_feature_tensors_for_branch(A_dd, A_dm, A_md, 
                                     drug_feats, disease_feats, microbe_feats):
    adj_dict = {
        "Drug-Disease": A_dd,
        "Drug-Microbe": A_dm,
        "Microbe-Disease": A_md,
        "Disease-Drug": A_dd.T,
        "Microbe-Drug": A_dm.T,
        "Disease-Microbe": A_md.T,
    }
    drug_feats_dict = {"Drug": drug_feats, "Disease": disease_feats, "Microbe": microbe_feats}
    drug_metapaths = [["Drug","Disease","Drug"], ["Drug","Microbe","Drug"]]
    drug_meta = precompute_metapath_features(adj_dict, drug_feats_dict, drug_metapaths)
    disease_feats_dict = {"Disease": disease_feats, "Drug": drug_feats, "Microbe": microbe_feats}
    disease_metapaths = [["Disease","Drug","Disease"], ["Disease","Microbe","Disease"]]
    disease_meta = precompute_metapath_features(adj_dict, disease_feats_dict, disease_metapaths)

    features_tensor_drug = {
        "Drug": torch.FloatTensor(drug_feats).to(device),
        "Drug-Disease-Drug": torch.FloatTensor(drug_meta["Drug-Disease-Drug"]).to(device),
        "Drug-Microbe-Drug": torch.FloatTensor(drug_meta["Drug-Microbe-Drug"]).to(device),
    }
    features_tensor_disease = {
        "Disease": torch.FloatTensor(disease_feats).to(device),
        "Disease-Drug-Disease": torch.FloatTensor(disease_meta["Disease-Drug-Disease"]).to(device),
        "Disease-Microbe-Disease": torch.FloatTensor(disease_meta["Disease-Microbe-Disease"]).to(device),
    }
    return features_tensor_drug, features_tensor_disease

class LinkPredictor(nn.Module):
    def forward(self, h_drug, h_disease, edges):
        d_idx, dis_idx = edges[:,0], edges[:,1]
        return (h_drug[d_idx] * h_disease[dis_idx]).sum(dim=1)


def cross_view_contrastive_sampled(h_dict, temperature=0.5, num_samples=1024):
    views = list(h_dict.keys())
    N = h_dict[views[0]].size(0)
    device_local = h_dict[views[0]].device

    if N > num_samples:
        idx = torch.randperm(N, device=device_local)[:num_samples]
    else:
        idx = torch.arange(N, device=device_local)

    loss = 0
    for i in range(len(views)):
        for j in range(i+1, len(views)):
            z1, z2 = h_dict[views[i]][idx], h_dict[views[j]][idx]
            z1 = F.normalize(z1, dim=1)
            z2 = F.normalize(z2, dim=1)
            logits = torch.matmul(z1, z2.T) / temperature   # [B,B]
            labels = torch.arange(z1.size(0), device=device_local)
            loss += F.cross_entropy(logits, labels)
    return loss

def intra_view_contrastive_sampled(h, labels, temperature=0.5, num_samples=1024, num_negatives=512):
    """
    h: [N,d], labels: [N]
    """
    N = h.size(0)
    device_local = h.device
    h = F.normalize(h, dim=1)

    if N > num_samples:
        idx = torch.randperm(N, device=device_local)[:num_samples]
    else:
        idx = torch.arange(N, device=device_local)

    loss_all = []
    for i in idx:
        anchor = h[i]
        label = labels[i]
        pos_mask = (labels == label).nonzero(as_tuple=True)[0]
        pos_mask = pos_mask[pos_mask != i]
        if len(pos_mask) == 0:
            continue
        pos_idx = pos_mask[torch.randint(0, len(pos_mask), (1,))]
        pos = h[pos_idx]
        neg_mask = (labels != label).nonzero(as_tuple=True)[0]
        if len(neg_mask) > num_negatives:
            neg_idx = neg_mask[torch.randint(0, len(neg_mask), (num_negatives,))]
        else:
            neg_idx = neg_mask
        if len(neg_idx) == 0:
            continue
        neg = h[neg_idx]

        sim_pos = torch.matmul(anchor, pos.T) / temperature
        sim_neg = torch.matmul(anchor, neg.T) / temperature
        logits = torch.cat([sim_pos, sim_neg], dim=0).unsqueeze(0)
        target = torch.zeros(1, dtype=torch.long, device=device_local)
        loss_all.append(F.cross_entropy(logits, target))
    if len(loss_all) == 0:
        return torch.tensor(0.0, device=device_local, requires_grad=True)
    return torch.stack(loss_all).mean()

def _adaptive_negative_sampler(
    epoch:int, T:int,
    train_pos_edges:np.ndarray,
    h_drug_final:torch.Tensor, 
    h_disease_final:torch.Tensor
):
    if NEG_SAMPLING_MODE != "adaptive":
        num_t = max(1, int(NEG_BETA * (epoch / T) * len(train_pos_edges)))
        return np.array(random.sample(neg_candidates_global, k=min(num_t, len(neg_candidates_global))), dtype=np.int64)

    num_t = max(1, int(NEG_BETA * (epoch / T) * len(train_pos_edges)))
    cand_size = min(len(neg_candidates_global), max(num_t * NEG_CAND_MULT, num_t+1))

    cand_pairs = random.sample(neg_candidates_global, k=cand_size)
    cand_pairs = np.array(cand_pairs, dtype=np.int64)
    d_idx = cand_pairs[:,0]
    dis_idx = cand_pairs[:,1]

    R_ns = np.empty(cand_pairs.shape[0], dtype=np.float32)
    P = P_struct.tocsr()
    for k in range(cand_pairs.shape[0]):
        i, j = int(d_idx[k]), int(dis_idx[k])
        row_start, row_end = P.indptr[i], P.indptr[i+1]
        row_idx = P.indices[row_start:row_end]
        row_data = P.data[row_start:row_end]
        hit = np.where(row_idx == j)[0]
        p_ij = 0 if len(hit) == 0 else int(row_data[hit[0]])
        R_ns[k] = 1.0 / (1.0 + float(p_ij))

    with torch.no_grad():
        z_d = F.normalize(h_drug_final[d_idx], dim=1)
        z_s = F.normalize(h_disease_final[dis_idx], dim=1)
        cos = (z_d * z_s).sum(dim=1).clamp(-1,1).cpu().numpy()
    R_fs = 1.0 - cos

    C = cand_pairs.shape[0]
    top_struct = max(1, int(NEG_STRUCT_TOP_PCT * C))
    top_sem    = max(1, int(NEG_SEM_TOP_PCT * C))

    idx_struct = np.argsort(-R_ns)[:top_struct]
    idx_sem    = np.argsort(-R_fs)[:top_sem]

    union_idx = np.unique(np.concatenate([idx_struct, idx_sem], axis=0))
    if len(union_idx) > num_t:
        R_mix = np.maximum(R_ns[union_idx], R_fs[union_idx])
        order = np.argsort(-R_mix)[:num_t]
        sel = union_idx[order]
    else:
        sel = union_idx
        if len(sel) < num_t:
            rest_idx_pool = np.setdiff1d(np.arange(C), sel, assume_unique=False)
            if len(rest_idx_pool) > 0:
                supplement = np.random.choice(rest_idx_pool, size=min(num_t-len(sel), len(rest_idx_pool)), replace=False)
                sel = np.concatenate([sel, supplement], axis=0)

    return cand_pairs[sel]

def run_coldstart_for_fold(fold, ratio_list, topk,
                           A_dd_raw, A_dm_raw, A_md_raw,
                           drug_feats, disease_feats, microbe_feats):
    print(f"[Fold {fold}] Cold Start Experiments ...")
    for ratio in ratio_list:
        exp_dir = os.path.join(RESULT_DIR, f"fold{fold}", f"coldstart_{int(ratio*100)}")
        os.makedirs(exp_dir, exist_ok=True)
        all_drugs = np.arange(drug_feats.shape[0])
        n_cold = int(len(all_drugs) * ratio)
        cold_drugs = np.random.choice(all_drugs, size=n_cold, replace=False)
        noncold_drugs = np.setdiff1d(all_drugs, cold_drugs)
        A_dd_cs = A_dd_raw.copy().tolil()
        for d in cold_drugs:
            A_dd_cs[d,:] = 0
            A_dd_cs[:,d] = 0
        A_dd_cs = A_dd_cs.tocsr()

        A_dm_cs = A_dm_raw.copy().tolil()
        for d in cold_drugs:
            A_dm_cs[d,:] = 0
        A_dm_cs = A_dm_cs.tocsr()
        sim = cosine_similarity(drug_feats[cold_drugs], drug_feats[noncold_drugs])
        for i, d in enumerate(cold_drugs):
            top_idx = np.argsort(-sim[i])[:topk]
            neigh = noncold_drugs[top_idx]
            for n in neigh:
                A_dd_cs[d, n] = 1
                A_dd_cs[n, d] = 1
        A_dd_cs = A_dd_cs.tocsr()
        feat_drug, feat_dis = build_feature_tensors_for_branch(
            A_dd=A_dd_cs, A_dm=A_dm_cs, A_md=A_md_raw,
            drug_feats=drug_feats, disease_feats=disease_feats, microbe_feats=microbe_feats
        )
        def _new_sehgnn(feat_dict):
            kwargs = dict(
                metapath_dims={k:v.shape[1] for k,v in feat_dict.items()},
                hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM, num_classes=NUM_CLASSES
            )
            try:
                return SeHGNN(**kwargs, num_layers=NUM_GNN_LAYERS, dropout=DROPOUT).to(device)
            except TypeError:
                return SeHGNN(**kwargs).to(device)

        model_drug = _new_sehgnn(feat_drug)
        model_disease = _new_sehgnn(feat_dis)

        proj_drug = nn.Linear(sum(v.shape[1] for v in feat_drug.values()), FUSION_OUT_DIM).to(device)
        proj_disease = nn.Linear(sum(v.shape[1] for v in feat_dis.values()), FUSION_OUT_DIM).to(device)
        predictor = LinkPredictor().to(device)
        dropout_layer = nn.Dropout(DROPOUT)

        optimizer = torch.optim.Adam(
            list(model_drug.parameters()) +
            list(model_disease.parameters()) +
            list(proj_drug.parameters()) +
            list(proj_disease.parameters()) +
            list(predictor.parameters()), lr=1e-3, weight_decay=WEIGHT_DECAY
        )

        pos_edges = np.vstack(A_dd_cs.nonzero()).T
        all_pairs = set((i,j) for i in range(drug_feats.shape[0]) for j in range(disease_feats.shape[0]))
        pos_set = set(map(tuple, pos_edges.tolist()))
        neg_edges = list(all_pairs - pos_set)
        random.shuffle(neg_edges)
        neg_edges = np.array(neg_edges[:len(pos_edges)], dtype=np.int64)

        edges = np.vstack([pos_edges, neg_edges])
        labels = np.array([1]*len(pos_edges) + [0]*len(neg_edges), dtype=np.int64)

        idx = np.random.permutation(len(edges))
        split = int(0.8 * len(edges))
        train_edges, test_edges = edges[idx[:split]], edges[idx[split:]]
        train_labels, test_labels = labels[idx[:split]], labels[idx[split:]]

        T_cs = COLDSTART_EPOCHS
        for epoch in range(1, T_cs+1):
            model_drug.train(); model_disease.train()
            h_drug = model_drug(feat_drug)
            h_dis  = model_disease(feat_dis)

            fused_drug = torch.cat(list(feat_drug.values()), dim=1)
            fused_dis  = torch.cat(list(feat_dis.values()), dim=1)
            h_drug_final = proj_drug(dropout_layer(fused_drug))
            h_dis_final  = proj_disease(dropout_layer(fused_dis))

            train_edges_t = torch.LongTensor(train_edges).to(device)
            train_labels_t = torch.FloatTensor(train_labels).to(device)
            scores = predictor(h_drug_final, h_dis_final, train_edges_t)
            loss = F.binary_cross_entropy_with_logits(scores, train_labels_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                metrics, _, _ = evaluate(h_drug_final, h_dis_final, test_edges, test_labels)
                print(f"[Fold {fold} | ColdStart {ratio:.2f} | epoch={epoch}] "
                      f"Loss={loss.item():.4f} AUC={metrics['AUC']:.4f}")

        with torch.no_grad():
            metrics_train, preds_train, probs_train = evaluate(h_drug_final, h_dis_final, train_edges, train_labels)
            metrics_test, preds_test, probs_test = evaluate(h_drug_final, h_dis_final, test_edges, test_labels)

        json.dump(metrics_train, open(os.path.join(exp_dir, "train_metrics.json"),"w"), indent=2)
        json.dump(metrics_test, open(os.path.join(exp_dir, "test_metrics.json"),"w"), indent=2)

        pd.DataFrame({
            "drug": train_edges[:,0], "disease": train_edges[:,1],
            "label": train_labels, "pred": preds_train, "prob": probs_train
        }).to_csv(os.path.join(exp_dir,"train_pred.csv"), index=False)

        pd.DataFrame({
            "drug": test_edges[:,0], "disease": test_edges[:,1],
            "label": test_labels, "pred": preds_test, "prob": probs_test
        }).to_csv(os.path.join(exp_dir,"test_pred.csv"), index=False)

        print(f"[Fold {fold} | Cold Start {ratio:.2f}] Results saved to {exp_dir}")

