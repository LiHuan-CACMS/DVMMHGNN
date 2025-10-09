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
from utils import * 
import warnings
warnings.filterwarnings("ignore")
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[info] Using device: {device}")


def load_json(path): return json.load(open(path, "r"))
def load_npy(path): return np.load(path)
def load_adj(path): return sp.load_npz(path)
drug_feats    = load_npy(os.path.join(PROC_DIR, "drug_feats.npy"))
disease_feats = load_npy(os.path.join(PROC_DIR, "disease_feats.npy"))
microbe_feats = load_npy(os.path.join(PROC_DIR, "microbe_feats.npy"))
drug_idx2id    = load_json(os.path.join(PROC_DIR, "drug_id_map.json"))
disease_idx2id = load_json(os.path.join(PROC_DIR, "disease_id_map.json"))
microbe_idx2id = load_json(os.path.join(PROC_DIR, "microbe_id_map.json"))
n_drug, n_disease, n_microbe = len(drug_idx2id), len(disease_idx2id), len(microbe_idx2id)
A_dd_raw = load_adj(os.path.join(PROC_DIR, "adj_Drug-Disease.npz"))
A_dm_raw = load_adj(os.path.join(PROC_DIR, "adj_Drug-Microbe.npz"))
A_md_raw = load_adj(os.path.join(PROC_DIR, "adj_Microbe-Disease.npz"))


def evaluate(h_drug, h_disease, edges, labels):
    edges_t = torch.LongTensor(edges).to(device)
    scores = predictor(h_drug, h_disease, edges_t)
    probs = torch.sigmoid(scores).detach().cpu().numpy()
    preds = (probs > 0.5).astype(int)
    y_true = labels
    metrics = {
        "AUC":  roc_auc_score(y_true, probs),
        "AUPR": average_precision_score(y_true, probs),
        "F1":   f1_score(y_true, preds),
        "Precision": precision_score(y_true, preds),
        "Recall": recall_score(y_true, preds),
        "Accuracy": accuracy_score(y_true, preds),
    }
    return metrics, preds, probs


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



if __name__ == "__main__":
    if AUG_STRUCT_ENABLE:
        print("[aug] Structural-level augmentation ...")
        A_dd_struct = augment_bipartite_by_reachability_and_similarity(
            A=A_dd_raw,
            F_left=drug_feats, F_right=disease_feats,
            hops=AUG_STRUCT_HOPS, p_thr=AUG_STRUCT_P_THR,
            sim_thr=AUG_STRUCT_SIM_THR, clear_weak=AUG_STRUCT_CLEAR_WEAK
        )
    else:
        A_dd_struct = A_dd_raw.copy()
    A_dmd_struct = (A_dm_raw.dot(A_md_raw)).tocsr()

    if AUG_META_ENABLE:
        print("[aug] Metapath-level (graphon) augmentation ...")
        A_dd_meta = A_dd_raw.copy()
        A_dmd_meta = (A_dm_raw.dot(A_md_raw)).tocsr()
        extra_edges_graphon = run_graphon_augmentation(
            A_dd=A_dd_meta, A_dmd=A_dmd_meta,
            drug_feats=drug_feats, disease_feats=disease_feats,
            k=AUG_META_K, lam_drug=AUG_META_LAMBDA_DRUG,
            lam_dis=AUG_META_LAMBDA_DIS, ratio=AUG_META_SYN_RATIO, seed=seed
        )
        if AUG_META_ADD_TO_GRAPH and len(extra_edges_graphon) > 0:
            print(f"[aug] add {len(extra_edges_graphon)} synthetic edges into A_dd_meta")
            A_dd_meta = add_edges_to_adj(A_dd_meta, extra_edges_graphon, shape=A_dd_meta.shape)
    else:
        A_dd_meta = A_dd_raw.copy()
    A_dmd_meta = (A_dm_raw.dot(A_md_raw)).tocsr()


    direct_edges_struct = np.vstack(A_dd_struct.nonzero()).T
    direct_edges_meta   = np.vstack(A_dd_meta.nonzero()).T

    dmd_edges_struct = np.vstack(A_dmd_struct.nonzero()).T
    dmd_edges_meta   = np.vstack(A_dmd_meta.nonzero()).T

    pos_edges = np.vstack([
        direct_edges_struct,
        direct_edges_meta,
        dmd_edges_struct,
        dmd_edges_meta
    ])
    if AUG_META_ENABLE and not AUG_META_ADD_TO_GRAPH:
        if 'extra_edges_graphon' in locals() and len(extra_edges_graphon) > 0:
            pos_edges = np.vstack([pos_edges, extra_edges_graphon])
    pos_edges = np.unique(pos_edges, axis=0)
    all_pairs = set((i, j) for i in range(n_drug) for j in range(n_disease))
    pos_set = set(map(tuple, pos_edges.tolist()))
    neg_candidates_global = list(all_pairs - pos_set)
    random.shuffle(neg_candidates_global)
    neg_edges_uniform = np.array(neg_candidates_global[:len(pos_edges)], dtype=np.int64)
    edges_uniform = np.vstack([pos_edges, neg_edges_uniform])
    labels_uniform = np.array([1]*len(pos_edges) + [0]*len(neg_edges_uniform), dtype=np.int64)
    print(f"[data] pos={len(pos_edges)}, neg(uniform)={len(neg_edges_uniform)}, total(uniform)={len(edges_uniform)}")
    A_struct_base = (A_dd_raw + (A_dm_raw.dot(A_md_raw))).tocsr()
    try:
        P_struct = _csr_bin_power_sum(A_struct_base, hops=max(1, NEG_STRUCT_HOPS)).astype(np.int64)
    except Exception:
        P_struct = A_struct_base.copy().astype(np.int64)
    pos_dict_by_drug = {}
    for i, j in pos_edges:
        pos_dict_by_drug.setdefault(int(i), set()).add(int(j))
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(kf.split(edges_uniform), 1):
        print(f"\n========== Fold {fold} ==========")
        fold_dir = os.path.join(RESULT_DIR, f"fold{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        train_edges_all, test_edges_all = edges_uniform[train_idx], edges_uniform[test_idx]
        train_labels_all, test_labels_all = labels_uniform[train_idx], labels_uniform[test_idx]

        mask_pos_train = (train_labels_all == 1)
        train_pos_edges = train_edges_all[mask_pos_train]
        test_edges, test_labels = test_edges_all, test_labels_all

        feat_drug_struct, feat_dis_struct = build_feature_tensors_for_branch(
            A_dd=A_dd_struct, A_dm=A_dm_raw, A_md=A_md_raw,
            drug_feats=drug_feats, disease_feats=disease_feats, microbe_feats=microbe_feats
        )
        feat_drug_meta, feat_dis_meta = build_feature_tensors_for_branch(
            A_dd=A_dd_meta, A_dm=A_dm_raw, A_md=A_md_raw,
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

        model_drug_struct = _new_sehgnn(feat_drug_struct)
        model_disease_struct = _new_sehgnn(feat_dis_struct)
        model_drug_meta = _new_sehgnn(feat_drug_meta)
        model_disease_meta = _new_sehgnn(feat_dis_meta)

        with torch.no_grad():
            tmp_drug_struct = model_drug_struct(feat_drug_struct)       # [n_drug, d1]
            tmp_drug_meta   = model_drug_meta(feat_drug_meta)           # [n_drug, d2]
            tmp_dis_struct  = model_disease_struct(feat_dis_struct)     # [n_disease, d3]
            tmp_dis_meta    = model_disease_meta(feat_dis_meta)         # [n_disease, d4]
            in_dim_drug = tmp_drug_struct.size(1) + tmp_drug_meta.size(1)
            in_dim_dis  = tmp_dis_struct.size(1)  + tmp_dis_meta.size(1)
        proj_drug = nn.Linear(in_dim_drug, FUSION_OUT_DIM).to(device)
        proj_disease = nn.Linear(in_dim_dis, FUSION_OUT_DIM).to(device)
        predictor = LinkPredictor().to(device)
        dropout_layer = nn.Dropout(DROPOUT)
        alpha = nn.Parameter(torch.tensor(0.1, device=device))  # cross-view
        beta  = nn.Parameter(torch.tensor(0.1, device=device))  # intra-view
        optimizer = torch.optim.Adam(
            list(model_drug_struct.parameters()) +
            list(model_disease_struct.parameters()) +
            list(model_drug_meta.parameters()) +
            list(model_disease_meta.parameters()) +
            list(proj_drug.parameters()) +
            list(proj_disease.parameters()) +
            list(predictor.parameters()) +
            [alpha, beta],
            lr=LR, weight_decay=WEIGHT_DECAY
        )
        for epoch in tqdm(range(1, NUM_EPOCHS + 1), desc=f"Fold {fold} Training"):
            model_drug_struct.train(); model_disease_struct.train()
            model_drug_meta.train();   model_disease_meta.train()
            h_drug_struct = model_drug_struct(feat_drug_struct)           # [n_drug, d1]
            h_disease_struct = model_disease_struct(feat_dis_struct)      # [n_disease, d3]
            h_drug_meta = model_drug_meta(feat_drug_meta)                 # [n_drug, d2]
            h_disease_meta = model_disease_meta(feat_dis_meta)            # [n_disease, d4]
            h_drug_fusion = torch.cat([h_drug_struct, h_drug_meta], dim=1)         # [n_drug, in_dim_drug]
            h_disease_fusion = torch.cat([h_disease_struct, h_disease_meta], dim=1)# [n_disease, in_dim_dis]
            h_drug_final = proj_drug(dropout_layer(h_drug_fusion))             # [n_drug, FUSION_OUT_DIM]
            h_disease_final = proj_disease(dropout_layer(h_disease_fusion))    # [n_disease, FUSION_OUT_DIM]
            if NEG_SAMPLING_MODE == "uniform":
                mask_neg_train = (train_labels_all == 0)
                train_neg_edges_epoch = train_edges_all[mask_neg_train]
                k_neg = min(len(train_pos_edges), len(train_neg_edges_epoch))
                if k_neg < len(train_neg_edges_epoch):
                    idx_sel = np.random.choice(len(train_neg_edges_epoch), size=k_neg, replace=False)
                    train_neg_edges_epoch = train_neg_edges_epoch[idx_sel]
            else:
                train_neg_edges_epoch = _adaptive_negative_sampler(
                    epoch=epoch, T=NUM_EPOCHS,
                    train_pos_edges=train_pos_edges,
                    h_drug_final=h_drug_final.detach(),
                    h_disease_final=h_disease_final.detach()
                )
            train_edges_epoch = np.vstack([train_pos_edges, train_neg_edges_epoch])
            train_labels_epoch = np.array([1]*len(train_pos_edges) + [0]*len(train_neg_edges_epoch), dtype=np.float32)
            perm = np.random.permutation(len(train_edges_epoch))
            train_edges_epoch = train_edges_epoch[perm]
            train_labels_epoch = train_labels_epoch[perm]
            train_edges_t = torch.LongTensor(train_edges_epoch).to(device)
            train_labels_t = torch.FloatTensor(train_labels_epoch).to(device)
            scores = predictor(h_drug_final, h_disease_final, train_edges_t)
            loss_bce = F.binary_cross_entropy_with_logits(scores, train_labels_t)
            proj_dd_struct = model_drug_struct.projectors["Drug-Disease-Drug"](feat_drug_struct["Drug-Disease-Drug"])
            proj_dm_struct = model_drug_struct.projectors["Drug-Microbe-Drug"](feat_drug_struct["Drug-Microbe-Drug"])
            loss_cross_struct = cross_view_contrastive_sampled({
                "Drug-DD-Drug(struct)": proj_dd_struct,
                "Drug-DM-Drug(struct)": proj_dm_struct,
            }, num_samples=256, temperature=CONTRASTIVE_TAU)
            proj_dd_meta = model_drug_meta.projectors["Drug-Disease-Drug"](feat_drug_meta["Drug-Disease-Drug"])
            proj_dm_meta = model_drug_meta.projectors["Drug-Microbe-Drug"](feat_drug_meta["Drug-Microbe-Drug"])
            loss_cross_meta = cross_view_contrastive_sampled({
                "Drug-DD-Drug(meta)": proj_dd_meta,
                "Drug-DM-Drug(meta)": proj_dm_meta,
            }, num_samples=256, temperature=CONTRASTIVE_TAU)
            loss_cross = loss_cross_struct + loss_cross_meta
            dis_idx_epoch = train_edges_t[:, 1]
            node_labels_epoch = train_labels_t 
            loss_intra = intra_view_contrastive_sampled(
                h_disease_final[dis_idx_epoch], node_labels_epoch.long(),
                num_samples=256, num_negatives=256, temperature=CONTRASTIVE_TAU
            )
            loss = loss_bce + F.softplus(alpha) * loss_cross + F.softplus(beta) * loss_intra
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if SAVE_EMBED_PER_EPOCH:
                emb_dir = os.path.join(fold_dir, "embeddings")
                os.makedirs(emb_dir, exist_ok=True)
                drug_idx_epoch = train_edges_t[:, 0]
                dis_idx_epoch  = train_edges_t[:, 1]
                y_epoch = train_labels_t.detach().cpu().numpy()
                h_drug_epoch = h_drug_final[drug_idx_epoch].detach().cpu().numpy()
                h_dis_epoch  = h_disease_final[dis_idx_epoch].detach().cpu().numpy()
                h_pair_epoch = np.concatenate([h_drug_epoch, h_dis_epoch], axis=1)
                np.save(os.path.join(emb_dir, f"epoch{epoch}_label0.npy"), h_pair_epoch[y_epoch == 0])
                np.save(os.path.join(emb_dir, f"epoch{epoch}_label1.npy"), h_pair_epoch[y_epoch == 1])
            if epoch % 20 == 0:
                model_drug_struct.eval(); model_disease_struct.eval()
                model_drug_meta.eval();   model_disease_meta.eval()
                with torch.no_grad():
                    h_drug_struct = model_drug_struct(feat_drug_struct)
                    h_disease_struct = model_disease_struct(feat_dis_struct)
                    h_drug_meta = model_drug_meta(feat_drug_meta)
                    h_disease_meta = model_disease_meta(feat_dis_meta)
                    h_drug_final = proj_drug(dropout_layer(torch.cat([h_drug_struct, h_drug_meta], dim=1)))
                    h_disease_final = proj_disease(dropout_layer(torch.cat([h_disease_struct, h_disease_meta], dim=1)))
                    metrics, _, _ = evaluate(h_drug_final, h_disease_final, test_edges, test_labels)
                    print(f"[Fold {fold} | Epoch {epoch}] "
                        f"Loss={loss.item():.4f} | AUC={metrics['AUC']:.4f} "
                        f"| alpha={alpha.item():.4f}, beta={beta.item():.4f}")
        torch.save(model_drug_struct.state_dict(),   os.path.join(fold_dir, "model_drug_struct.pt"))
        torch.save(model_disease_struct.state_dict(),os.path.join(fold_dir, "model_disease_struct.pt"))
        torch.save(model_drug_meta.state_dict(),     os.path.join(fold_dir, "model_drug_meta.pt"))
        torch.save(model_disease_meta.state_dict(),  os.path.join(fold_dir, "model_disease_meta.pt"))
        torch.save(proj_drug.state_dict(),           os.path.join(fold_dir, "proj_drug.pt"))
        torch.save(proj_disease.state_dict(),        os.path.join(fold_dir, "proj_disease.pt"))
        model_drug_struct.eval(); model_disease_struct.eval()
        model_drug_meta.eval();   model_disease_meta.eval()
        with torch.no_grad():
            h_drug_struct = model_drug_struct(feat_drug_struct)
            h_disease_struct = model_disease_struct(feat_dis_struct)
            h_drug_meta = model_drug_meta(feat_drug_meta)
            h_disease_meta = model_disease_meta(feat_dis_meta)
            h_drug_final = proj_drug(dropout_layer(torch.cat([h_drug_struct, h_drug_meta], dim=1)))
            h_disease_final = proj_disease(dropout_layer(torch.cat([h_disease_struct, h_disease_meta], dim=1)))
        train_metrics, train_preds, train_probs = evaluate(h_drug_final, h_disease_final, train_edges_all, train_labels_all)
        test_metrics,  test_preds,  test_probs  = evaluate(h_drug_final, h_disease_final, test_edges,  test_labels)
        json.dump(train_metrics, open(os.path.join(fold_dir, "train_metrics.json"),"w"), indent=2)
        json.dump(test_metrics,  open(os.path.join(fold_dir, "test_metrics.json"),"w"),  indent=2)
        pd.DataFrame({
            "drug": train_edges_all[:,0], "disease": train_edges_all[:,1],
            "label": train_labels_all, "pred": train_preds, "prob": train_probs
        }).to_csv(os.path.join(fold_dir,"train_pred.csv"), index=False)
        pd.DataFrame({
            "drug": test_edges[:,0], "disease": test_edges[:,1],
            "label": test_labels, "pred": test_preds, "prob": test_probs
        }).to_csv(os.path.join(fold_dir,"test_pred.csv"), index=False)
        print(f"[Fold {fold}] Results saved to {fold_dir}")
        import joblib
        feat_save_dir = os.path.join(fold_dir, "features")
        os.makedirs(feat_save_dir, exist_ok=True)
        joblib.dump(feat_drug_struct,   os.path.join(feat_save_dir, "feat_drug_struct.pkl"))
        joblib.dump(feat_dis_struct,    os.path.join(feat_save_dir, "feat_dis_struct.pkl"))
        joblib.dump(feat_drug_meta,     os.path.join(feat_save_dir, "feat_drug_meta.pkl"))
        joblib.dump(feat_dis_meta,      os.path.join(feat_save_dir, "feat_dis_meta.pkl"))
        json.dump(disease_idx2id, open(os.path.join(feat_save_dir, "disease_idx2id.json"), "w"), indent=2)
        with open(os.path.join(feat_save_dir, "n_disease.txt"), "w") as f:
            f.write(str(n_disease))
        run_coldstart_for_fold(fold, [0.1, 0.3, 0.5], topk=5,
                        A_dd_raw=A_dd_raw, A_dm_raw=A_dm_raw, A_md_raw=A_md_raw,
                        drug_feats=drug_feats, disease_feats=disease_feats, microbe_feats=microbe_feats)
