import os
import math
import json
import random
import argparse
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.data import Data
from torch_geometric.utils import (to_undirected, remove_self_loops, coalesce,
                                   negative_sampling)
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import GATConv

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def try_read_init_emb(path_pt: str, n_expected: int) -> Optional[torch.Tensor]:
    if os.path.isfile(path_pt):
        x = torch.load(path_pt, map_location="cpu")
        if isinstance(x, torch.Tensor) and x.shape[0] == n_expected:
            print(f"[init] load precomputed node features: {path_pt} (shape={tuple(x.shape)})")
            return x.float()
        else:
            print(f"[init] {path_pt} shape mismatch: got {tuple(x.shape)} expect ({n_expected}, D).")
    return None

def encode_with_bert(names: List[str], model_name: str = "dmis-lab/biobert-base-cased-v1.1",
                     batch_size: int = 32, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> torch.Tensor:
    from transformers import AutoTokenizer, AutoModel
    print(f"[BERT] encoding {len(names)} names with {model_name} ...")
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device).eval()

    embs = []
    with torch.no_grad():
        for i in range(0, len(names), batch_size):
            batch = names[i:i+batch_size]
            enc = tok(batch, padding=True, truncation=True, max_length=32, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = mdl(**enc).last_hidden_state  # [B, T, H]
            mask = enc["attention_mask"].unsqueeze(-1).float()  # [B, T, 1]
            hid = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)  # mean pool -> [B, H]
            embs.append(hid.detach().cpu())
    X = torch.cat(embs, dim=0).float()
    print(f"[BERT] done: {tuple(X.shape)}")
    return X


class GATEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, latent_dim=64,
                 heads1=4, heads2=4, dropout=0.2):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim // heads1, heads=heads1, dropout=dropout)
        self.conv2 = GATConv(hidden_dim, latent_dim, heads=heads2, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        z = self.conv2(x, edge_index)
        return z

class DotDecoder(nn.Module):
    def forward(self, z, edge_index):
        s, t = edge_index
        return (z[s] * z[t]).sum(dim=-1)

class GraphAE(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, latent_dim=64,
                 heads1=4, heads2=4, dropout=0.2,
                 feat_recon_coef=0.0):
        super().__init__()
        self.encoder = GATEncoder(in_dim, hidden_dim, latent_dim, heads1, heads2, dropout)
        self.decoder = DotDecoder()
        self.feat_recon_coef = feat_recon_coef
        if feat_recon_coef > 0:
            self.feat_decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, in_dim)
            )
        else:
            self.feat_decoder = None

    def encode(self, x, edge_index): return self.encoder(x, edge_index)
    def decode(self, z, edge_index): return self.decoder(z, edge_index)

def get_pos_neg_train_edges(train_data, num_nodes):
    pos_edge_index = train_data.edge_index
    num_pos = pos_edge_index.size(1)
    neg_edge_index = negative_sampling(
        pos_edge_index, num_nodes=num_nodes,
        num_neg_samples=num_pos, method="sparse"
    )
    return pos_edge_index, neg_edge_index

@torch.no_grad()
def evaluate(model, full_data, split, device):
    model.eval()
    x = full_data.x.to(device)
    z = model.encode(x, split.edge_index.to(device))

    if hasattr(split, 'pos_edge_label_index') and hasattr(split, 'neg_edge_label_index'):
        pos_edge = split.pos_edge_label_index.to(device)
        neg_edge = split.neg_edge_label_index.to(device)
        pos_logits = model.decode(z, pos_edge).cpu()
        neg_logits = model.decode(z, neg_edge).cpu()
        y_true = np.concatenate([np.ones(pos_logits.numel()), np.zeros(neg_logits.numel())])
        y_score = torch.sigmoid(torch.cat([pos_logits, neg_logits], dim=0)).numpy()
    elif hasattr(split, 'edge_label_index') and hasattr(split, 'edge_label'):
        eidx = split.edge_label_index.to(device)
        logits = model.decode(z, eidx).cpu()
        y_true = split.edge_label.cpu().numpy()
        y_score = torch.sigmoid(logits).numpy()

    auc = roc_auc_score(y_true, y_score)
    ap  = average_precision_score(y_true, y_score)
    return float(auc), float(ap)

def train_loop(
    data, train_data, val_data, test_data,
    in_dim, hidden_dim=256, latent_dim=64,
    heads1=4, heads2=4, dropout=0.2,
    feat_recon_coef=0.0,
    lr=1e-3, weight_decay=1e-4,
    max_epochs=300, patience=40,
    device="cuda" if torch.cuda.is_available() else "cpu",
    outdir="process", node_names=None
):
    model = GraphAE(in_dim, hidden_dim, latent_dim, heads1, heads2, dropout, feat_recon_coef).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_auc, best_state, bad = -1.0, None, 0

    for ep in range(1, max_epochs+1):
        model.train(); opt.zero_grad()
        x = data.x.to(device)
        z = model.encode(x, train_data.edge_index.to(device))

        pos_edge, neg_edge = get_pos_neg_train_edges(train_data, data.num_nodes)
        pos_edge, neg_edge = pos_edge.to(device), neg_edge.to(device)

        pos_logits = model.decode(z, pos_edge)
        neg_logits = model.decode(z, neg_edge)

        pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
        neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
        link_loss = pos_loss + neg_loss

        if model.feat_decoder is not None and feat_recon_coef > 0:
            x_recon = model.feat_decoder(z)
            feat_loss = F.mse_loss(x_recon, x)
            loss = link_loss + feat_recon_coef * feat_loss
        else:
            feat_loss = torch.tensor(0.0, device=device)
            loss = link_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()

        val_auc, val_ap = evaluate(model, data, val_data, device)
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if ep % 10 == 0 or ep == 1:
            print(f"[{ep:03d}] loss={loss.item():.4f} link={link_loss.item():.4f} feat={feat_loss.item():.4f} | val AUC={val_auc:.4f} AP={val_ap:.4f}")

        if bad >= patience:
            print(f"Early stop at {ep}, best val AUC={best_auc:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_auc, test_ap = evaluate(model, data, test_data, device)
    print(f"[TEST] AUC={test_auc:.4f} AP={test_ap:.4f}")

    with torch.no_grad():
        model.eval()
        z_all = model.encode(data.x.to(device), data.edge_index.to(device)).cpu()

    os.makedirs(outdir, exist_ok=True)
    torch.save(z_all, os.path.join(outdir, "emb_disease_gat_gae.pt"))
    if node_names is None:
        node_names = [str(i) for i in range(z_all.size(0))]
    pd.DataFrame(z_all.numpy(), index=node_names).to_csv(os.path.join(outdir, "emb_disease_gat_gae.csv"), header=False)
    print("[save] embeddings ->", os.path.join(outdir, "emb_disease_gat_gae.csv"))

    return model, z_all, {"val_auc": best_auc, "test_auc": test_auc, "test_ap": test_ap}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node_xlsx", type=str, default="./data/Node/Disease.xlsx")
    ap.add_argument("--edge_csv", type=str, default="process/disease_edges_topk.csv")
    ap.add_argument("--outdir", type=str, default="process")
    ap.add_argument("--init_emb_pt", type=str, default="process/emb_disease_init.pt")
    ap.add_argument("--use_bert", action="store_true")
    ap.add_argument("--bert_model", type=str, default="dmis-lab/biobert-base-cased-v1.1")
    ap.add_argument("--latent_dim", type=int, default=64)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--heads1", type=int, default=4)
    ap.add_argument("--heads2", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--feat_recon_coef", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--max_epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_excel(args.node_xlsx)
    diseases = df["Disease"].astype(str).tolist()
    mesh_ids = df["MESH ID"].astype(str).tolist()
    name2id = {n: i for i, n in enumerate(diseases)}
    mid2id  = {m: i for i, m in enumerate(mesh_ids)}
    N = len(diseases)
    print(f"[nodes] N={N}")

    x = try_read_init_emb(args.init_emb_pt, N)
    if x is None and args.use_bert:
        x = encode_with_bert(diseases, model_name=args.bert_model, device=device)
        torch.save(x, os.path.join(args.outdir, "emb_disease_text_bert.pt"))
        print("[init] saved BERT features ->", os.path.join(args.outdir, "emb_disease_text_bert.pt"))
    if x is None:
        print("[init] fallback to one-hot features.")
        x = torch.eye(N, dtype=torch.float32)

    edges = pd.read_csv(args.edge_csv)

    def map_series_to_idx(series):
        s = series.astype(str)
        idx = s.map(name2id)
        if idx.isna().any():
            idx = s.map(mid2id)
        return idx

    src = map_series_to_idx(edges["source"]).to_numpy()
    dst = map_series_to_idx(edges["target"]).to_numpy()
    if np.isnan(src).any() or np.isnan(dst).any():
        bad = np.isnan(src) | np.isnan(dst)
        missing_vals = edges[bad][["source","target"]]

    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = to_undirected(edge_index, num_nodes=N)
    edge_index = coalesce(edge_index, num_nodes=N)
    print(f"[edges] E={edge_index.size(1)} (undirected, dedup)")

    data = Data(x=x, edge_index=edge_index)

    splitter = RandomLinkSplit(num_val=0.05, num_test=0.10, is_undirected=True, add_negative_train_samples=False)
    train_data, val_data, test_data = splitter(data)
    print(train_data); print(val_data); print(test_data)

    model, z_all, metrics = train_loop(
        data, train_data, val_data, test_data,
        in_dim=data.num_features,
        hidden_dim=args.hidden_dim, latent_dim=args.latent_dim,
        heads1=args.heads1, heads2=args.heads2, dropout=args.dropout,
        feat_recon_coef=args.feat_recon_coef,
        lr=args.lr, weight_decay=args.weight_decay,
        max_epochs=args.max_epochs, patience=args.patience,
        device=device, outdir=args.outdir, node_names=diseases
    )
    print("Done. metrics:", json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()

