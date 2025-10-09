import os
import math
import random
import argparse
from typing import Tuple, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, remove_self_loops, coalesce, negative_sampling
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import GATConv


def build_topk_graph_from_similarity(
    tsv_path: str,
    topk: int = 15,
    min_sim: float = None, 
    outdir: str = "process"
) -> Tuple[pd.DataFrame, Dict[str, int], np.ndarray]:
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_table(tsv_path, sep="\t")
    df = df.rename(columns={df.columns[0]: "GO_term1",
                            df.columns[1]: "GO_term2",
                            df.columns[2]: "similarity"})
    df["GO_term1"] = df["GO_term1"].astype(str)
    df["GO_term2"] = df["GO_term2"].astype(str)
    df["similarity"] = pd.to_numeric(df["similarity"], errors="coerce").fillna(0.0)

    go_names = pd.Index(pd.unique(pd.concat([df["GO_term1"], df["GO_term2"]], ignore_index=True))).tolist()
    go_names.sort()
    name2id = {n: i for i, n in enumerate(go_names)}
    id2name = np.array(go_names, dtype=object)

    if min_sim is not None:
        df = df[df["similarity"] >= float(min_sim)]

    df["src_id"] = df["GO_term1"].map(name2id)
    df["tgt_id"] = df["GO_term2"].map(name2id)

    df_sorted = df.sort_values(["GO_term1", "similarity"], ascending=[True, False])
    df_topk_1 = df_sorted.groupby("GO_term1", as_index=False).head(topk)

    df_swap = df.rename(columns={"GO_term1": "GO_term2", "GO_term2": "GO_term1",
                                 "src_id": "tgt_id", "tgt_id": "src_id"}).copy()
    df_sorted2 = df_swap.sort_values(["GO_term1", "similarity"], ascending=[True, False])
    df_topk_2 = df_sorted2.groupby("GO_term1", as_index=False).head(topk)

    df_uni = pd.concat([
        df_topk_1[["GO_term1", "GO_term2", "src_id", "tgt_id", "similarity"]],
        df_topk_2.rename(columns={"GO_term1": "GO_term2", "GO_term2": "GO_term1",
                                  "src_id": "tgt_id", "tgt_id": "src_id"})[["GO_term1", "GO_term2", "src_id", "tgt_id", "similarity"]],
    ], ignore_index=True)

    u = df_uni["src_id"].to_numpy()
    v = df_uni["tgt_id"].to_numpy()
    a = np.minimum(u, v)
    b = np.maximum(u, v)
    key = a * len(go_names) + b

    idx = np.argsort(key, kind="mergesort")
    key_sorted = key[idx]
    sim_sorted = df_uni["similarity"].to_numpy()[idx]
    a_sorted = a[idx]
    b_sorted = b[idx]

    keep = np.ones_like(idx, dtype=bool)
    unique_keys, start_idx = np.unique(key_sorted, return_index=True)
    for i in range(len(unique_keys)):
        s = start_idx[i]
        e = start_idx[i+1] if i+1 < len(unique_keys) else len(key_sorted)
        j = s + int(np.argmax(sim_sorted[s:e]))
        keep[s:e] = False
        keep[j] = True

    a_kept = a_sorted[keep]
    b_kept = b_sorted[keep]
    sim_kept = sim_sorted[keep]

    edges_undirected = pd.DataFrame({
        "source": [id2name[i] for i in a_kept],
        "target": [id2name[j] for j in b_kept],
        "score": sim_kept
    })

    edges_csv = os.path.join(outdir, "go_edges_topk_undirected.csv")
    map_csv = os.path.join(outdir, "go_name2id.csv")
    edges_undirected.to_csv(edges_csv, index=False)
    pd.Series(name2id).rename("id").to_csv(map_csv, header=True)
    print(f"Saved edges: {edges_csv}  | nodes: {len(go_names)}")

    return edges_undirected, name2id, id2name


class GATEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, latent_dim=64,
                 heads1=4, heads2=4, dropout=0.2):
        super().__init__()
        Conv = GATConv
        self.conv1 = Conv(in_dim, hidden_dim // heads1, heads=heads1, dropout=dropout)
        self.conv2 = Conv(hidden_dim, latent_dim, heads=heads2, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        z = self.conv2(x, edge_index)
        return z

class DotDecoder(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, z, edge_index):
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=-1)

class GraphAE(nn.Module):
    def __init__(self, num_nodes, emb_dim=128,
                 hidden_dim=256, latent_dim=64,
                 heads1=4, heads2=4, dropout=0.2,
                 feat_recon_coef=0.0):
        super().__init__()
        self.embed = nn.Embedding(num_nodes, emb_dim)
        nn.init.xavier_uniform_(self.embed.weight)

        self.encoder = GATEncoder(emb_dim, hidden_dim, latent_dim, heads1, heads2, dropout)
        self.decoder = DotDecoder()

        self.feat_recon_coef = feat_recon_coef
        if feat_recon_coef > 0:
            self.feat_decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, emb_dim)
            )
        else:
            self.feat_decoder = None

    def encode(self, node_ids_long, edge_index):
        x = self.embed(node_ids_long)  # [N, emb_dim]
        z = self.encoder(x, edge_index)
        return z

    def decode(self, z, edge_index):
        return self.decoder(z, edge_index)


@torch.no_grad()
def evaluate(model: GraphAE, data_split: Data, device: str, node_ids_long: torch.Tensor):
    model.eval()
    z = model.encode(node_ids_long.to(device), data_split.edge_index.to(device))
    if hasattr(data_split, 'pos_edge_label_index') and hasattr(data_split, 'neg_edge_label_index'):
        pos_edge = data_split.pos_edge_label_index.to(device)
        neg_edge = data_split.neg_edge_label_index.to(device)
        pos_logits = model.decode(z, pos_edge).detach().cpu()
        neg_logits = model.decode(z, neg_edge).detach().cpu()
        y_true = np.concatenate([
            np.ones(pos_logits.numel(), dtype=np.int64),
            np.zeros(neg_logits.numel(), dtype=np.int64)
        ])
        y_score = torch.sigmoid(torch.cat([pos_logits, neg_logits], dim=0)).numpy()
    elif hasattr(data_split, 'edge_label_index') and hasattr(data_split, 'edge_label'):
        logits = model.decode(z, data_split.edge_label_index.to(device)).detach().cpu()
        y_true = data_split.edge_label.detach().cpu().numpy()
        y_score = torch.sigmoid(logits).numpy()
    else:
        raise AttributeError("evaluate() 缺少边标签字段。")

    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    return float(auc), float(ap)


def get_pos_neg_train_edges(train_data: Data, num_nodes: int):
    pos_edge_index = train_data.edge_index
    num_pos = pos_edge_index.size(1)
    neg_edge_index = negative_sampling(
        pos_edge_index, num_nodes=num_nodes, num_neg_samples=num_pos, method='sparse'
    )
    return pos_edge_index, neg_edge_index


def train_loop(
    data: Data, train_data: Data, val_data: Data,
    num_nodes: int,
    emb_dim: int = 128, hidden_dim: int = 256, latent_dim: int = 64,
    heads1: int = 4, heads2: int = 4, dropout: float = 0.2,
    feat_recon_coef: float = 0.0,
    lr: float = 1e-3, weight_decay: float = 1e-4,
    max_epochs: int = 300, patience: int = 40,
    device: str = None, outdir: str = "process",
    id2name: np.ndarray = None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = GraphAE(num_nodes, emb_dim, hidden_dim, latent_dim,
                    heads1, heads2, dropout, feat_recon_coef).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    node_ids_long = torch.arange(num_nodes, dtype=torch.long)  # 作为整数特征索引
    best_auc, best_state, patience_ctr = -1.0, None, 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        opt.zero_grad()

        z = model.encode(node_ids_long.to(device), train_data.edge_index.to(device))

        pos_edge, neg_edge = get_pos_neg_train_edges(train_data, num_nodes)
        pos_edge = pos_edge.to(device)
        neg_edge = neg_edge.to(device)

        pos_logits = model.decode(z, pos_edge)
        neg_logits = model.decode(z, neg_edge)

        # link loss
        pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
        neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
        link_loss = pos_loss + neg_loss

        if model.feat_decoder is not None and feat_recon_coef > 0:
            x_recon = model.feat_decoder(z)
            x_true = model.embed(node_ids_long.to(device)).detach()
            feat_loss = F.mse_loss(x_recon, x_true)
            loss = link_loss + feat_recon_coef * feat_loss
        else:
            feat_loss = torch.tensor(0.0, device=device)
            loss = link_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()

        val_auc, val_ap = evaluate(model, val_data, device, node_ids_long)

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"[Epoch {epoch:03d}] loss={loss.item():.4f} (link={link_loss.item():.4f}, feat={feat_loss.item():.4f}) "
                  f"| val AUC={val_auc:.4f}, AP={val_ap:.4f}")

        if patience_ctr >= patience:
            print(f"Early stop at epoch {epoch} (best val AUC={best_auc:.4f}).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)


    test_auc, test_ap = evaluate(model, test_data, device, node_ids_long)
    print(f"[TEST] AUC={test_auc:.4f}, AP={test_ap:.4f}")

    with torch.no_grad():
        model.eval()
        z_all = model.encode(node_ids_long.to(device), data.edge_index.to(device)).detach().cpu()

    os.makedirs(outdir, exist_ok=True)
    torch.save(z_all, os.path.join(outdir, "emb_go_gat_gae.pt"))

    if id2name is None:
        id2name = np.array([str(i) for i in range(num_nodes)], dtype=object)
    dfz = pd.DataFrame(z_all.numpy(), index=id2name)
    dfz.to_csv(os.path.join(outdir, "emb_go_gat_gae.csv"), header=False)
    print("Saved embeddings to:", os.path.join(outdir, "emb_go_gat_gae.csv"))

    return model, z_all, {"val_auc": best_auc, "test_auc": test_auc, "test_ap": test_ap}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_tsv", type=str, default="data/Node/GeneGO/goSim_BP.tsv")
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--min_sim", type=float, default=None)
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--heads1", type=int, default=4)
    parser.add_argument("--heads2", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--feat_recon_coef", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="process")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    edges_undirected, name2id, id2name = build_topk_graph_from_similarity(
        args.sim_tsv, topk=args.topk, min_sim=args.min_sim, outdir=args.outdir
    )

    src_idx = edges_undirected["source"].map(name2id).to_numpy()
    tgt_idx = edges_undirected["target"].map(name2id).to_numpy()
    edge_index = torch.tensor(np.vstack([src_idx, tgt_idx]), dtype=torch.long)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = to_undirected(edge_index, num_nodes=len(name2id))
    edge_index = coalesce(edge_index, num_nodes=len(name2id))
    print('edge_index', edge_index.shape)

    N = len(name2id)
    data = Data(x=torch.zeros((N, 1), dtype=torch.float32), edge_index=edge_index)

    splitter = RandomLinkSplit(
        num_val=0.05, num_test=0.10,
        is_undirected=True,
        add_negative_train_samples=False
    )
    train_data, val_data, test_data = splitter(data)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, z_all, metrics = train_loop(
        data, train_data, val_data,
        num_nodes=N,
        emb_dim=args.emb_dim, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim,
        heads1=args.heads1, heads2=args.heads2, dropout=args.dropout,
        feat_recon_coef=args.feat_recon_coef,
        lr=args.lr, weight_decay=args.weight_decay,
        max_epochs=args.max_epochs, patience=args.patience,
        device=device, outdir=args.outdir, id2name=id2name
    )
    print("Done. metrics:", metrics)


if __name__ == "__main__":
    main()
