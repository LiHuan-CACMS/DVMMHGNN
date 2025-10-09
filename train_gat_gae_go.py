import os
import random
import json
import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.data import Data
from torch_geometric.utils import (
    to_undirected, remove_self_loops, coalesce, negative_sampling
)
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import GATConv

OUTDIR = "process"
os.makedirs(OUTDIR, exist_ok=True)
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def load_graph(suffix: str):
    GO_XLSX = os.path.join(OUTDIR, f"GO_{suffix}.norm.xlsx")
    EMB_PT  = os.path.join(OUTDIR, f"emb_go_text_{suffix}.pt")
    EDGE_CSV= os.path.join(OUTDIR, f"go_edges_topk_undirected_{suffix}.csv")

    df = pd.read_excel(GO_XLSX)
    ids = df["GOID"].astype(str).tolist()
    id2idx = {g: i for i, g in enumerate(ids)}
    emb = torch.load(EMB_PT).float()
    emb = F.normalize(emb, p=2, dim=1)

    edges = pd.read_csv(EDGE_CSV)
    edges = edges[edges["source"].astype(str).isin(id2idx) & edges["target"].astype(str).isin(id2idx)]
    src = edges["source"].astype(str).map(id2idx).to_numpy()
    tgt = edges["target"].astype(str).map(id2idx).to_numpy()

    edge_index = torch.from_numpy(np.vstack([src, tgt])).long()
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = to_undirected(edge_index, num_nodes=emb.shape[0])
    edge_index = coalesce(edge_index, num_nodes=emb.shape[0])
    data = Data(x=emb, edge_index=edge_index)  # x:[N,D]
    return data, ids

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
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=-1)

class GraphAE(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, latent_dim=64,
                 heads1=4, heads2=4, dropout=0.2, feat_recon_coef=0.0):
        super().__init__()
        self.encoder = GATEncoder(in_dim, hidden_dim, latent_dim, heads1, heads2, dropout)
        self.decoder = DotDecoder()
        self.feat_recon_coef = feat_recon_coef
        if feat_recon_coef > 0:
            self.feat_decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, in_dim),
            )
        else:
            self.feat_decoder = None

    def encode(self, x, edge_index):
        return self.encoder(x, edge_index)

    def decode(self, z, edge_index):
        return self.decoder(z, edge_index)

def get_pos_neg_train_edges(train_data, num_nodes):
    pos_edge = train_data.edge_index
    num_pos = pos_edge.size(1)
    neg_edge = negative_sampling(
        pos_edge, num_nodes=num_nodes, num_neg_samples=num_pos, method='sparse'
    )
    return pos_edge, neg_edge

def _binary_from_scores(y_true: np.ndarray, y_score: np.ndarray, thr: float):
    y_pred = (y_score >= thr).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return tp, tn, fp, fn, y_pred

def _safe_div(a, b):
    return a / b if b > 0 else 0.0

def compute_threshold_metrics(y_true: np.ndarray, y_score: np.ndarray, thr: float):
    tp, tn, fp, fn, y_pred = _binary_from_scores(y_true, y_score, thr)
    precision   = _safe_div(tp, tp + fp)
    recall      = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    accuracy    = _safe_div(tp + tn, tp + tn + fp + fn)
    f1          = _safe_div(2 * precision * recall, precision + recall)
    return {
        "threshold": float(thr),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "accuracy": float(accuracy),
        "f1": float(f1),
        "y_pred": y_pred,
    }


def find_best_threshold_by_f1(y_true: np.ndarray, y_score: np.ndarray, num_grid: int = 200):
    a, b = float(y_score.min()), float(y_score.max())
    if not np.isfinite(a) or not np.isfinite(b) or abs(b - a) < 1e-8:
        return 0.5, compute_threshold_metrics(y_true, y_score, 0.5)
    best_thr, best_f1, best_metrics = 0.5, -1.0, None
    for thr in np.linspace(a, b, num_grid):
        m = compute_threshold_metrics(y_true, y_score, thr)
        if m["f1"] > best_f1:
            best_f1, best_thr, best_metrics = m["f1"], thr, m
    return float(best_thr), best_metrics


@torch.no_grad()
def evaluate(model, data_all, data_split, device, select_threshold_on_split=False):
    model.eval()
    x = data_all.x.to(device)
    z = model.encode(x, data_split.edge_index.to(device))

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
        edge_label_index = data_split.edge_label_index.to(device)
        logits = model.decode(z, edge_label_index).detach().cpu()
        y_true = data_split.edge_label.detach().cpu().numpy()
        y_score = torch.sigmoid(logits).numpy()

    best_thr = None
    if select_threshold_on_split:
        best_thr, _ = find_best_threshold_by_f1(y_true, y_score)

    auc = roc_auc_score(y_true, y_score)
    aupr = average_precision_score(y_true, y_score)

    m05 = compute_threshold_metrics(y_true, y_score, 0.5)
    metrics = {
        "AUC": float(auc),
        "AUPR": float(aupr),
        "threshold@0.5": m05["threshold"],
        "F1@0.5": m05["f1"],
        "accuracy@0.5": m05["accuracy"],
        "recall@0.5": m05["recall"],
        "specificity@0.5": m05["specificity"],
        "precision@0.5": m05["precision"],
    }

    if best_thr is not None:
        mb = compute_threshold_metrics(y_true, y_score, best_thr)
        metrics.update({
            "best_threshold": mb["threshold"],
            "F1@best": mb["f1"],
            "accuracy@best": mb["accuracy"],
            "recall@best": mb["recall"],
            "specificity@best": mb["specificity"],
            "precision@best": mb["precision"],
        })

    return metrics, (y_true, y_score)


def train_loop(
    data, train_data, val_data, test_data,
    in_dim, hidden_dim=256, latent_dim=64,
    heads1=4, heads2=4, dropout=0.2,
    feat_recon_coef=0.0,
    lr=1e-3, weight_decay=1e-4,
    max_epochs=300, patience=40,
    device=None, outdir="process", names=None, suffix: str = "BP"
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = GraphAE(in_dim, hidden_dim, latent_dim,
                    heads1, heads2, dropout, feat_recon_coef).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_auc, best_state, patience_ctr = -1.0, None, 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        opt.zero_grad()

        x = data.x.to(device)
        z = model.encode(x, train_data.edge_index.to(device))

        pos_edge, neg_edge = get_pos_neg_train_edges(train_data, data.num_nodes)
        pos_edge = pos_edge.to(device)
        neg_edge = neg_edge.to(device)

        pos_logits = model.decode(z, pos_edge)
        neg_logits = model.decode(z, neg_edge)

        pos_loss = torch.nn.functional.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
        neg_loss = torch.nn.functional.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
        link_loss = pos_loss + neg_loss

        if model.feat_decoder is not None and feat_recon_coef > 0:
            x_recon = model.feat_decoder(z)
            feat_loss = torch.nn.functional.mse_loss(x_recon, x)
            loss = link_loss + feat_recon_coef * feat_loss
        else:
            feat_loss = torch.tensor(0.0, device=device)
            loss = link_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()

        val_metrics, _ = evaluate(model, data, val_data, device, select_threshold_on_split=False)
        val_auc = val_metrics["AUC"]

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"[Epoch {epoch:03d}] loss={loss.item():.4f} "
                  f"(link={link_loss.item():.4f}, feat={feat_loss.item():.4f}) | "
                  f"val AUC={val_metrics['AUC']:.4f}, AUPR={val_metrics['AUPR']:.4f}, "
                  f"F1@0.5={val_metrics['F1@0.5']:.4f}")

        if patience_ctr >= patience:
            print(f"Early stop at epoch {epoch} (best val AUC={best_auc:.4f}).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_sel, _ = evaluate(model, data, val_data, device, select_threshold_on_split=True)
    best_thr = val_sel.get("best_threshold", 0.5)
    print(f"[VAL] AUC={val_sel['AUC']:.4f}, AUPR={val_sel['AUPR']:.4f} | "
          f"best_thr={best_thr:.4f}, F1@best={val_sel.get('F1@best', float('nan')):.4f}")

    test_metrics_raw, (y_true_test, y_score_test) = evaluate(
        model, data, test_data, device, select_threshold_on_split=False
    )
    m_best_test = compute_threshold_metrics(y_true_test, y_score_test, best_thr)

    print(f"[TEST] AUC={test_metrics_raw['AUC']:.4f}, AUPR={test_metrics_raw['AUPR']:.4f} | "
          f"thr={best_thr:.4f} -> "
          f"F1={m_best_test['f1']:.4f}, Acc={m_best_test['accuracy']:.4f}, "
          f"Recall={m_best_test['recall']:.4f}, Specificity={m_best_test['specificity']:.4f}, "
          f"Precision={m_best_test['precision']:.4f}")

    with torch.no_grad():
        model.eval()
        z_all = model.encode(data.x.to(device), data.edge_index.to(device)).detach().cpu()
    torch.save(z_all, os.path.join(outdir, f"emb_go_gat_gae_{suffix}.pt"))
    out_csv = os.path.join(outdir, f"emb_go_gat_gae_{suffix}.csv")
    pd.DataFrame(z_all.numpy(), index=names).to_csv(out_csv, header=False)
    print("Saved:", out_csv)

    metrics = {
        "val": val_sel,
        "test": {
            "AUC": float(test_metrics_raw["AUC"]),
            "AUPR": float(test_metrics_raw["AUPR"]),
            "threshold": float(m_best_test["threshold"]),
            "F1": float(m_best_test["f1"]),
            "accuracy": float(m_best_test["accuracy"]),
            "recall": float(m_best_test["recall"]),
            "specificity": float(m_best_test["specificity"]),
            "precision": float(m_best_test["precision"]),
        }
    }
    return model, z_all, metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", type=str, required=True, help="BP / CC / MF")
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--latent_dim", type=int, default=64)
    ap.add_argument("--heads1", type=int, default=4)
    ap.add_argument("--heads2", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--max_epochs", type=int, default=1000)
    ap.add_argument("--patience", type=int, default=1000)
    args = ap.parse_args()

    suffix = args.suffix.upper()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    data, names = load_graph(suffix)
    print(data)

    splitter = RandomLinkSplit(
        num_val=0.05, num_test=0.10,
        is_undirected=True,
        add_negative_train_samples=False
    )
    train_data, val_data, test_data = splitter(data)

    model, z_all, metrics = train_loop(
        data=data,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        in_dim=data.num_features,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        heads1=args.heads1, heads2=args.heads2,
        dropout=args.dropout,
        feat_recon_coef=0.0,
        lr=args.lr, weight_decay=args.weight_decay,
        max_epochs=args.max_epochs, patience=args.patience,
        device=device, outdir=OUTDIR, names=names, suffix=suffix
    )

    print("Final metrics:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
