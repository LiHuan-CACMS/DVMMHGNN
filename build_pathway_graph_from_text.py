import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import numpy as np

pathway_text = np.load("data/Node/Pathway_text.npy", allow_pickle=True).item()
pathway_ids = list(pathway_text.keys())
texts = [pathway_text[pid] for pid in pathway_ids]

print("N pathway:", len(pathway_ids))
@torch.no_grad()
def encode_texts(texts, model_name="dmis-lab/biobert-base-cased-v1.1",
                 batch_size=16, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device).eval()
    def mean_pool(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        masked = last_hidden_state * mask
        summed = masked.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        return summed / counts
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        out = mdl(**enc)
        vec = mean_pool(out.last_hidden_state, enc["attention_mask"])
        embs.append(vec.detach().cpu())
    X = torch.cat(embs, dim=0)
    X = F.normalize(X, p=2, dim=1) 
    return X
emb = encode_texts(texts)
print("Emb shape:", emb.shape)
torch.save(emb, "process/emb_pathway_text.pt")
pd.DataFrame({"Pathway": pathway_ids}).to_excel("process/Pathway.norm.xlsx", index=False)




emb = torch.load("process/emb_pathway_text.pt")
names = pd.read_excel("process/Pathway.norm.xlsx")["Pathway"].tolist()
def build_topk_edges(emb, names, topk=10):
    N = emb.size(0)
    S = emb @ emb.t()  # 余弦相似
    S.fill_diagonal_(-1.0)

    scores, idx = torch.topk(S, k=topk, dim=1)
    edges = set()
    rows = []
    for i in range(N):
        for j, sc in zip(idx[i].tolist(), scores[i].tolist()):
            u, v = (names[i], names[j]) if names[i] <= names[j] else (names[j], names[i])
            if u != v and (u, v) not in edges:
                edges.add((u, v))
                rows.append((u, v, float(sc)))
    return pd.DataFrame(rows, columns=["source", "target", "score"])
edges = build_topk_edges(emb, names, topk=15)
edges.to_csv("process/pathway_edges_topk_undirected.csv", index=False)
print("Edges:", edges.shape)


