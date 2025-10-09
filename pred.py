import numpy as np
import torch
import pandas as pd
from sehgnn.sehgnn_model import SeHGNN
import os, json, joblib


def load_features_and_mapping(fold_dir):
    feat_save_dir = os.path.join(fold_dir, "features")
    feat_drug_struct = joblib.load(os.path.join(feat_save_dir, "feat_drug_struct.pkl"))
    feat_dis_struct  = joblib.load(os.path.join(feat_save_dir, "feat_dis_struct.pkl"))
    feat_drug_meta   = joblib.load(os.path.join(feat_save_dir, "feat_drug_meta.pkl"))
    feat_dis_meta    = joblib.load(os.path.join(feat_save_dir, "feat_dis_meta.pkl"))
    with open(os.path.join(feat_save_dir, "disease_idx2id.json")) as f:
        disease_idx2id = json.load(f)
    with open(os.path.join(feat_save_dir, "n_disease.txt")) as f:
        n_disease = int(f.read().strip())
    return feat_drug_struct, feat_dis_struct, feat_drug_meta, feat_dis_meta, disease_idx2id, n_disease

class LinkPredictor(torch.nn.Module):
    def forward(self, h_drug, h_disease, edges):
        d_idx, dis_idx = edges[:, 0], edges[:, 1]
        return (h_drug[d_idx] * h_disease[dis_idx]).sum(dim=1)

def idx2id(mapping, i):
    if isinstance(mapping, dict):
        return mapping[str(i)]
    elif isinstance(mapping, list):
        return mapping[i]
    else:
        raise TypeError("Unsupported mapping type: {}".format(type(mapping)))


def predict_links_for_drug_by_name(model_drug_struct, model_drug_meta,
                                   model_dis_struct, model_dis_meta,
                                   proj_drug, proj_disease,
                                   features_drug_struct, features_drug_meta,
                                   features_dis_struct, features_dis_meta,
                                   predictor, n_disease, drug_name,
                                   topk=20, device="cpu"):
    drug_id_map = json.load(open("process/drug_id_map.json"))
    drug_feats  = np.load("process/drug_feats.npy")
    if drug_name not in drug_id_map:
        raise ValueError(f"Drug name {drug_name} not found in drug_id_map.json")
    drug_idx = drug_id_map.index(drug_name)
    drug_feat_vec = torch.tensor(drug_feats[drug_idx], dtype=torch.float32).unsqueeze(0).to(device)
    feats_struct_new = dict(features_drug_struct)
    feats_struct_new["Drug"] = torch.cat([features_drug_struct["Drug"], drug_feat_vec], dim=0)
    for k, v in features_drug_struct.items():
        if k != "Drug":
            feats_struct_new[k] = torch.cat([v, torch.zeros(1, v.shape[1]).to(device)], dim=0)
    feats_meta_new = dict(features_drug_meta)
    feats_meta_new["Drug"] = torch.cat([features_drug_meta["Drug"], drug_feat_vec], dim=0)
    for k, v in features_drug_meta.items():
        if k != "Drug":
            feats_meta_new[k] = torch.cat([v, torch.zeros(1, v.shape[1]).to(device)], dim=0)
    model_drug_struct.eval(); model_drug_meta.eval()
    model_dis_struct.eval();  model_dis_meta.eval()
    h_drug_struct = model_drug_struct(feats_struct_new)
    h_drug_meta   = model_drug_meta(feats_meta_new)
    h_dis_struct  = model_dis_struct(features_dis_struct)
    h_dis_meta    = model_dis_meta(features_dis_meta)
    h_drug_final = proj_drug(torch.cat([h_drug_struct, h_drug_meta], dim=1))
    h_dis_final  = proj_disease(torch.cat([h_dis_struct, h_dis_meta], dim=1))
    new_idx = h_drug_final.shape[0] - 1
    edges = np.array([[new_idx, dis] for dis in range(n_disease)])
    edges_t = torch.LongTensor(edges).to(device)
    scores = predictor(h_drug_final, h_dis_final, edges_t)
    probs = torch.sigmoid(scores).detach().cpu().numpy()
    df = pd.DataFrame({
        "drug_name": drug_name,
        "drug_idx": new_idx,
        "disease_idx": np.arange(n_disease),
        "prob": probs
    }).sort_values("prob", ascending=False)
    return df.head(topk)


if __name__ == "__main__":
    drug_name = "Acetaminophen"
    FUSION_OUT_DIM = 64
    fold_dir = "results/OurModel/fold1"
    feat_drug_struct, feat_dis_struct, feat_drug_meta, feat_dis_meta, disease_idx2id, n_disease = load_features_and_mapping(fold_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_drug_struct = SeHGNN(metapath_dims={k: v.shape[1] for k, v in feat_drug_struct.items()},
                               hidden_dim=64, out_dim=32, num_classes=16).to(device)
    model_drug_struct.load_state_dict(torch.load(os.path.join(fold_dir, "model_drug_struct.pt"), map_location=device))
    model_drug_meta = SeHGNN(metapath_dims={k: v.shape[1] for k, v in feat_drug_meta.items()},
                             hidden_dim=64, out_dim=32, num_classes=16).to(device)
    model_drug_meta.load_state_dict(torch.load(os.path.join(fold_dir, "model_drug_meta.pt"), map_location=device))
    model_dis_struct = SeHGNN(metapath_dims={k: v.shape[1] for k, v in feat_dis_struct.items()},
                              hidden_dim=64, out_dim=32, num_classes=16).to(device)
    model_dis_struct.load_state_dict(torch.load(os.path.join(fold_dir, "model_disease_struct.pt"), map_location=device))
    model_dis_meta = SeHGNN(metapath_dims={k: v.shape[1] for k, v in feat_dis_meta.items()},
                            hidden_dim=64, out_dim=32, num_classes=16).to(device)
    model_dis_meta.load_state_dict(torch.load(os.path.join(fold_dir, "model_disease_meta.pt"), map_location=device))

    with torch.no_grad():
        tmp_drug_struct = model_drug_struct(feat_drug_struct)
        tmp_drug_meta   = model_drug_meta(feat_drug_meta)
        tmp_dis_struct  = model_dis_struct(feat_dis_struct)
        tmp_dis_meta    = model_dis_meta(feat_dis_meta)
        in_dim_drug = tmp_drug_struct.size(1) + tmp_drug_meta.size(1)
        in_dim_dis  = tmp_dis_struct.size(1) + tmp_dis_meta.size(1)
    proj_drug = torch.nn.Linear(in_dim_drug, FUSION_OUT_DIM).to(device)
    proj_drug.load_state_dict(torch.load(os.path.join(fold_dir, "proj_drug.pt"), map_location=device))
    proj_disease = torch.nn.Linear(in_dim_dis, FUSION_OUT_DIM).to(device)
    proj_disease.load_state_dict(torch.load(os.path.join(fold_dir, "proj_disease.pt"), map_location=device))
    predictor = LinkPredictor().to(device)
    df_pred = predict_links_for_drug_by_name(
        model_drug_struct, model_drug_meta,
        model_dis_struct, model_dis_meta,
        proj_drug, proj_disease,
        feat_drug_struct, feat_drug_meta,
        feat_dis_struct, feat_dis_meta,
        predictor, n_disease, drug_name=drug_name, device=device, topk=20
    )
    df_pred["disease_id"] = df_pred["disease_idx"].apply(lambda i: idx2id(disease_idx2id, i))
    print("Top Diseases：")
    print(df_pred)
