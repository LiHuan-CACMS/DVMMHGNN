import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticFusionTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=1):
        super().__init__()
        self.query_proj = nn.Linear(input_dim, hidden_dim)
        self.key_proj = nn.Linear(input_dim, hidden_dim)
        self.value_proj = nn.Linear(input_dim, input_dim)
        self.beta = nn.Parameter(torch.ones(1))  # residual scaling

    def forward(self, feature_list):
        # feature_list: list of tensors [N x d] for each metapath
        H = torch.stack(feature_list, dim=1)  # [N, K, d]
        Q = self.query_proj(H)  # [N, K, h]
        K = self.key_proj(H)    # [N, K, h]
        V = self.value_proj(H)  # [N, K, d]

        attention_scores = torch.matmul(Q, K.transpose(-1, -2))  # [N, K, K]
        attention_scores = attention_scores / (Q.shape[-1] ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)  # [N, K, K]

        attended = torch.matmul(attention_weights, V)  # [N, K, d]
        H_out = self.beta * attended + H  # residual connection
        H_out = H_out.view(H_out.size(0), -1)  # flatten: [N, K*d]
        return H_out
