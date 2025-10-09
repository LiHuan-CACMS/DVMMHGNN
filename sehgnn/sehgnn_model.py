import torch
import torch.nn as nn
from .feature_projection import FeatureProjector
from .semantic_fusion_transformer import SemanticFusionTransformer

class SeHGNN(nn.Module):
    def __init__(self, metapath_dims, hidden_dim=128, out_dim=128, num_classes=3, dropout=0.5):
        """
        metapath_dims: dict {metapath_name: feature_dim}
        """
        super().__init__()
        self.metapaths = list(metapath_dims.keys())
        self.projectors = nn.ModuleDict({
            mp: FeatureProjector(in_dim, hidden_dim, out_dim, dropout)
            for mp, in_dim in metapath_dims.items()
        })
        self.semantic_fusion = SemanticFusionTransformer(out_dim, hidden_dim // 4)
        self.classifier = nn.Sequential(
            nn.Linear(out_dim * len(metapath_dims), out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, num_classes)
        )

    def forward(self, features_by_metapath):
        """
        features_by_metapath: dict of {metapath_name: [N x dim]}
        """
        projected = [self.projectors[mp](features_by_metapath[mp]) for mp in self.metapaths]
        fused = self.semantic_fusion(projected)
        return self.classifier(fused)
