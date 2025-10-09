import numpy as np
import scipy.sparse as sp

def normalize_adj(adj):
    """Row-normalize sparse matrix"""
    rowsum = np.array(adj.sum(1)).flatten()
    rowsum[rowsum == 0] = 1  # avoid division by zero
    d_inv = 1.0 / rowsum
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj)

def aggregate_features_by_metapath(adj_matrices, features_dict, metapath):
    """
    adj_matrices: dict of adjacency matrices between node types
    features_dict: dict of raw feature matrices {type: np.ndarray}
    metapath: list of node types, e.g. ['A', 'P', 'A']
    """
    x = features_dict[metapath[-1]]
    for i in reversed(range(len(metapath) - 1)):
        src_type, dst_type = metapath[i], metapath[i+1]
        A = adj_matrices[f"{src_type}-{dst_type}"]
        A_norm = normalize_adj(A)
        x = A_norm.dot(x)
    return x
