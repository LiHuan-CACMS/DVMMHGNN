# DVMMHGNN

DVMMHGNN is a microbe-informed heterogeneous graph neural network framework for drug repurposing that integrates multi-modal biological data to predict drug–disease associations. The model constructs a unified drug–microbe–gene–disease heterogeneous graph and embeds diverse biological entities into a shared latent space through multi-modal feature fusion. To capture higher-order structural and semantic information, DVMMHGNN incorporates a graph masked autoencoder and dual-level graph augmentation, together with structural- and meta-path–aware contrastive learning. This design enables the model to preserve semantic consistency while enhancing robustness under sparse and noisy biological networks, ultimately supporting accurate and reproducible drug–disease association prediction.

---

## 🔍 Overview

Complex biological systems involve multiple interaction types (disease–gene, gene–gene, gene–pathway, etc.).  
DVMMHGNN builds a **heterogeneous graph** combining these relationships and applies **multi-modal message passing** to learn robust embeddings for disease and gene prediction tasks.

**Key features:**
- DVMMHGNN constructs a unified drug–microbe–gene–disease heterogeneous graph to explicitly model microbiota-mediated regulatory pathways for drug repurposing  
- The model integrates multi-modal biological features using pretrained encoders and learns robust representations via a graph masked autoencoder  
- Dual-level graph augmentation and dual-view contrastive learning are introduced to enhance structural robustness and semantic consistency under sparse conditions 

---

## 🧩 Project Structure

```
DVMMHGNN/
├── build_disease_similarity.py
├── build_go_graph_and_emb.py
├── build_pathway_graph_from_text.py
├── build_hetero_graph_sehgnn.py
├── build_sehgnn_data_hg.py
├── train_gat_disease_gae.py
├── train_gat_gae_go.py
├── train_gat_gae_pathway.py
├── main.py
├── pred.py
├── configs.py
├── utils.py
├── plot_superparam.py
├── stat_plot.ipynb
├── sehgnn/
│ └── (model implementation)
└── requirements.txt
```

### Main dependencies:

- PyTorch
- NumPy
- Pandas
- SciPy
- scikit-learn
- NetworkX


## 🚀 Usage

### 1. Build data 

Prepare disease–gene, gene–gene, GO, and pathway data, with `build_disease_similarity.py`, `build_go_graph_and_emb.py`, `build_pathway_graph_from_text.py`, `build_hetero_graph_sehgnn.py`. 

Their usage is similar.
For example, in "constructing the disease similarity matrix and computing edge links",
the corresponding code is: `build_disease_similarity.py`

```
python build_disease_similarity.py --mesh_xml data/desc2025.xml --disease_file data/Node/Disease.xlsx --out_matrix ./process/disease_edges_topk.csv --out_edges ./p
rocess/disease_edges_topk.csv
```


### 2. Train model

`configs.py` is the parameter configuration file for model training. The specific parameters are as follows:

```
PROC_DIR = "process"
RESULT_DIR = "results/OurModel"

# ----------------------------
# Training / structural hyperparameters
# ----------------------------
NUM_EPOCHS = 120           # Total number of epochs for each fold in cross-validation
COLDSTART_EPOCHS = 50      # Number of epochs for the cold-start sub-experiment
DROPOUT = 0.1              # Dropout probability before fusion projection
WEIGHT_DECAY = 0.0001      # L2 regularization (Adam weight decay)
HIDDEN_DIM = 64            # Hidden dimension for SeHGNN
OUT_DIM = 32               # Output dimension for SeHGNN
NUM_CLASSES = 16           # Number of classes for SeHGNN
NUM_GNN_LAYERS = 1         # Number of GNN layers
FUSION_OUT_DIM = 64        # Unified projection dimension after fusion

# ----------------------------
# Augmentation hyperparameters
# ----------------------------
AUG_STRUCT_ENABLE = True       # Enable structural augmentation
AUG_STRUCT_HOPS = 2            # Number of hops for structural augmentation
AUG_STRUCT_P_THR = 1           # Probability threshold for edge retention
AUG_STRUCT_SIM_THR = 0.6       # Structural similarity threshold
AUG_STRUCT_CLEAR_WEAK = False  # Whether to remove weak connections

AUG_META_ENABLE = True         # Enable meta-augmentation
AUG_META_K = 8                 # Top-k neighbors for meta-augmentation
AUG_META_LAMBDA_DRUG = 0.1     # Lambda coefficient for drug-related similarity
AUG_META_LAMBDA_DIS  = 0.1     # Lambda coefficient for disease-related similarity
AUG_META_SYN_RATIO = 0.1       # Ratio of synthetic samples generated
AUG_META_ADD_TO_GRAPH = True   # Whether to add synthetic nodes/edges into the graph

# ----------------------------
# Negative sampling configuration
# ----------------------------
NEG_SAMPLING_MODE = "uniform"   # Options: "uniform" or "adaptive"
NEG_BETA = 0.8                  # β: sampling ratio upper bound (relative to |P_train|)
NEG_STRUCT_HOPS = 2             # Order of structural dissimilarity counting
NEG_STRUCT_TOP_PCT = 0.5        # Top percentage for structural dissimilarity
NEG_SEM_TOP_PCT = 0.5           # Top percentage for semantic dissimilarity
NEG_CAND_MULT = 6               # Candidate pool size multiplier
NEG_PER_POS_CAP = 50            # Max negative samples considered per positive sample (safety limit)

SAVE_EMBED_PER_EPOCH = True     # Whether to save link-level concatenated embeddings each epoch

MODEL_TYPE = "SeHGNN"           # Options: "SeHGNN", "HGT", "GAT"

CONTRASTIVE_TAU = 0.1           # Temperature coefficient for contrastive learning

LR = 0.001                      # Learning rate
```

```
python main.py
```

### 3. Predict

```
python pred.py
```

--- 

Please contact the authors via email to obtain it.



