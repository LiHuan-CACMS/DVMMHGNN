import os 

PROC_DIR = "process"
RESULT_DIR = "results/OurModel"
os.makedirs(RESULT_DIR, exist_ok=True)

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
