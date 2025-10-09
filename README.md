# DVMMHGNN

**DVMMHGNN (Disease–Variant–Multi-Modal Heterogeneous Graph Neural Network)** is a framework for integrating multi-source biological data (disease, gene, pathway, GO terms, etc.) into a unified heterogeneous graph and learning representations through graph neural networks.

---

## 🔍 Overview

Complex biological systems involve multiple interaction types (disease–gene, gene–gene, gene–pathway, etc.).  
DVMMHGNN builds a **heterogeneous graph** combining these relationships and applies **multi-modal message passing** to learn robust embeddings for disease and gene prediction tasks.

**Key features:**
- Integrates multiple biological networks and annotations  
- Supports heterogeneous node and edge types  
- Uses attention-based or adaptive fusion across modalities  
- Applicable to disease-gene prediction, biological classification, and knowledge discovery  

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
