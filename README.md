# GitHub Social Network Analysis and Node Classification

This repository contains a comprehensive analysis and graph neural network-based classification of the GitHub Social Network dataset. The project includes network analysis techniques, community detection, link prediction, graphlet analysis, and node classification using GNNs, GATs, and Graph Transformers.

## 📁 Dataset Description

**Source**: Public GitHub API, June 2019  
**Files Included**:
- `musae_git_edges.csv` — mutual follower edges between developers  
- `musae_git_features.json` — node features (location, repositories starred, employer, email address)  
- `musae_git_target.csv` — binary labels (web vs. machine learning developer)

**Properties**:
- Nodes: 37,700  
- Edges: 289,003 (undirected)  
- Node features: ✔️  
- Node labels: ✔️ (binary classification task)  
- Edge features: ❌  
- Density: 0.001  
- Transitivity: 0.013  

---

## 🧪 Project Tasks and Notebooks

### 1. `01_load_visualize_centrality.ipynb`  
**Tasks**:
- Load and preprocess the dataset  
- Network visualization using NetworkX and related tools  
- Compute and compare different **centrality measures**:
  - Degree Centrality  
  - Betweenness Centrality  
  - Closeness Centrality  
  - Katz Centrality
  - Eigenvector Centrality  
- Visualize distributions and identify top-k influential nodes across centrality types

---

### 2. `02_community_detection.ipynb`  
**Tasks**:
- Apply community detection algorithms:
  - **Louvain Method**
  - **Leiden Algorithm**
- Analyze and compare resulting community structures:
  - Community size, density, internal structure  
  - Visualization by community color coding  
  - Evaluate partition quality via:
    - Modularity  
    - NMI (Normalized Mutual Information)  
    - ARI (Adjusted Rand Index)

---

### 3. `03_link_prediction_graphlets.ipynb`  
**Tasks**:

#### 🔗 Link Prediction:
- Predict likely future connections between developers using:
  - Common Neighbors  
  - Jaccard Coefficient  
  - Adamic-Adar Index  
  - Preferential Attachment  
- Evaluate prediction performance using:
  - AUC (Area Under ROC Curve)  
  - Average Precision (AP)

#### 🧩 Graphlet Analysis:
- Identify and analyze small subgraph patterns (graphlets):
  - 2-node, 3-node, and 4-node (using Fast Graphlet Transform)
- Count and visualize graphlet distributions to capture local structural patterns

---

### 4. `04_gnn_classification.ipynb`  
**Task**:
- Perform **node classification** using a **Graph Convolutional Network (GCN)**  
- Inputs: node features + graph structure  
- Predict whether a GitHub user is a web or ML developer  
- Train/validation/test split  
- Evaluate classification metrics (accuracy.)

---

### 5. `05_gat_transformer_classification.ipynb`  
**Task**:
- Extend node classification with:
  - **Graph Attention Network (GAT)**
  - **Basic Graph Transformer Models**
- Compare performance against GCN baseline
