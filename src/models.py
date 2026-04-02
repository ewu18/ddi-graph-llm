"""
Model definitions for DDI interaction-type prediction.

Condition B1: Random Forest on hand-crafted graph features
Condition B2: GNN (GCN/GAT) for end-to-end edge classification
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict


# ============================================================
# Condition B1: Random Forest
# ============================================================

FEATURE_COLS = [
    "out_degree_u", "in_degree_u", "betweenness_u", "clustering_u", "pagerank_u",
    "out_degree_v", "in_degree_v", "betweenness_v", "clustering_v", "pagerank_v",
    "common_neighbors", "jaccard", "same_community", "degree_diff",
]


def train_random_forest(
    df: pd.DataFrame,
    feature_cols: list = FEATURE_COLS,
    label_col: str = "label",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[RandomForestClassifier, LabelEncoder, Dict]:
    """
    Train a Random Forest classifier on graph features.
    
    Returns:
        (model, label_encoder, results_dict)
    """
    le = LabelEncoder()
    df = df.dropna(subset=[label_col])
    
    X = df[feature_cols].values
    y = le.fit_transform(df[label_col].values)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    
    print(f"=== Condition B1: Random Forest ===")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    results = {
        "macro_f1": macro_f1,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "label_encoder": le,
    }
    
    return clf, le, results


# ============================================================
# Condition B2: GNN (skeleton — implement with PyG)
# ============================================================

def build_gnn_data():
    """
    Convert the DDI graph into PyTorch Geometric Data object
    for edge classification.
    
    TODO: Implement when running GNN experiments.
    
    Steps:
        1. Map drug names to integer node IDs
        2. Create edge_index tensor from edge list
        3. Create node feature matrix (degree, centrality, etc.)
        4. Create edge label tensor from NLP-extracted labels
        5. Split edges into train/test masks
    """
    raise NotImplementedError("GNN data construction — implement in notebook 06")


def train_gnn():
    """
    Train a GCN or GAT for edge-level classification.
    
    TODO: Implement when running GNN experiments.
    
    Architecture sketch:
        - 2-3 GCN/GAT layers for node embedding
        - For each edge (u, v): concat [h_u || h_v] → MLP → K classes
        - Cross-entropy loss, Adam optimizer
        - Evaluate with macro-F1 on held-out edges
    """
    raise NotImplementedError("GNN training — implement in notebook 06")
