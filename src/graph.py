"""
Graph construction and feature engineering module.

Builds a directed DDI graph from labeled edge data and computes
node-level and edge-level topological features for downstream tasks.
"""

import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def build_graph(df: pd.DataFrame,
                src_col: str = "Drug 1",
                dst_col: str = "Drug 2",
                label_col: str = "label") -> nx.DiGraph:
    """
    Build a directed graph from the labeled DDI DataFrame.
    
    Each edge (u, v) carries the interaction-type label as an attribute.
    
    Args:
        df: DataFrame with drug pairs and labels.
        src_col: Column name for source drug.
        dst_col: Column name for target drug.
        label_col: Column name for interaction-type label.
    
    Returns:
        A NetworkX DiGraph.
    """
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(
            row[src_col],
            row[dst_col],
            label=row[label_col]
        )
    
    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def compute_node_features(G: nx.DiGraph) -> Dict[str, Dict[str, float]]:
    """
    Compute node-level topological features.
    
    Features per node:
        - in_degree, out_degree
        - betweenness_centrality
        - clustering_coefficient (on undirected projection)
        - pagerank
    
    Returns:
        Dict mapping node -> {feature_name: value}
    """
    G_undirected = G.to_undirected()
    
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    betweenness = nx.betweenness_centrality(G)
    clustering = nx.clustering(G_undirected)
    pagerank = nx.pagerank(G)
    
    node_features = {}
    for node in G.nodes():
        node_features[node] = {
            "in_degree": in_deg[node],
            "out_degree": out_deg[node],
            "betweenness": betweenness[node],
            "clustering": clustering[node],
            "pagerank": pagerank[node],
        }
    
    return node_features


def compute_edge_features(G: nx.DiGraph,
                          node_features: Dict[str, Dict[str, float]],
                          communities: Dict[str, int] = None) -> pd.DataFrame:
    """
    Compute edge-level features for each (u, v) pair.
    
    Features per edge:
        - out_degree_u, in_degree_v
        - betweenness_u, betweenness_v
        - common_neighbors (on undirected projection)
        - jaccard_coefficient (on undirected projection)
        - same_community (binary, if communities provided)
        - degree_diff (|deg_u - deg_v| on undirected)
    
    Returns:
        DataFrame with one row per edge, feature columns, and label.
    """
    G_undirected = G.to_undirected()
    
    records = []
    for u, v, data in G.edges(data=True):
        nf_u = node_features[u]
        nf_v = node_features[v]
        
        # Common neighbors (undirected)
        cn = len(list(nx.common_neighbors(G_undirected, u, v)))
        
        # Jaccard coefficient (undirected)
        jc_iter = nx.jaccard_coefficient(G_undirected, [(u, v)])
        _, _, jc = next(jc_iter)
        
        # Same community
        sc = 0
        if communities is not None:
            sc = int(communities.get(u, -1) == communities.get(v, -2))
        
        record = {
            "drug_u": u,
            "drug_v": v,
            "label": data.get("label"),
            # Node features of u
            "out_degree_u": nf_u["out_degree"],
            "in_degree_u": nf_u["in_degree"],
            "betweenness_u": nf_u["betweenness"],
            "clustering_u": nf_u["clustering"],
            "pagerank_u": nf_u["pagerank"],
            # Node features of v
            "out_degree_v": nf_v["out_degree"],
            "in_degree_v": nf_v["in_degree"],
            "betweenness_v": nf_v["betweenness"],
            "clustering_v": nf_v["clustering"],
            "pagerank_v": nf_v["pagerank"],
            # Pairwise features
            "common_neighbors": cn,
            "jaccard": jc,
            "same_community": sc,
            "degree_diff": abs(
                (nf_u["in_degree"] + nf_u["out_degree"]) -
                (nf_v["in_degree"] + nf_v["out_degree"])
            ),
        }
        records.append(record)
    
    return pd.DataFrame(records)


def detect_communities(G: nx.DiGraph) -> Dict[str, int]:
    """
    Detect communities using Louvain on the undirected projection.
    
    Returns:
        Dict mapping node -> community_id
    """
    from networkx.algorithms.community import louvain_communities
    
    G_undirected = G.to_undirected()
    communities_list = louvain_communities(G_undirected, seed=42)
    
    community_map = {}
    for idx, community in enumerate(communities_list):
        for node in community:
            community_map[node] = idx
    
    print(f"Detected {len(communities_list)} communities")
    return community_map


def graph_summary(G: nx.DiGraph) -> None:
    """Print basic graph statistics."""
    G_undirected = G.to_undirected()
    
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    print(f"Density: {nx.density(G):.4f}")
    print(f"Is weakly connected: {nx.is_weakly_connected(G)}")
    n_wcc = nx.number_weakly_connected_components(G)
    print(f"Weakly connected components: {n_wcc}")
    if nx.is_weakly_connected(G):
        print(f"Diameter (undirected): {nx.diameter(G_undirected)}")
    print(f"Average clustering (undirected): {nx.average_clustering(G_undirected):.4f}")
