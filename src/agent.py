"""
LangGraph agent for DDI interaction-type prediction (Condition C).

Pipeline:
    1. Retrieve: Look up graph-derived features for a drug pair
    2. Format: Convert features into structured natural-language context
    3. Reason: Prompt LLM with drug names + graph context → predict type
"""

from typing import Dict, Optional


def format_graph_context(drug_u: str, drug_v: str, features: Dict) -> str:
    """
    Format graph-derived features into a natural-language context string
    that will be injected into the LLM prompt.
    
    Args:
        drug_u: Name of the source drug.
        drug_v: Name of the target drug.
        features: Dict of edge-level features from graph.py
    
    Returns:
        Formatted context string.
    """
    context = f"""Graph-derived structural features for the drug pair ({drug_u}, {drug_v}):

Drug {drug_u}:
  - Out-degree (number of drugs it affects): {features.get('out_degree_u', 'N/A')}
  - In-degree (number of drugs that affect it): {features.get('in_degree_u', 'N/A')}
  - Betweenness centrality: {features.get('betweenness_u', 'N/A'):.4f}
  - Clustering coefficient: {features.get('clustering_u', 'N/A'):.4f}
  - PageRank: {features.get('pagerank_u', 'N/A'):.6f}

Drug {drug_v}:
  - Out-degree: {features.get('out_degree_v', 'N/A')}
  - In-degree: {features.get('in_degree_v', 'N/A')}
  - Betweenness centrality: {features.get('betweenness_v', 'N/A'):.4f}
  - Clustering coefficient: {features.get('clustering_v', 'N/A'):.4f}
  - PageRank: {features.get('pagerank_v', 'N/A'):.6f}

Pairwise features:
  - Common neighbors: {features.get('common_neighbors', 'N/A')}
  - Jaccard similarity: {features.get('jaccard', 'N/A'):.4f}
  - Same community: {features.get('same_community', 'N/A')}
  - Degree difference: {features.get('degree_diff', 'N/A')}"""
    
    return context


def build_classification_prompt(drug_u: str, drug_v: str, 
                                 graph_context: Optional[str] = None) -> str:
    """
    Build the LLM prompt for DDI type classification.
    
    For Condition A: graph_context=None (drug names only)
    For Condition C: graph_context=formatted string from format_graph_context()
    
    Args:
        drug_u: Source drug name.
        drug_v: Target drug name.
        graph_context: Optional graph feature context string.
    
    Returns:
        Complete prompt string.
    """
    base_prompt = f"""You are a pharmacology expert. Given a pair of drugs that are known to interact, 
predict the type of drug-drug interaction.

The possible interaction types are:
1. metabolism — one drug affects the metabolism (breakdown) of the other
2. concentration — one drug changes the serum concentration of the other
3. adverse_effects — the combination increases the risk or severity of side effects
4. absorption — one drug affects the absorption of the other
5. activity — one drug increases or decreases a specific pharmacological activity of the other

Drug pair: {drug_u} → {drug_v}
(This means {drug_u} has a documented effect on {drug_v}.)
"""
    
    if graph_context:
        base_prompt += f"""
The following structural information from the drug interaction network is available:

{graph_context}

Use both your pharmacological knowledge and the network structural information to make your prediction.
"""
    
    base_prompt += """
Respond with ONLY one of these labels: metabolism, concentration, adverse_effects, absorption, activity
Your prediction:"""
    
    return base_prompt


def build_langgraph_agent():
    """
    Build the LangGraph multi-step agent workflow.
    
    TODO: Implement when running agent experiments.
    
    Workflow graph:
        START → retrieve_features → format_context → llm_classify → END
    
    Each node is a function that reads/writes to a shared state dict.
    """
    raise NotImplementedError("LangGraph agent — implement in notebook 08")
