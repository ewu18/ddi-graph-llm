# DDI-Graph-LLM

**Can Graph Structure Help LLMs Reason About Drug Interactions?**  
A LangGraph Agent Approach to DDI Type Prediction

## Overview

This project investigates whether augmenting a large language model (LLM) with structural features from a drug–drug interaction (DDI) network improves its ability to predict the mechanism type of a given interaction.

We compare four conditions:
- **A (LLM-only):** LLM prompted with drug names only
- **B1 (ML-only):** Random Forest trained on hand-crafted graph features
- **B2 (GNN):** Graph neural network learning edge representations end-to-end
- **C (LangGraph agent):** LLM augmented with graph-derived context via a multi-step agent

## Project Structure

```
ddi-graph-llm/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                  # Dataset (not tracked by git)
├── notebooks/             # Step-by-step Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_nlp_label_extraction.ipynb
│   ├── 03_graph_construction.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_baseline_rf.ipynb
│   ├── 06_gnn.ipynb
│   ├── 07_llm_only.ipynb
│   ├── 08_langgraph_agent.ipynb
│   └── 09_evaluation.ipynb
└── src/                   # Reusable modules
    ├── nlp.py             # Regex-based label extraction
    ├── graph.py           # Graph construction & feature computation
    ├── models.py          # RF & GNN model definitions
    └── agent.py           # LangGraph agent pipeline
```

## Dataset

- **Source:** [Kaggle Drug-Drug Interactions](https://www.kaggle.com/datasets/mghobashy/drug-drug-interactions)
- **Format:** CSV with columns `Drug 1`, `Drug 2`, `Interaction Description`
- **Derived from:** DrugBank

Place the downloaded CSV in `data/` as `drug_interactions.csv`.

## Pipeline

1. **NLP Label Extraction** — Regex + template matching to extract interaction types (metabolism, concentration, adverse effects, absorption, activity) from description text
2. **Graph Construction** — Build directed labeled DDI graph with NetworkX
3. **Feature Engineering** — Compute node-level (degree, centrality) and edge-level (common neighbors, Jaccard, community) features
4. **Four-Way Comparison:**
   - Condition A: LLM-only baseline
   - Condition B1: Random Forest on graph features
   - Condition B2: GNN (GCN/GAT) edge classification
   - Condition C: LangGraph agent with graph context
5. **Evaluation** — Macro-F1 comparison across conditions

## References

- Wishart et al. (2018). DrugBank 5.0. *Nucleic Acids Research*.
- Wang et al. (2021). DDI Predictions via Knowledge Graph and Text Embedding. *JMIR Medical Informatics*.
- Kipf & Welling (2017). Semi-Supervised Classification with Graph Convolutional Networks. *ICLR*.
- LangChain (2024). LangGraph. https://github.com/langchain-ai/langgraph

## Author

Eddie (Mingjun) Wu — NYU CSCI-UA 480 Social Networking, Spring 2026
