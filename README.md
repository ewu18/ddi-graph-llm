# DDI-Graph-LLM

**Does graph scope matter for drug–drug interaction type prediction?**
A hypothesis-driven study comparing Random Forest, GCN, and retrieval-augmented LLM approaches across four graph scopes.

NYU CSCI-UA 480 Social Networking · Spring 2026 · Mingjun (Eddie) Wu

---

## Overview

Drug–drug interaction (DDI) prediction systems are increasingly deployed in specialty clinical contexts — but it is unclear whether subgraphs restricted to a specialty (e.g. oncology) capture sufficient structure for accurate type prediction, or whether full-network features are required.

This project tests one focused hypothesis:

> **H₁.** Graph features computed over the full DDI network yield significantly better DDI type prediction than features computed over an oncology-only subgraph.

We construct four graph scopes (full, oncology, liquid cancer, solid cancer), compute 14 graph features (10 node-level + 4 pairwise) under a strict inductive evaluation protocol, and compare three model families: a feature-based **Random Forest**, a 2-layer **Graph Convolutional Network**, and three **LLM** conditions including a retrieval-augmented variant.

## Dataset

- **Source:** [Kaggle Drug-Drug Interactions](https://www.kaggle.com/datasets/mghobashy/drug-drug-interactions) (DrugBank-derived)
- **Scale:** 1,701 drugs · 191,541 typed directed interactions · 7 interaction types
- **Average degree:** ~225 (very dense)
- **Label extraction:** regex pipeline against 89 description templates, 99.997% coverage

The seven interaction types, by frequency: *adverse_effects* (32%), *activity* (24%), *metabolism* (21%), *concentration* (18%), *efficacy* (4%), *excretion* (1%), *absorption* (0.5%).

Place the downloaded CSV in `data/` as `drug_interactions.csv`.

## Graph scopes

| Scope         | Drugs  | Edges    | Subset rule                                |
|---------------|-------:|---------:|--------------------------------------------|
| Full          | 1,701  | 191,541  | All DrugBank-derived interactions          |
| Oncology      |    97  | ~22,000  | Antineoplastic / cancer-therapy drugs      |
| Liquid cancer |    35  |  ~8,300  | Hematological malignancy drugs             |
| Solid cancer  |    62  | ~14,400  | Solid-tumor drugs                          |

Subgraph rule: edge $(u,v)$ included if $u \in S$ **or** $v \in S$ (inclusive — reflects realistic specialty deployment).

## Methodology

**Features (14 total):**
- Node-level (5 per endpoint, 10 total): in-degree, out-degree, betweenness centrality, clustering coefficient, PageRank
- Pairwise (4 total): common neighbors, Jaccard coefficient, same-Louvain-community indicator, degree difference

**Models:**
- **Random Forest** — 500 trees, balanced class weights, hyperparameters frozen across scopes
- **2-layer GCN** — 5-dim node features → GCN(64) → GCN(64) → concat → MLP → 7-way softmax
- **LLM (GPT-4o-mini)** — three conditions: drug-names-only, raw graph features as text, RAG with K=5 retrieved analogues

**Inductive evaluation protocol** *(critical methodological control)*: test edges are removed from the graph **before** any feature computation. This prevents leakage from neighborhood-overlap features. Pilot transductive RF reached macro-F1 ≈ 0.97; the inductive setup reduces this to a defensible 0.926.

**Statistical validation:** McNemar's test on shared test pairs (drug pairs that appear in the test sets of both scopes being compared), with permutation tests as a parallel non-parametric check.

## Main results

**Random Forest macro-F1, by scope** (inductive, 95% bootstrap CIs):

| Scope     | macro-F1   | 95% CI            |
|-----------|-----------:|-------------------|
| Full      | **0.926**  | [0.921, 0.931]    |
| Oncology  | 0.848      | [0.799, 0.878]    |
| Liquid    | 0.727      | [0.703, 0.744]    |
| Solid     | 0.710      | [0.678, 0.845]    |

Monotone in scope size — consistent with H₁.

**Statistical tests on shared test pairs:**

| Comparison           | Shared n | Δ pairwise acc. | p       | Sig.            |
|----------------------|---------:|----------------:|---------|-----------------|
| Full vs Oncology     |      888 |          +0.025 | 0.036   | *               |
| Full vs Liquid       |      323 |          +0.081 | 0.0003  | ***             |
| Full vs Solid        |      623 |          +0.003 | 0.901   | n.s.            |
| Liquid vs Solid      |       23 |          −0.130 | 0.504   | n.s. (underpowered) |

**H₁ supported** for Full vs Oncology and Full vs Liquid. Full vs Solid: the macro-F1 gap is real but is driven by rare-class enrichment in Solid's full test set, not per-pair accuracy on shared pairs (both reach ~88%). Liquid vs Solid is underpowered (n=23 shared pairs).

## Mechanism finding: scope effect is model-specific

**GCN macro-F1 by scope (non-monotone):**

| Scope     | GCN macro-F1 |
|-----------|-------------:|
| Full      | 0.44         |
| Oncology  | 0.37         |
| Liquid    | 0.47         |
| Solid     | 0.36         |

Unlike RF, the GCN does **not** benefit from larger scope, and the non-monotonicity (Liquid > Full) rules out a simple scale interpretation.

The contrast between RF and GCN identifies the scope effect as **mechanism-specific** to models that consume explicit pairwise features. RF feature-importance ranking confirms: across every scope, **common neighbors and Jaccard similarity** are the two strongest predictors. Both encode neighborhood overlap, which is informative only when the surrounding network is dense enough — exactly the property a specialty subgraph dilutes.

A 2-layer GCN on a graph with average degree ~225 is also a plausible candidate for over-smoothing (Li et al. 2018), which would explain its scope-insensitivity through representation collapse rather than through scope itself.

## LLM follow-up: representation, not information

A separate LLM extension study tests whether graph information helps when delivered in different forms. All conditions use GPT-4o-mini on a sampled test subset (n=300):

| Condition                                  | macro-F1   |
|--------------------------------------------|-----------:|
| Random baseline (1/7)                       | 0.143     |
| LLM-only (drug names, zero-shot)            | 0.219     |
| LLM + raw graph features (text in prompt)   | 0.232     |
| LLM + retrieved cases (RAG, K=5)            | **0.839** |

The intermediate condition — passing the same 14 graph features that drive RF to 0.926, this time as text — improves macro-F1 only marginally over zero-shot. The retrieval condition, with no explicit feature vector but K=5 labeled analogues, reaches 0.839.

Same model, same task, comparable in-context information, dramatically different macro-F1 (0.232 → 0.839). We label this finding **"representation, not information"**: the LLM bottleneck is not lack of structural information but the form in which it's presented. Numerical feature vectors are not the input modality on which pretrained LLMs are well calibrated; labeled analogues are.

*LLM-side numbers are from a single inference run with gpt-4o-mini and may vary on re-run; RF/GCN numbers use a fixed random seed.*

## Repository structure

```
ddi-graph-llm/
├── README.md
├── requirements.txt
├── data/                              # Dataset (gitignored)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_graph_features.ipynb
│   ├── 03_baseline_rf.ipynb
│   ├── 04_llm_only.ipynb
│   ├── 05_gnn.ipynb
│   ├── 06_langgraph_agent.ipynb
│   ├── 07_rag_agent.ipynb
│   ├── 08_oncology_subgraph.ipynb     # 4-scope RF comparison
│   ├── 09_graph_visualizations.ipynb
│   └── 10_statistical_tests.ipynb     # bootstrap, McNemar, permutation
└── src/
    ├── nlp.py                         # Regex-based label extraction
    ├── graph.py                       # Graph construction & features
    ├── models.py                      # RF & GCN definitions
    └── agent.py                       # LangGraph + RAG pipelines
```

## Pipeline

1. **Data exploration & label extraction** *(01, 02)* — Regex template matching extracts 7 interaction types from description text
2. **Graph construction & features** *(02, 03)* — Build directed labeled DDI graph; compute 14 features
3. **Random Forest baseline** *(03)* — RF on graph features, full graph
4. **Scope comparison** *(08)* — RF across full / oncology / liquid / solid scopes under inductive protocol
5. **GCN** *(05)* — 2-layer graph convolutional network, end-to-end edge type classification
6. **LLM conditions** *(04, 06, 07)* — drug-names-only baseline, LangGraph with raw features, RAG with retrieved analogues
7. **Statistical validation** *(10)* — bootstrap CIs, McNemar's test on shared pairs, permutation tests
8. **Visualization** *(09)* — graph layouts and result figures

## References

- Wishart, D. S. et al. (2018). DrugBank 5.0: a major update to the DrugBank database for 2018. *Nucleic Acids Research* 46(D1).
- Kipf, T. N. & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. *ICLR*.
- Li, Q., Han, Z. & Wu, X.-M. (2018). Deeper insights into graph convolutional networks for semi-supervised learning. *AAAI*. *(over-smoothing)*
- Liben-Nowell, D. & Kleinberg, J. (2007). The link-prediction problem for social networks. *JASIST* 58(7).
- Barabási, A.-L., Gulbahce, N. & Loscalzo, J. (2011). Network medicine: a network-based approach to human disease. *Nature Reviews Genetics* 12(1).
- Blondel, V. D. et al. (2008). Fast unfolding of communities in large networks. *J. Stat. Mech.* P10008.
- McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika* 12(2).
- Efron, B. (1979). Bootstrap methods: another look at the jackknife. *Annals of Statistics* 7(1).
- Lewis, P. et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS*.
- LangChain (2024). LangGraph. https://github.com/langchain-ai/langgraph

## Author

Mingjun (Eddie) Wu — NYU CSCI-UA 480 Social Networking, Spring 2026
