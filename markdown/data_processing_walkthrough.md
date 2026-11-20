# `01-data_processing.ipynb` — End-to-End Flow

The notebook turns the raw Zheng 68k PBMC dataset into the exact artifacts every
later model consumes: a node-feature matrix, a graph, a measured description of
that graph, a clean evaluation split, and reusable saved files. It is organized
as a pipeline where each part produces something the next part (and ultimately
Steps 2–4) depends on.

## Part 0 — Framing (Motivation & Description)
Two markdown cells state the research question: single-cell annotation as
**transductive semi-supervised node classification**, and whether a
**homophily-gated GNN** preserves class boundaries in a heterophilous graph. The
Motivation reflects the *measured* framing — heterophily is a label-granularity
effect in the T-cell family, not a rarity effect.
**Contribution:** sets the hypothesis the rest of the notebook tests rather than
assumes.

## Part 1 — Data Ingestion & Review
Downloads/loads `pbmc68k.h5ad` (65,877 × 33,939), then systematically
interrogates it: obs schema (11 celltypes), var schema (gene identifiers), the
`X`/`spliced`/`unspliced` matrices, and `matrix_snapshot` to confirm `X` is
count-like (non-negative, integer-like, sparse) *without assuming it*.
**Contribution:** establishes trust in the raw data and identifies the label
column and feature matrix — you can't preprocess soundly without knowing what the
numbers actually are.

## Part 2 — Exploratory Review (metadata & distributions)
Per-column obs distributions and value counts surface the **severe class
imbalance** (abundant T subtypes down to the 91-cell CD4+ T Helper2) and flag
mitochondrial genes.
**Contribution:** the imbalance discovered here drives two downstream decisions —
the per-class QC guardrail and the stratified split — and motivates the macro-F1
/ class-weighted-loss plan for Steps 2–3.

## Part 3 — Quality Control
Computes per-cell and per-gene QC metrics, visualizes them (depth,
genes-per-cell, MT%, count–depth), then applies recalibrated thresholds
(`MIN_GENES=20`, `MIN_COUNTS=1`, `MAX_PCT_MT=10`, `MIN_CELLS_PER_GENE=3`) with
**per-rule/per-class retention diagnostics**, a **threshold sweep**, and a
**guardrail** asserting every class keeps ≥90%. Result: `work` =
65,578 × 13,200 (99.5% retained, every class ≥96%).
**Contribution:** removes technical artifacts without biasing against any cell
type — directly protecting the rare classes the project is about. The guardrail
makes "we didn't throw away the minorities" a *checked* claim, not a hope.

## Part 4 — Normalization & Feature Selection
Preserves raw counts in a layer, applies `normalize_total(1e4)` + `log1p`
(removing sequencing-depth as a confounder and stabilizing variance), selects
2,000 HVGs, and documents that **scaling is intentionally omitted** (kept as an
optional ablation).
**Contribution:** puts expression on a comparable, denoised scale so that
cell-to-cell distances — which every graph and embedding depends on — reflect
biology, not capture depth.

## Part 5 — Representation: PCA
Runs PCA on the HVGs (50 components, top 50 → `P`) with a variance-ratio curve
and per-PC gene-loading bars.
**Contribution:** this is the core dimensionality reduction. It compresses
~13,200 noisy, correlated genes into 50 orthogonal axes that track real lineage
programs, escaping the curse of dimensionality. `P` becomes the **node-feature
matrix** for every model, and the loadings are the first interpretability hook
(precursor to Step 4's gradient-through-loadings attribution).

## Part 6 — Graph & Exploratory Clusters
Builds a k=15 kNN graph in PCA space and runs Leiden clustering (framed
explicitly as a sanity check, not labels).
**Contribution:** the kNN graph **is** the structure the GNNs message-pass over —
the single most important artifact for the graph half of the project. Leiden
checks whether data-driven communities align with annotated types, validating
that PCA + graph captured real biology.

## Part 7 — Soundness Analyses (the proactive core)
Three measured checks on that exact graph:
- **Homophily audit** — global edge homophily = 0.412; measures the project's
  central premise instead of asserting it.
- **Coarse-vs-fine homophily** — collapses T subtypes to test whether the
  heterophily is a label-granularity effect.
- **Transductive split** — stratified train/val/test node masks with
  disjoint/coverage assertions.

**Contribution:** turns the project's thesis into evidence (and reframed the
hypothesis from rarity to granularity), and establishes the single reusable
evaluation protocol every model in Steps 2–4 shares — the backbone of fair
comparison.

## Part 8 — Feature Engineering
Augments `P` with three **ablatable** blocks — curated marker genes (24), QC
covariates (3), label-free graph-structure features (6) — into
`work.obsm["X_features"]` with a block→column map, z-scored on train statistics
only. Local homophily is computed as an **analysis-only** column (no leakage).

The graph block is `degree, clustering, mean_nbr_dist, nbr_entropy, reciprocity,
nbr_dispersion` — each a distinct, label-free neighbourhood summary (connectivity,
triangle-closure, local density, categorical mixing, edge reciprocity, boundary
offset).

> **Why not classic centrality (PageRank, Katz, eigenvector, betweenness)?** The
> graph is a kNN graph built from PCA, so every node has (near) the same degree
> by construction — no hubs, no scale-free structure. On a degree-regular graph
> these centralities collapse into local-density proxies that add little beyond
> `degree` and nothing about class identity; betweenness is also ~O(V·E)
> (prohibitive at 65k nodes) and only re-finds boundary cells that `nbr_entropy`/
> `nbr_dispersion` already detect. We therefore use neighbourhood-heterogeneity
> features, which measure the boundary/mixing structure this project is about.

**Contribution:** injects known biology and topology that PCA averages away — the
lever for minority-class performance — while the toggles make each block a clean
ablation and the block map feeds Step 4 attribution.

## Part 9 — Visualization
Five embeddings (PCA, Laplacian, LLE, t-SNE, UMAP) on an independent stratified
`viz` subsample, colored by celltype and Leiden.
**Contribution:** the "EDA/UMAP" deliverable — lets you *see* whether types
separate and whether classes sit at boundaries (a visual complement to the
homophily numbers), with method breadth appropriate for the course. Decoupling to
`viz` keeps the slow methods fast without touching the analytical numbers.

## Part 10 — Persistence
Saves `work_processed.h5ad` (full object) plus GNN-ready side files
(`X_pca.npy`, `X_features.npy`, `feature_blocks.json`, `connectivities.npz`,
`cell_metadata.csv`) to the git-ignored `data/`.
**Contribution:** lets collaborators load the processed representation, graph,
features, and split in one line and jump straight to modeling — the clean
hand-off from Step 1 to Steps 2–4.

---

**The through-line:** every part feeds the next — review justifies QC, QC enables
sound normalization, normalization makes PCA meaningful, PCA defines both the
features and the graph, the graph is measured (homophily) and partitioned
(split), features are engineered on top, and everything is persisted. The
notebook's job is to produce **(1) node features `X_features`/`P`, (2) the kNN
graph, (3) a measured homophily profile, (4) a reusable split** — the four things
the entire GNN study is built on.



## What has been done for data processed 

  Cleaning & normalization
  - QC-filtered raw 65,877 × 33,939 → 65,578 cells × 13,200 genes
  (99.5% kept, every class ≥96%, per-class guardrail enforced)
  - Normalized to 10,000 counts/cell + log1p; raw counts preserved in
  a layer
  - Selected 2,000 highly variable genes for PCA
  
  Representation & graph
  - PCA: 50 components, top 50 → P (the node-feature matrix)
  - kNN graph (k=15) in PCA space + Leiden clusters (sanity check,
  not labels)
  
  Soundness artifacts
  - Homophily audit: global edge homophily = 0.412 (measured;
  reframed hypothesis to label-granularity)
  - Coarse-vs-fine homophily (celltype_coarse)
  - Stratified transductive split: train/val/test_mask (60/20/20)
  
  Engineered node features → work.obsm["X_features"] (65,578 × 83)
  - PCA (50) + marker genes (24) + QC covariates (3) +
  graph-structure (6)
  - Graph block: degree, clustering, mean_nbr_dist, nbr_entropy,
  reciprocity, nbr_dispersion (label-free; PageRank/wdegree dropped)
  - Block→column map in uns["feature_blocks"]; each block toggleable
  for ablation
  - local_homophily per node stored as analysis-only (not a feature,
  avoids leakage)
  
  Saved to data/processed/ (git-ignored)
  - work_processed.h5ad (~108 MB, complete object)
  - X_pca.npy (13 MB), X_features.npy (33 MB), feature_blocks.json
  - connectivities.npz (kNN graph), cell_metadata.csv (labels + masks
  + local_homophily)
  
  In short: raw counts → cleaned, normalized, 50-D PCA representation
  + kNN graph + measured homophily + clean split + a 83-D augmented
  feature matrix, all persisted for one-line reload by teammates.