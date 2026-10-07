
Project Motivation
Standard GNN classifiers achieve high global accuracy, but minority-class recall can collapse where isotropic message passing averages a node's features with neighbours that mostly belong to other classes (a heterophilous neighbourhood), smearing its signal into majority centroids. We began assuming this heterophily was driven by rarity (e.g., CD34+ cells at <1%), but measuring edge homophily on the actual kNN graph (global 0.412) refuted that: the most heterophilous classes are the abundant T-cell subtypes, not the rare ones. The heterophily is therefore largely a label-granularity effect — closely-related subtypes share one expression manifold, strongest in the T-cell family — with rarity compounding it only for CD4+ T Helper2. We evaluate whether an adaptive, homophily-gated architecture preserves decision boundaries in this measured, pervasive heterophilous regime.

Project Description
We formulate single-cell annotation as transductive semi-supervised node classification on Zheng 68k PBMC (N68,000, 11 discrete classes) across label-scarcity splits (1%, 2%, 5%, 10%, 20%, or 5–20 labeled cells/class). Severe class imbalance (abundant T cells to <1% CD34+ cells) combined with the measured subtype-driven heterophily makes this a natural test of class smoothing under graph message passing. 
Proposed Solution
Data retrieval: Download the full dataset and join it to the authors’ barcode annotations. Record data sources, matched barcodes, retained cells, and class counts.
Exploration & Pre-processing: Profile library depth and class imbalance. Apply QC, normalization, (1+x), 2,000 HVGs, and 50-D PCA (Scanpy used for the preprocessing engine and I/O); scaling kept as an optional ablation.
Data Analysis & Deep Learning: We benchmark a 4-tier PyTorch ladder: (1) Tabular MLP (features only), (2) Scratch Label Propagation (graph only), (3) GCN/GraphSAGE/GAT on k-NN/SNN graphs (homophily audit), and (4) our Homophily-Gated GCN (Initiative 2). A learnable gate balances self-features and neighborhood mean: 

Analysis and interpretability: Report accuracy, macro-F1, per-class F1, and variation across seeds. Relate class-specific gains and failures to homophily; test whether learned gates respond to neighborhood reliability. Where feasible, inspect marker-gene recovery. Ablate graph construction, neighborhood size, and model depth.

## STEPS 

Step 1 Data Engineering & Preprocessing
Download Zheng 68k, harmonise cell-type labels.
Apply QC, normalisation, log(1+p), HVG selection and PCA (Scanpy); treat scaling as an optional ablation.
Exploratory analysis: sparsity, class imbalance, UMAP.

Step 2 Graph Engineering & Baselines
Implement batched k-NN, SNN and graph construction.
Compute graph statistics and per-type homophily.
Implement LR, SVM, k-NN, MLP baselines and Label Propagation; run scANVI reference.

Step 3  GNN Architecture & Training
Build GCN, GraphSAGE, GAT and the homophily-gated layer from first principles (sparse scatter ops).
Semi-supervised masking pipeline, seeds, early stopping.
Run scarcity curve, few-shot, depth ablation experiments.

Step 4 Interpretability & Scientific Delivery
Extract model gradients back-projected through PCA loadings.
Quantitatively score recovery of canonical lineage markers (CD3E, MS4A1, NKG7, LYZ).
Correlate learned gate values gi against empirical homophily across all 11 classes.
Build interactive Plotly UMAPs, coordinate clean-kernel verification, compile slides/report.




