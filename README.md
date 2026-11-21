# Graph-ML

CS5284 Group 16. Label-scarce cell-type classification on the Zheng 68k PBMC
dataset: k-NN and SNN cell graphs, MLP and label-propagation baselines, GCN,
GraphSAGE, GAT and a homophily-gated GCN, all in plain PyTorch.

## Repository layout

| Path | Step | What it does | Writes to |
|---|---|---|---|
| `01-data_processing.ipynb` | 1 | Download, QC, normalisation, 2,000 HVGs, 50-PC PCA, exploratory clustering and UMAP, 60/20/20 split masks | `data/processed/` |
| `02-graph_engineering_baselines.ipynb` | 2 | Exact k-NN and SNN graphs, graph statistics, homophily, MLP (3 seeds) and label-propagation baselines, GNN input export | `data/graph_engineering_50pc/`, `data/mlp_50pc/` |
| `03-gnn_training.ipynb` | 3 | GCN, GraphSAGE, GAT and gated GCN on both graphs at 5 % and full labels, plus scarcity and depth ablations | `data/gnn_training_50pc/` |
| `04-interpretability.ipynb` | 4 | Per-class errors, gate analysis, marker attribution through PCA loadings, UMAP panels | `data/interpretability/` |
| `src/models.py` | 3 | The four GNN layers and the classifier wrapper | |
| `src/pipeline_train.py` | 3 | Data loading, seeding, training loop, experiment grid | |
| `src/interpretability.py` | 4 | Evaluation and attribution helpers for notebook 04 | |
| `markdown/` | | Walkthroughs for notebooks 01 to 03 | |
| `data/`, `temp/` | | Git-ignored: raw data, generated outputs, scratch files | |

## Install

Python 3.11 is required.

```bash
python3.11 -m venv .venv              # or: conda create -n graph-ml python=3.11 && conda activate graph-ml
source .venv/bin/activate             # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name graph-ml
```

For a GPU, install the CUDA build of torch from
<https://pytorch.org/get-started/locally/>

## Run

Start Jupyter from the repository root, choose the `graph-ml` kernel and run each
notebook top to bottom, in order. Each notebook reads only what the previous ones
wrote.

1. `01-data_processing.ipynb` downloads `data/PBMC/pbmc68k.h5ad` on the first run.
2. `02-graph_engineering_baselines.ipynb`. Keep `EVALUATE_TEST = False` until the
   experiment is final; the exported graphs are reused on later runs, the
   baselines train again.
3. `03-gnn_training.ipynb`. `RUN_MAIN` runs the 48 main trainings and
   `RUN_ABLATIONS` the 60 ablation trainings. Under an hour on a GPU, several
   hours on a CPU.
4. `04-interpretability.ipynb` needs the outputs of 02 and 03.

Details per notebook are in `markdown/`.

## Loading the processed data directly

```python
import anndata as ad, numpy as np, scipy.sparse as sp, pandas as pd, json
work = ad.read_h5ad("data/processed/work_processed.h5ad")        # full object
X    = np.load("data/processed/X_features.npy")                  # augmented node features
cols = json.load(open("data/processed/feature_blocks.json"))     # {"pca": [0, 50], "markers": [...], ...}
A    = sp.load_npz("data/processed/connectivities.npz")          # scanpy kNN graph (reference only)
meta = pd.read_csv("data/processed/cell_metadata.csv", index_col=0)  # labels, masks, local_homophily
```

The graphs the GNNs train on are the exact k-NN and SNN graphs under
`data/graph_engineering_50pc/graphs/`, not the scanpy graph above.

## Git

- Pull before you start, push small commits, tell the group when you push.
- Do not commit anything under `data/` or `temp/`. Do not edit `.gitignore` unless
  you are adding more files to ignore.

```bash
git pull
git add <files>
git commit -m "[NAME] what you did"
git push
```
