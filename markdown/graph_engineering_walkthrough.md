# Graph engineering and baselines walkthrough

[02-graph_engineering_baselines.ipynb](../02-graph_engineering_baselines.ipynb) builds k-NN and SNN graphs from the PCA features saved by `01-data_processing.ipynb`. It checks the graphs, runs MLP and Label Propagation baselines, and saves the inputs for GNN training.

## Experiment scope

Use **MLP as the core baseline** and Label Propagation as an additional graph-based comparison. Both run at two label budgets.

The proposed homophily-gated GCN is developed separately. Compare it with MLP to assess the value of the graph-based model. Compare it with an ungated GCN to assess the gate itself.

Other classifiers, extra GNN architectures, rejection methods and marker recovery are outside this notebook's scope. The k-NN graph remains part of the experiment and is distinct from a k-NN classifier.

## How to run

1. Use a Python 3.11 environment with the packages in `requirements.txt`. Install them with `python -m pip install -r requirements.txt` if needed.
2. Run [01-data_processing.ipynb](../01-data_processing.ipynb) first. Skip this step if its processed files are already available.
3. Start Jupyter from the repository root. Open `02-graph_engineering_baselines.ipynb` and select the **graph-ml** kernel, or your environment's kernel.
4. Keep `EVALUATE_TEST = False` during development. Run the cells from top to bottom.

The notebook expects these files in **`data/processed/`**:

```text
X_pca.npy
cell_metadata.csv
connectivities.npz
work_processed.h5ad
```

Outputs go to **`data/graph_engineering/`**. Matching graph files are reused, but the baseline models train again. Changes to inputs, settings or code rebuild the generated exports in this folder. Copy any results you want to keep before changing an experiment.

## Notebook sections

| Section | What it does |
|---|---|
| 1. Settings | Uses seed 42 and the existing 60/20/20 train, validation and test masks. Defines two training-label budgets. |
| 2. Load inputs | Checks cell order and split masks. Uses the first 30 PCs as features. |
| 3. k-NN | Finds 15 other neighbours per cell using Euclidean distance. Every query batch searches all cells. Keeps an undirected edge if either cell selects the other. |
| 4. SNN | Weights each k-NN edge by the Jaccard overlap of its neighbour lists. Removes edges with zero overlap. |
| 5. Graph statistics | Checks graph structure and reports degrees, components, isolated cells, clustering and weights. Includes the Scanpy graph as a reference. |
| 6. Homophily | Measures same-type connections using only the selected training labels. Reports global and per-type values with supporting counts. |
| 7. Export | Saves graphs, features, masks and targets for GNN training. |
| 8. MLP | Compares hidden-layer widths 64 and 128. Selects the width and checkpoint using validation macro-F1. |
| 9. Label Propagation | Propagates the available training labels over each graph. Reports convergence and any cells assigned a fallback class. |
| 10. Validation | Compares MLP, LP on k-NN and LP on SNN at both budgets. Saves overall and per-type scores. |
| 11. Test | Scores the fixed models only when final test evaluation is enabled. |
| 12. GNN loading example | Loads the exported inputs and prepares edge indices and weights. |

**Why can degree exceed 15?** Each cell selects 15 neighbours, but other cells can also select it. The undirected graph keeps both sets of connections.

## Two label budgets

| Folder | Training labels used | Current count |
|---|---|---|
| `labels_5pct/` | 5% of each class in the training pool, rounded up | 1,971 cells, about 3% of all cells |
| `labels_full/` | All cells in the training pool | 39,346 cells, about 60% of all cells |

Both experiments use the same graph and validation/test masks. Unselected training cells stay in the graph, but their labels are unavailable to the model. Each budget has a scaler fitted only on its selected training cells.

## Where to find the outputs

Paths below are relative to **`data/graph_engineering/`**. Section 12 of the notebook includes a loading example.

| File or folder | Use |
|---|---|
| `manifest.json` | Settings, file hashes and completion status. `graph_exports_complete` confirms the GNN inputs were saved. `baselines_complete` confirms all six baseline runs finished. |
| `cell_ids.csv`, `class_names.json` | Shared cell order and the mapping from integer labels to cell types. |
| `X_pca.npy` | Original 30-PC features used to build the graphs. |
| `graphs/` | Sparse k-NN and SNN adjacency matrices, neighbour indices and distances. |
| `labels_5pct/`, `labels_full/` | Each budget's `X_scaled.npy`, `masks.npz`, `training_labels.npy`, `validation_labels.npy` and fitted scaler. |
| `analysis/` | Graph statistics and homophily tables. Use for analysis only. |
| `results/` | Validation scores, per-type metrics, predictions, confusion matrices and MLP training histories. |
| `models/` | MLP checkpoints and Label Propagation run details. |

## Using the exports for GNN training

1. Choose `labels_5pct/` or `labels_full/`. Load its scaled features, masks and targets.
2. Load `graphs/knn_adjacency.npz` or `graphs/snn_adjacency.npz` with `scipy.sparse.load_npz`. Keep the graph sparse and preserve SNN edge weights.
3. Keep every array in `cell_ids.csv` order. Run message passing on all cells. Compute supervised loss only where `masks["labelled_train"]` is true.
4. Use `masks["validation"]` and the separate validation targets to choose checkpoints. Keep test scoring off until the experiment is fixed.
5. Check whether the GNN layer already adds self-loops and normalises edges. The saved graphs do neither.

Training targets are **-1 outside `labelled_train`**. For the small budget, `train_pool` includes cells whose labels are unavailable, so it must not be used as the supervised loss mask. Test targets are not included in these exports. Saved predictions for test cells are predictions only.

## How the transductive split prevents label leakage

**All cells' features and graph connections are available, but their labels have separate roles.** Edges can connect training, validation and test cells. Message passing across those edges is allowed in this transductive experiment.

| Labels | Allowed use |
|---|---|
| Selected training labels | Fit the MLP, seed Label Propagation and compute the GNN training loss. |
| Unselected training labels | Used to sample the per-class label budget, then hidden from fitting. |
| Validation labels | Evaluate predictions and select model settings or checkpoints. Never use them as training targets or propagation seeds. |
| Test labels | Score the fixed models at the end. Never use them to choose features, graphs, settings or checkpoints. |

The notebook checks masks and separates targets as follows:

- **Check the masks.** Train, validation and test masks are disjoint and cover all cells. `labelled_train` must be a subset of the training pool. Both budgets reuse the same validation/test masks.
- **Hide unavailable targets.** `training_labels.npy` contains labels only at `labelled_train`, with **-1 everywhere else**. MLP fits only those rows. Label Propagation receives this masked label array.
- **Keep graph construction label-free.** k-NN uses PCA distances. SNN uses neighbour-list overlap. Neither uses cell types, label agreement or homophily to add, remove or weight edges.
- **Separate analysis from features.** Homophily uses only selected training labels and is saved for analysis. Do not feed it into a GNN or its gate. Fit each feature scaler only on that budget's selected training cells.
- **Keep test scoring off during development.** Test targets are excluded from the GNN export. `EVALUATE_TEST=False` prevents this notebook from using them for scoring. The original metadata still contains labels. Masks specify which labels each step may use.

For a GNN, use all cells in the forward pass but select training nodes before computing loss:

```python
logits = model(node_features, edge_index)  # Use edge weights if required.
train_mask = masks["labelled_train"]
loss = criterion(logits[train_mask], train_targets[train_mask])
```

This assumes the loaded arrays have been converted to tensors for the chosen framework. Do not use `train_pool` as the loss mask for the small budget. Do not compute loss on the full target array containing -1 values.

Cell types are used to stratify the original split and sample training labels. This defines the experiment without exposing held-out labels to fitting. Validation scores intentionally guide model selection, so final performance must be reported on the test split after those choices are fixed.

PCA and gene selection can use all cells' expression values under this **transductive** protocol. This evaluates labels for the existing cells, not performance on unseen donors. These steps keep held-out labels out of fitting and graph construction. Earlier label-based QC, exploratory analysis and test inspection still affect evaluation independence.
