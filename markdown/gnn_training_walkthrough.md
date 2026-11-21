# GNN training walkthrough

`03-gnn_training.ipynb` implements Step 3 using the exports from
`02-graph_engineering_baselines.ipynb`. The models are written in
`src/models.py`; training and experiment orchestration are in
`src/pipeline_train.py`.

## Architecture coverage

| Model       | Implementation   | Message passing                                        |
|-------------|------------------|--------------------------------------------------------|
| GCN         | `GCNLayer`       | Weighted symmetric normalization with self-loops       |
| GraphSAGE   | `GraphSAGELayer` | Weighted neighbor mean concatenated with self features |
| GAT         | `GATLayer`       | Multi-head destination-wise attention softmax          |
| Homophily-gated GCN | `HomophilyGatedLayer` | Learnable element-wise balance of self and neighbor means |

All aggregation uses PyTorch `index_add_` and scatter reduction. PyTorch
Geometric is not required. Saved adjacency matrices are kept sparse until
conversion to edge lists.

## Run order

1. Install the environment as described in the [README](../README.md).
2. Run `01-data_processing.ipynb`.
3. Run `02-graph_engineering_baselines.ipynb` to create
   `data/graph_engineering_50pc/`.
4. Run `03-gnn_training.ipynb` top to bottom. `RUN_MAIN` runs the 48 main
   trainings and `RUN_ABLATIONS` the scarcity and depth grids. CUDA is used
   when available; on a CPU the full grid takes several hours.


## Evaluation protocol

The graph is transductive. Features and edges for every cell are available to
every forward pass. Labels are used as follows:

| Label set        | Use                                                    |
|------------------|--------------------------------------------------------|
| `labelled_train` | Supervised loss only                                   |
| `validation`     | Macro-F1 checkpoint selection and reporting            |
| `test`           | Reserved for final evaluation after settings are fixed |

Do not use `train_pool` as the loss mask for a partial budget. The training
labels outside `labelled_train` remain `-1` and are never passed to
cross-entropy. The graph and feature construction remain label-free.

`load_data` uses the existing budget exports when present. For missing scarcity
budgets (`1pct`, `2pct`, `10pct`, or `20pct`), it samples at least one cell per
class from `train_pool` with a deterministic seed and fits a scaler only on the
selected nodes.

## Outputs

Results are written under `data/gnn_training_50pc/`:

- `validation_results.csv`: one row per model, graph, budget, depth, and seed
- `*.pt`: validation-selected model state dictionaries
- `*.json`: metrics and training history for each checkpoint
- `gated_gcn_*_gate_values.npy`: node-wise sigmoid gate values from the
   validation-selected gated-GCN checkpoint, in `cell_ids.csv` order
- `gated_gcn_*_gate_values_metadata.csv`: cell IDs, cell types, split masks,
   and empirical local homophily aligned row-for-row with the gate array
- `scarcity/` and `depth/`: optional ablation outputs from the notebook
- `plots/`: `model_comparison.png`, `scarcity_curve.png`, `gate_vs_homophily.png`,
   `training_dynamics.png`, drawn by the last cell from the saved results

The gate array has shape `(n_cells, hidden_features)` for the final gated
layer. Use its row mean for a scalar gate per cell, then group by `celltype`
and compare with `local_homophily`. The exporter is also available as
`pipeline_train.export_gate_values` for previously saved checkpoints.

Each result includes accuracy, macro-F1, per-class F1, best epoch, epochs run,
model depth, and (for the gated model) mean gate values.

## Reproducibility and reporting

The pipeline seeds Python, NumPy, and PyTorch for every run. Report mean and
standard deviation across seeds for accuracy and macro-F1. Include all 11
per-class F1 values, especially the rare CD4+ T Helper2 class and the
heterophilous T-cell subtypes. Compare the gated model against the ungated GCN
on the same graph, budget, depth, and seeds.

Test scoring should be added only after validation choices are frozen. The
current Step 3 pipeline intentionally writes validation results and checkpoints
without consuming test labels.
