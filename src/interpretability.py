"""
Step 4 evaluation and interpretability utilities.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import seaborn as sns
import torch
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

try:
    from .models import GNNClassifier
    from .pipeline_train import ExperimentData, load_data
except ImportError:
    from models import GNNClassifier
    from pipeline_train import ExperimentData, load_data


T_SUBTYPES = (
    "CD4+ T Helper2",
    "CD4+/CD25 T Reg",
    "CD4+/CD45RA+/CD25- Naive T",
    "CD4+/CD45RO+ Memory",
    "CD8+ Cytotoxic T",
    "CD8+/CD45RA+ Naive Cytotoxic",
)

# Canonical lineage markers named in the team contract, mapped to the fine
# classes whose logit they should drive. CD34 was removed by QC, so the CD34+
# class has no canonical marker in this dataset.
CANONICAL_MARKERS = {
    "CD3E": T_SUBTYPES,
    "MS4A1": ("CD19+ B",),
    "NKG7": ("CD56+ NK",),
    "LYZ": ("CD14+ Monocyte",),
}


def to_coarse(label: str) -> str:
    """Collapse T-cell subtypes into one label (same rule as notebook 01)."""
    label = str(label)
    t_markers = ("T Helper", "Naive T", "Memory", "T Reg", "Cytotoxic")
    is_t = any(marker in label for marker in t_markers)
    return "T cell" if is_t or label.startswith(("CD8", "CD4")) else label


# ---------------------------------------------------------------------------
# Loading: data, checkpoints and the MLP baseline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunSpec:
    """One Step 3 training run, matching the checkpoint naming scheme."""

    model: str
    budget: str
    graph: str
    depth: int
    seed: int
    experiment: str = "main"
    hidden_features: int = 64

    @property
    def stem(self) -> str:
        return (
            f"{self.model}_{self.budget}_{self.graph}_depth{self.depth}_seed{self.seed}"
        )

    def checkpoint(self, checkpoint_dirs: dict[str, Path]) -> Path:
        return Path(checkpoint_dirs[self.experiment]) / f"{self.stem}.pt"

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> RunSpec:
        """
        Read the run settings back from a checkpoint path.
        """
        path = Path(path)
        model, budget, graph, depth, seed = path.stem.rsplit("_", 4)
        folder = path.parent.name
        return cls(
            model,
            budget,
            graph,
            int(depth.removeprefix("depth")),
            int(seed.removeprefix("seed")),
            experiment=folder if folder in ("scarcity", "depth") else "main",
        )


def run_grid(
    experiment: str,
    models: Iterable[str],
    budgets: Iterable[str],
    graphs: Iterable[str],
    depths: Iterable[int],
    seeds: Iterable[int],
) -> list[RunSpec]:
    """
    Every combination of the given settings for one Step 3 experiment.
    """
    return [
        RunSpec(model, budget, graph, depth, seed, experiment=experiment)
        for budget, graph, depth, model, seed in product(
            budgets, graphs, depths, models, seeds
        )
    ]


class DataCache:
    """Load each (budget, graph) export once, exactly as run_experiments did."""

    def __init__(self, export_root: str | Path) -> None:
        self.export_root = Path(export_root)
        self._data: dict[tuple[str, str], ExperimentData] = {}

    def get(self, budget: str, graph: str) -> ExperimentData:
        key = (budget, graph)
        if key not in self._data:
            # run_experiments always samples missing budgets with seed 42.
            self._data[key] = load_data(self.export_root, budget, graph, seed=42)
        return self._data[key]


def all_labels(data: ExperimentData) -> np.ndarray:
    """Integer labels for every cell, including validation and test cells."""
    class_to_id = {name: index for index, name in enumerate(data.class_names)}
    return np.array([class_to_id[name] for name in data.cell_types], dtype=np.int64)


def load_model(
    spec: RunSpec, data: ExperimentData, checkpoint_dirs: dict[str, Path]
) -> GNNClassifier:
    """Rebuild a model and load its validation-selected state dict."""
    model = GNNClassifier(
        spec.model,
        data.features.shape[1],
        spec.hidden_features,
        len(data.class_names),
        depth=spec.depth,
    )
    state = torch.load(spec.checkpoint(checkpoint_dirs), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def untrained_model(
    model: str,
    data: ExperimentData,
    depth: int,
    hidden_features: int = 64,
    seed: int = 0,
) -> GNNClassifier:
    """
    Randomly initialised model with the same architecture, as a control.
    """
    torch.manual_seed(seed)
    return GNNClassifier(
        model,
        data.features.shape[1],
        hidden_features,
        len(data.class_names),
        depth=depth,
    ).eval()


def _tensors(data: ExperimentData) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(data.features),
        torch.from_numpy(data.edge_index),
        torch.from_numpy(data.edge_weight),
    )


def predict(
    model: GNNClassifier, data: ExperimentData
) -> tuple[np.ndarray, np.ndarray | None]:
    """Full-graph forward pass. Returns logits and, for gated GCN, gate values."""
    features, edge_index, edge_weight = _tensors(data)
    with torch.no_grad():
        logits = model(features, edge_index, edge_weight)
    gate = model.gate_values()
    gate_array = None if gate is None else gate.cpu().numpy()
    return logits.cpu().numpy(), gate_array


def split_mask(data: ExperimentData, split: str) -> np.ndarray:
    """Return the evaluation mask for ``validation`` or ``test``."""
    if split not in ("validation", "test"):
        raise ValueError(f"split must be 'validation' or 'test', got {split!r}")
    return data.validation_mask if split == "validation" else data.test_mask


def load_mlp_baseline(
    prediction_paths: dict[str, Path], cache: DataCache
) -> pd.DataFrame:
    """
    Score notebook 02's saved MLP predictions on the validation split.
    """
    rows = []
    for budget, path in prediction_paths.items():
        data = cache.get(budget, "knn")
        predictions = np.load(path, allow_pickle=False)
        mask = data.validation_mask
        targets = all_labels(data)
        metrics = score(predictions[mask], targets[mask], data.class_names)
        rows.append({"budget": budget, "model": "mlp", "macro_f1": metrics["macro_f1"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Method 1: held-out scoring and seed summaries
# ---------------------------------------------------------------------------


def score(
    predictions: np.ndarray, targets: np.ndarray, class_names: list[str]
) -> dict[str, Any]:
    """Accuracy, macro-F1 and per-class F1 over a fixed class list."""
    per_class = f1_score(
        targets,
        predictions,
        labels=np.arange(len(class_names)),
        average=None,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(np.mean(per_class)),
        "per_class_f1": dict(zip(class_names, np.asarray(per_class).tolist())),
    }


@dataclass
class RunOutput:
    """Predictions for every cell; ``gate`` is the per-cell mean gate or None."""

    spec: RunSpec
    predictions: np.ndarray
    targets: np.ndarray
    mask: np.ndarray
    gate: np.ndarray | None


def evaluate_runs(
    specs: Iterable[RunSpec],
    cache: DataCache,
    checkpoint_dirs: dict[str, Path],
    split: str = "validation",
) -> tuple[pd.DataFrame, dict[RunSpec, RunOutput]]:
    """Score every checkpoint on one split. A missing checkpoint is an error."""
    rows: list[dict[str, Any]] = []
    outputs: dict[RunSpec, RunOutput] = {}
    for spec in specs:
        data = cache.get(spec.budget, spec.graph)
        mask = split_mask(data, split)
        logits, gate = predict(load_model(spec, data, checkpoint_dirs), data)
        predictions = logits.argmax(axis=1)
        targets = all_labels(data)
        metrics = score(predictions[mask], targets[mask], data.class_names)
        per_class = metrics.pop("per_class_f1")
        row: dict[str, Any] = {
            "experiment": spec.experiment,
            "model": spec.model,
            "budget": spec.budget,
            "graph": spec.graph,
            "depth": spec.depth,
            "seed": spec.seed,
            **metrics,
            **{f"f1::{name}": value for name, value in per_class.items()},
        }
        if gate is not None:
            row["mean_gate"] = float(gate[mask].mean())
            # One gate value per cell: 1 keeps the cell's own representation,
            # 0 takes the neighbour mean.
            gate = gate.mean(axis=1)
        rows.append(row)
        outputs[spec] = RunOutput(spec, predictions, targets, mask, gate)
    return pd.DataFrame(rows), outputs


def seed_summary(
    results: pd.DataFrame,
    keys: tuple[str, ...] = ("experiment", "budget", "graph", "model", "depth"),
    metrics: tuple[str, ...] = ("accuracy", "macro_f1"),
) -> pd.DataFrame:
    """Mean, standard deviation and seed count for each configuration."""
    summary = results.groupby(list(keys))[list(metrics)].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def headline_table(
    summary: pd.DataFrame,
    mlp: pd.DataFrame,
    models: list[str],
    budgets: list[str],
) -> pd.DataFrame:
    """
    Mean ± sd macro-F1 per setting and model, with the MLP as a last column.
    """
    score_text = (
        summary["macro_f1_mean"].map("{:.3f}".format)
        + " ± "
        + summary["macro_f1_std"].map("{:.3f}".format)
    )
    table = summary.assign(
        experiment=pd.Categorical(
            summary["experiment"], ["main", "scarcity", "depth"], ordered=True
        ),
        budget=pd.Categorical(summary["budget"], budgets, ordered=True),
        score=score_text,
    ).pivot_table(
        index=["experiment", "budget", "graph", "depth"],
        columns="model",
        values="score",
        aggfunc="first",
        observed=True,
    )
    mlp_text = mlp.set_index("budget")["macro_f1"].map("{:.3f}".format)
    row_budgets = table.index.get_level_values("budget").astype(str)
    return table[models].assign(mlp=row_budgets.map(mlp_text)).fillna("")


def paired_seed_delta(
    results: pd.DataFrame,
    model_a: str,
    model_b: str,
    metric: str = "macro_f1",
    keys: tuple[str, ...] = ("experiment", "budget", "graph", "depth"),
) -> pd.DataFrame:
    """Per-seed difference ``model_a - model_b`` on identical settings."""
    index = list(keys) + ["seed"]
    a = results[results["model"] == model_a].set_index(index)[metric]
    b = results[results["model"] == model_b].set_index(index)[metric]
    delta = (a - b).dropna().rename("delta").reset_index()
    return (
        delta.groupby(list(keys))["delta"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Method 2: confusion structure and label granularity
# ---------------------------------------------------------------------------


def confusion_table(
    predictions: np.ndarray, targets: np.ndarray, class_names: list[str]
) -> pd.DataFrame:
    """
    Row-normalised confusion matrix (rows: true class, columns: predicted).
    """
    matrix = confusion_matrix(
        targets, predictions, labels=np.arange(len(class_names)), normalize="true"
    )
    return pd.DataFrame(matrix, index=class_names, columns=class_names)


def coarse_score(
    predictions: np.ndarray, targets: np.ndarray, class_names: list[str]
) -> dict[str, Any]:
    """
    Re-score after merging T subtypes without retraining.
    """
    coarse_names = sorted({to_coarse(name) for name in class_names})
    coarse_id = {name: index for index, name in enumerate(coarse_names)}
    fine_to_coarse = np.array([coarse_id[to_coarse(name)] for name in class_names])
    return score(fine_to_coarse[predictions], fine_to_coarse[targets], coarse_names)


def within_t_error_share(
    predictions: np.ndarray, targets: np.ndarray, class_names: list[str]
) -> float:
    """
    Fraction of all errors that are one T subtype mistaken for another.
    """
    is_t = np.array([name in T_SUBTYPES for name in class_names])
    wrong = predictions != targets
    if not wrong.any():
        return float("nan")
    within_t = wrong & is_t[targets] & is_t[predictions]
    return float(within_t.sum() / wrong.sum())


def granularity_table(
    outputs: dict[RunSpec, RunOutput], cache: DataCache, experiment: str = "main"
) -> pd.DataFrame:
    """
    Fine vs T-merged macro-F1 and within-T error share, averaged over seeds.
    """
    rows = []
    for spec, run in outputs.items():
        if spec.experiment != experiment:
            continue
        names = cache.get(spec.budget, spec.graph).class_names
        predictions, targets = run.predictions[run.mask], run.targets[run.mask]
        rows.append(
            {
                "model": spec.model,
                "budget": spec.budget,
                "graph": spec.graph,
                "seed": spec.seed,
                "fine_macro_f1": score(predictions, targets, names)["macro_f1"],
                "coarse_macro_f1": coarse_score(predictions, targets, names)[
                    "macro_f1"
                ],
                "within_t_error_share": within_t_error_share(
                    predictions, targets, names
                ),
            }
        )
    return (
        pd.DataFrame(rows)
        .groupby(["budget", "graph", "model"])[
            ["fine_macro_f1", "coarse_macro_f1", "within_t_error_share"]
        ]
        .mean()
    )


# ---------------------------------------------------------------------------
# Method 3 and 4: homophily on the training graph and the gate
# ---------------------------------------------------------------------------


def _same_label_counts(
    adjacency: sp.spmatrix, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-cell neighbour count and same-label neighbour count without self-loops.
    """
    graph = sp.csr_matrix(adjacency, dtype=np.float64)
    graph = graph - sp.diags(graph.diagonal())
    graph.eliminate_zeros()
    degree = np.diff(graph.indptr)
    rows = np.repeat(np.arange(graph.shape[0]), degree)
    same = (labels[rows] == labels[graph.indices]).astype(float)
    return degree, np.bincount(rows, weights=same, minlength=graph.shape[0])


def local_homophily(adjacency: sp.spmatrix, labels: np.ndarray) -> np.ndarray:
    """
    Per-cell fraction of neighbours sharing the cell label.
    """
    degree, same = _same_label_counts(adjacency, labels)
    return np.divide(same, degree, out=np.full(len(degree), np.nan), where=degree > 0)


def class_homophily(
    adjacency: sp.spmatrix, labels: np.ndarray, class_names: list[str]
) -> pd.DataFrame:
    """
    Per-class endpoint and mean local homophily (notebook 02 definitions).
    """
    degree, same = _same_label_counts(adjacency, labels)
    local = np.divide(same, degree, out=np.full(len(degree), np.nan), where=degree > 0)
    records = []
    for class_id, name in enumerate(class_names):
        members = labels == class_id
        endpoints = degree[members].sum()
        records.append(
            {
                "celltype": name,
                "cells": int(members.sum()),
                "endpoint_homophily": (
                    float(same[members].sum() / endpoints) if endpoints else np.nan
                ),
                "mean_local_homophily": float(np.nanmean(local[members]))
                if members.any()
                else np.nan,
            }
        )
    return pd.DataFrame(records)


def per_class_vs_homophily(
    results: pd.DataFrame,
    homophily: pd.DataFrame,
    class_names: list[str],
    budget: str,
    graph: str,
) -> pd.DataFrame:
    """
    Seed-mean per-class F1 of each model next to the class homophily.
    """
    scores = results[
        (results["experiment"] == "main")
        & (results["budget"] == budget)
        & (results["graph"] == graph)
    ]
    per_class = scores.groupby("model")[[f"f1::{c}" for c in class_names]].mean().T
    per_class.index = class_names
    per_class = per_class.join(homophily.set_index("celltype"))
    per_class["gate_gain"] = per_class["gated_gcn"] - per_class["gcn"]
    return per_class


def permutation_spearman(
    x: np.ndarray, y: np.ndarray, n_perm: int = 10000, seed: int = 0
) -> dict[str, float]:
    """
    Spearman's rho with a two-sided permutation p-value.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    rho = float(spearmanr(x, y).statistic)
    generator = np.random.default_rng(seed)
    null = np.array(
        [spearmanr(x, generator.permutation(y)).statistic for _ in range(n_perm)]
    )
    p_value = float((np.sum(np.abs(null) >= abs(rho)) + 1) / (n_perm + 1))
    return {"rho": rho, "p_permutation": p_value, "n": int(keep.sum())}


def homophily_tests(
    per_class: pd.DataFrame, columns: list[str], x: str = "endpoint_homophily"
) -> pd.DataFrame:
    """
    Permutation Spearman of each column against class homophily.
    """
    return pd.DataFrame(
        [
            {
                "quantity": column,
                **permutation_spearman(per_class[x], per_class[column]),
            }
            for column in columns
        ]
    )


def gate_homophily_correlation(
    gate: np.ndarray,
    homophily: np.ndarray,
    cell_types: np.ndarray,
    mask: np.ndarray,
    class_names: list[str],
    n_perm: int = 10000,
) -> dict[str, Any]:
    """
    Correlate the per-cell gate with local homophily at three levels.
    """
    keep = mask & np.isfinite(homophily)
    overall = spearmanr(gate[keep], homophily[keep])
    within = []
    for name in class_names:
        members = keep & (cell_types == name)
        if members.sum() >= 10 and np.ptp(homophily[members]) > 0:
            result = spearmanr(gate[members], homophily[members])
            within.append(
                {
                    "celltype": name,
                    "rho": float(result.statistic),
                    "p": float(result.pvalue),
                    "n": int(members.sum()),
                }
            )
        else:
            within.append(
                {"celltype": name, "rho": np.nan, "p": np.nan, "n": int(members.sum())}
            )
    class_means = (
        pd.DataFrame(
            {
                "celltype": cell_types[keep],
                "gate": gate[keep],
                "homophily": homophily[keep],
            }
        )
        .groupby("celltype")[["gate", "homophily"]]
        .mean()
    )
    class_means = class_means.reindex(class_names)
    return {
        "overall": {
            "rho": float(overall.statistic),
            "p": float(overall.pvalue),
            "n": int(keep.sum()),
        },
        "within_class": pd.DataFrame(within),
        "class_means": class_means,
        "class_level": permutation_spearman(
            class_means["gate"].to_numpy(),
            class_means["homophily"].to_numpy(),
            n_perm=n_perm,
        ),
    }


def gate_vs_homophily(
    runs: dict[int, RunOutput],
    homophily: np.ndarray,
    cell_types: np.ndarray,
    class_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Per-seed gate-homophily correlations, plus within-class and class-mean tables
    averaged over seeds.
    """
    rows, within_tables, class_tables = [], [], []
    for seed, run in runs.items():
        result = gate_homophily_correlation(
            run.gate, homophily, cell_types, run.mask, class_names
        )
        rows.append(
            {
                "seed": seed,
                "overall_rho": result["overall"]["rho"],
                "class_level_rho": result["class_level"]["rho"],
                "class_level_p": result["class_level"]["p_permutation"],
                "median_within_class_rho": result["within_class"]["rho"].median(),
            }
        )
        within_tables.append(result["within_class"].assign(seed=seed))
        class_tables.append(result["class_means"].assign(seed=seed))
    within_class = pd.concat(within_tables).groupby("celltype")[["rho", "n"]].mean()
    class_means = pd.concat(class_tables).groupby(level=0)[["gate", "homophily"]].mean()
    return pd.DataFrame(rows), within_class, class_means


# ---------------------------------------------------------------------------
# Method 5: gradient attribution projected through PCA loadings
# ---------------------------------------------------------------------------


def load_expression(
    path: str | Path, cell_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, sp.csr_matrix]:
    """
    PCA loadings, gene names, HVG mask and HVG log-expression from notebook 01.
    """
    work = ad.read_h5ad(path)
    assert np.array_equal(work.obs_names.to_numpy(dtype=str), cell_ids)
    loadings = np.asarray(work.varm["PCs"], dtype=np.float64)
    gene_names = work.var_names.to_numpy(dtype=str)
    hvg = work.var["highly_variable"].to_numpy(dtype=bool)
    expression = sp.csr_matrix(work.layers["log_normalized"])[:, np.flatnonzero(hvg)]
    return loadings, gene_names, hvg, expression


def check_pca_loadings(
    expression: sp.csr_matrix,
    loadings: np.ndarray,
    hvg: np.ndarray,
    pcs: np.ndarray,
    n_cells: int = 1000,
    seed: int = 0,
) -> float:
    """
    Smallest correlation between ``(x - mean) @ loadings`` and the saved PCs.
    """
    cells = np.random.default_rng(seed).choice(len(pcs), size=n_cells, replace=False)
    centred = expression[cells].toarray() - np.asarray(expression.mean(axis=0)).ravel()
    rebuilt = centred @ loadings[hvg, : pcs.shape[1]]
    return min(
        np.corrcoef(rebuilt[:, pc], pcs[cells, pc])[0, 1] for pc in range(pcs.shape[1])
    )


def pc_gradients(
    model: GNNClassifier, data: ExperimentData, cell_mask: np.ndarray
) -> np.ndarray:
    """
    Mean gradient of each class logit with respect to the 30 scaled PCs.
    """
    features, edge_index, edge_weight = _tensors(data)
    labels = all_labels(data)
    gradients = np.zeros((len(data.class_names), features.shape[1]))
    for class_id in range(len(data.class_names)):
        members = np.flatnonzero(cell_mask & (labels == class_id))
        if members.size == 0:
            gradients[class_id] = np.nan
            continue
        inputs = features.clone().requires_grad_(True)
        logits = model(inputs, edge_index, edge_weight)
        logits[torch.from_numpy(members), class_id].sum().backward()
        gradients[class_id] = inputs.grad.sum(dim=0).numpy() / members.size
    return gradients


def gene_attributions(
    gradients: np.ndarray,
    scale: np.ndarray,
    loadings: np.ndarray,
    gene_names: Iterable[str],
    class_names: list[str],
) -> pd.DataFrame:
    """
    Chain rule from scaled PCs back to log-normalised genes.
    """
    n_pcs = gradients.shape[1]
    scores = loadings[:, :n_pcs] @ (gradients / scale[:n_pcs]).T
    return pd.DataFrame(scores, index=list(gene_names), columns=class_names)


def mean_attribution(
    models: list[GNNClassifier],
    data: ExperimentData,
    cell_mask: np.ndarray,
    scale: np.ndarray,
    loadings: np.ndarray,
    gene_names: Iterable[str],
) -> pd.DataFrame:
    """
    Gene attributions (genes x classes) averaged over several models.
    """
    tables = [
        gene_attributions(
            pc_gradients(model, data, cell_mask),
            scale,
            loadings,
            gene_names,
            data.class_names,
        )
        for model in models
    ]
    return sum(tables) / len(tables)


def marker_recovery(
    attributions: pd.DataFrame,
    eligible: pd.Series,
    markers: dict[str, Iterable[str]] = CANONICAL_MARKERS,
) -> pd.DataFrame:
    """
    Rank of each canonical marker in its class's attribution.
    """
    pool = attributions.loc[eligible[eligible].index]
    rows = []
    for gene, classes in markers.items():
        for name in classes:
            if name not in pool.columns:
                continue
            row = {"marker": gene, "celltype": name, "in_hvg": gene in pool.index}
            if row["in_hvg"]:
                ranks = pool[name].rank(ascending=False, method="min")
                row.update(
                    {
                        "attribution": float(pool.at[gene, name]),
                        "rank": int(ranks[gene]),
                        "n_genes": len(pool),
                        "top_percent": float(100 * ranks[gene] / len(pool)),
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def recovery_table(
    attributions: dict[str, pd.DataFrame], eligible: pd.Series
) -> pd.DataFrame:
    """
    ``marker_recovery`` for every model, stacked with a ``model`` column.
    """
    return pd.concat(
        marker_recovery(table, eligible).assign(model=name)
        for name, table in attributions.items()
    )


# ---------------------------------------------------------------------------
# Dropout: GNN vs MLP accuracy by genes detected per cell
# ---------------------------------------------------------------------------


def genes_detected(path: str | Path, cell_ids: np.ndarray) -> np.ndarray:
    """Genes detected per cell (notebook 01 QC); fewer means more dropout."""
    obs = ad.read_h5ad(path, backed="r").obs
    assert np.array_equal(obs.index.to_numpy(dtype=str), cell_ids)
    return obs["n_genes_by_counts"].to_numpy()


def dropout_table(
    runs: dict[str, dict[int, RunOutput]],
    mlp_predictions: np.ndarray,
    n_genes: np.ndarray,
    cell_types: np.ndarray,
    n_bins: int = 4,
) -> pd.DataFrame:
    """
    Validation accuracy of each GNN and the MLP per dropout bin.

    Bins are quantiles of ``n_genes`` within each cell type, so every bin has
    the same cell-type mix. Bin 1 has the fewest genes detected (most dropout).
    """
    first = next(iter(next(iter(runs.values())).values()))
    mask, targets = first.mask, first.targets
    bins = np.zeros(len(targets), dtype=int)
    for cell_type in np.unique(cell_types[mask]):
        members = mask & (cell_types == cell_type)
        ranks = pd.Series(n_genes[members]).rank(method="first", pct=True)
        bins[members] = np.ceil(ranks.to_numpy() * n_bins).astype(int)

    mlp_correct = mlp_predictions == targets
    rows = []
    for model, seeds in runs.items():
        for seed, run in seeds.items():
            assert np.array_equal(run.mask, mask)
            correct = run.predictions == targets
            for level in range(1, n_bins + 1):
                cells = mask & (bins == level)
                rows.append(
                    {
                        "bin": level,
                        "model": model,
                        "seed": seed,
                        "n_cells": int(cells.sum()),
                        "median_genes": float(np.median(n_genes[cells])),
                        "mlp_accuracy": mlp_correct[cells].mean(),
                        "accuracy": correct[cells].mean(),
                    }
                )
    table = pd.DataFrame(rows)
    table["gnn_minus_mlp"] = table["accuracy"] - table["mlp_accuracy"]
    return table.groupby(["bin", "model"]).agg(
        n_cells=("n_cells", "first"),
        median_genes=("median_genes", "first"),
        mlp_accuracy=("mlp_accuracy", "first"),
        accuracy_mean=("accuracy", "mean"),
        gnn_minus_mlp_mean=("gnn_minus_mlp", "mean"),
        gnn_minus_mlp_std=("gnn_minus_mlp", "std"),
    )


# ---------------------------------------------------------------------------
# Method 6: UMAP embedding
# ---------------------------------------------------------------------------


def stratified_subsample(labels: np.ndarray, size: int, seed: int = 0) -> np.ndarray:
    """Sorted cell indices, sampled per class so rare classes stay visible."""
    if size >= len(labels):
        return np.arange(len(labels))
    generator = np.random.default_rng(seed)
    chosen = []
    for class_id in np.unique(labels):
        members = np.flatnonzero(labels == class_id)
        count = max(1, round(size * len(members) / len(labels)))
        chosen.append(
            generator.choice(members, size=min(count, len(members)), replace=False)
        )
    return np.sort(np.concatenate(chosen))


def umap_coords(
    labels: np.ndarray,
    pcs: np.ndarray,
    cache_path: str | Path,
    n_cells: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Subsampled cell indices and their 2-D UMAP of the PCs, cached on disk.
    """
    cache_path = Path(cache_path)
    if cache_path.is_file():
        cached = np.load(cache_path)
        return cached["subset"], cached["coords"]
    import umap

    subset = stratified_subsample(labels, n_cells, seed=seed)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=seed)
    coords = reducer.fit_transform(pcs[subset])
    np.savez(cache_path, subset=subset, coords=coords)
    return subset, coords


# ---------------------------------------------------------------------------
# Plots: each returns the figure so the notebook decides where to save it
# ---------------------------------------------------------------------------


def plot_overview(
    results: pd.DataFrame,
    mlp: pd.DataFrame,
    budgets: list[str],
    graph: str,
    depth_budget: str,
) -> plt.Figure:
    """
    Main comparison, label scarcity and depth panels, with the MLP baseline.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    main = results[(results["experiment"] == "main") & (results["graph"] == graph)]
    sns.barplot(
        data=pd.concat([main, mlp]),
        x="budget",
        y="macro_f1",
        hue="model",
        errorbar="sd",
        ax=axes[0],
    )
    axes[0].set_title(f"Main comparison ({graph}) vs MLP")

    scarcity = results[results["experiment"] == "scarcity"].copy()
    scarcity["budget"] = pd.Categorical(scarcity["budget"], budgets, ordered=True)
    sns.lineplot(
        data=scarcity,
        x="budget",
        y="macro_f1",
        hue="model",
        marker="o",
        errorbar="sd",
        ax=axes[1],
    )
    axes[1].scatter(
        mlp["budget"], mlp["macro_f1"], marker="s", color="black", label="mlp"
    )
    axes[1].legend()
    axes[1].set_title("Label scarcity (kNN) vs MLP")

    depth = results[results["experiment"] == "depth"]
    sns.lineplot(
        data=depth,
        x="depth",
        y="macro_f1",
        hue="model",
        marker="o",
        errorbar="sd",
        ax=axes[2],
    )
    axes[2].set_title(f"Depth ({depth_budget} labels, kNN)")
    axes[2].set_xticks(sorted(depth["depth"].unique()))
    for axis in axes:
        axis.set_ylabel("validation macro-F1 (GNNs: mean ± sd over seeds)")
    fig.tight_layout()
    return fig


def plot_confusions(
    runs: dict[str, RunOutput], class_names: list[str], title: str
) -> plt.Figure:
    """
    Row-normalised confusion matrix for each model, side by side.
    """
    fig, axes = plt.subplots(1, len(runs), figsize=(8 * len(runs), 7))
    for axis, (model, run) in zip(np.atleast_1d(axes), runs.items()):
        table = confusion_table(
            run.predictions[run.mask], run.targets[run.mask], class_names
        )
        sns.heatmap(
            table,
            vmin=0,
            vmax=1,
            cmap="Blues",
            annot=True,
            fmt=".2f",
            cbar=False,
            ax=axis,
        )
        axis.set_title(f"{model} | {title}")
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
    fig.tight_layout()
    return fig


def plot_per_class_vs_homophily(per_class: pd.DataFrame) -> plt.Figure:
    """
    Per-class F1 and the gated-minus-GCN gain against class homophily.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for model in ["gcn", "gated_gcn"]:
        axes[0].scatter(per_class["endpoint_homophily"], per_class[model], label=model)
    axes[1].axhline(0, color="grey", lw=1)
    axes[1].scatter(per_class["endpoint_homophily"], per_class["gate_gain"])
    for name, row in per_class.iterrows():
        axes[0].annotate(
            name, (row["endpoint_homophily"], row["gated_gcn"]), fontsize=7
        )
        axes[1].annotate(
            name, (row["endpoint_homophily"], row["gate_gain"]), fontsize=7
        )
    axes[0].set_ylabel("validation F1")
    axes[0].legend()
    axes[1].set_ylabel("F1 gain: gated GCN - GCN")
    for axis in axes:
        axis.set_xlabel("Class endpoint homophily (training graph)")
    fig.tight_layout()
    return fig


def plot_gate_vs_homophily(
    run: RunOutput, homophily: np.ndarray, class_means: pd.DataFrame, seed: int
) -> plt.Figure:
    """
    Per-cell gate vs local homophily (one seed) and the class means.
    """
    keep = run.mask & np.isfinite(homophily)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hexbin(homophily[keep], run.gate[keep], gridsize=40, mincnt=1)
    axes[0].set_xlabel("Local homophily (training graph)")
    axes[0].set_ylabel("Mean gate (1 = keep self)")
    axes[0].set_title(f"Cells, seed {seed}")
    axes[1].scatter(class_means["homophily"], class_means["gate"])
    for name, row in class_means.iterrows():
        axes[1].annotate(name, (row["homophily"], row["gate"]), fontsize=7)
    axes[1].set_xlabel("Class mean local homophily")
    axes[1].set_ylabel("Class mean gate")
    axes[1].set_title("Class level (mean over seeds)")
    fig.tight_layout()
    return fig


def plot_umap(
    coords: np.ndarray, colourings: dict[str, np.ndarray], title: str
) -> plt.Figure:
    """
    One UMAP panel per colouring: categorical arrays get a legend, floats a colorbar.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=72)
    for axis, (name, values) in zip(axes.flat, colourings.items()):
        if values.dtype.kind == "f":
            points = axis.scatter(*coords.T, c=values, s=1, cmap="viridis")
            fig.colorbar(points, ax=axis)
        else:
            for index, value in enumerate(np.unique(values)):
                axis.scatter(
                    *coords[values == value].T,
                    s=1,
                    color=plt.cm.tab20(index),
                    label=value,
                )
            axis.legend(markerscale=8, fontsize=7)
        axis.set_title(name)
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    return fig
