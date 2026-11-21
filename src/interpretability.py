"""
Step 4 evaluation and interpretability utilities.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
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
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

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

NAIVE_T = ("CD4+/CD45RA+/CD25- Naive T", "CD8+/CD45RA+ Naive Cytotoxic")

# Canonical PBMC markers mapped to the fine classes whose logit they should
# drive. The first five are the team contract. Only HVGs enter the PCA input,
# so a marker outside the HVGs cannot influence any model.
CANONICAL_MARKERS = {
    "CD3E": T_SUBTYPES,
    "MS4A1": ("CD19+ B",),
    "NKG7": ("CD56+ NK",),
    "LYZ": ("CD14+ Monocyte",),
    "CD34": ("CD34+",),
    "CD3D": T_SUBTYPES,
    "CD8A": ("CD8+ Cytotoxic T", "CD8+/CD45RA+ Naive Cytotoxic"),
    "GZMH": ("CD8+ Cytotoxic T",),
    "CCL5": ("CD8+ Cytotoxic T",),
    "CCR7": NAIVE_T,
    "SELL": NAIVE_T,
    "IL7R": ("CD4+/CD45RO+ Memory",),
    "S100A4": ("CD4+/CD45RO+ Memory",),
    "FOXP3": ("CD4+/CD25 T Reg",),
    "IL2RA": ("CD4+/CD25 T Reg",),
    "GATA3": ("CD4+ T Helper2",),
    "PTGDR2": ("CD4+ T Helper2",),
    "CD79A": ("CD19+ B",),
    "GNLY": ("CD56+ NK",),
    "S100A8": ("CD14+ Monocyte",),
    "FCER1A": ("Dendritic",),
    "CLEC10A": ("Dendritic",),
    "PRSS57": ("CD34+",),
    "MYB": ("CD34+",),
}


def to_coarse(label: str) -> str:
    """Collapse T-cell subtypes into one label (same rule as notebook 01)."""
    label = str(label)
    t_markers = ("T Helper", "Naive T", "Memory", "T Reg", "Cytotoxic")
    is_t = any(marker in label for marker in t_markers)
    return "T cell" if is_t or label.startswith(("CD8", "CD4")) else label


# Names, colours and short labels used in every figure, matching the report.
MODEL_LABELS = {
    "mlp": "MLP",
    "gcn": "GCN",
    "graphsage": "GraphSAGE",
    "gat": "GAT",
    "gated_gcn": "Gated GCN",
    "untrained_gated_gcn": "Untrained Gated GCN",
}
MODEL_COLOURS = {
    "mlp": "black",
    "gcn": "tab:blue",
    "graphsage": "tab:orange",
    "gat": "tab:green",
    "gated_gcn": "tab:red",
}
BUDGET_LABELS = {
    "1pct": "1%",
    "2pct": "2%",
    "5pct": "5%",
    "10pct": "10%",
    "20pct": "20%",
    "full": "Full",
}
GRAPH_LABELS = {"knn": "$k$-NN", "snn": "SNN"}
SHORT_CLASS_NAMES = {
    "CD4+/CD25 T Reg": "CD4+ T Reg",
    "CD4+/CD45RA+/CD25- Naive T": "CD4+ Naive T",
    "CD4+/CD45RO+ Memory": "CD4+ Memory",
    "CD8+/CD45RA+ Naive Cytotoxic": "CD8+ Naive Cytotoxic",
}


def model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def budget_label(budget: str) -> str:
    return BUDGET_LABELS.get(budget, budget)


def graph_label(graph: str) -> str:
    if graph in GRAPH_LABELS:
        return GRAPH_LABELS[graph]
    if graph.startswith("knn"):
        return f"$k$-NN ($k$ = {graph.removeprefix('knn')})"
    return graph


def class_label(name: str) -> str:
    return SHORT_CLASS_NAMES.get(name, name)


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
            experiment=folder
            if folder in ("scarcity", "depth", "neighbourhood")
            else "main",
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


SETTING_KEYS = ("experiment", "budget", "graph", "depth")


@dataclass(frozen=True)
class Setting:
    """One trained configuration: every model and seed share it."""

    experiment: str
    budget: str
    graph: str
    depth: int

    @property
    def name(self) -> str:
        return f"{self.experiment}_{self.budget}_{self.graph}_depth{self.depth}"

    @property
    def columns(self) -> dict[str, Any]:
        return dict(
            zip(SETTING_KEYS, (self.experiment, self.budget, self.graph, self.depth))
        )

    @classmethod
    def of(cls, spec: RunSpec) -> Setting:
        return cls(spec.experiment, spec.budget, spec.graph, spec.depth)


def group_runs(
    outputs: dict[RunSpec, RunOutput],
) -> dict[Setting, dict[str, dict[int, RunOutput]]]:
    """
    Nest run outputs as setting -> model -> seed, keeping the evaluation order.
    """
    grouped: dict[Setting, dict[str, dict[int, RunOutput]]] = {}
    for spec, run in outputs.items():
        models = grouped.setdefault(Setting.of(spec), {})
        models.setdefault(spec.model, {})[spec.seed] = run
    return grouped


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


# Notebook 02's MLP: one ReLU hidden layer (64 or 128 units, chosen on validation
# macro-F1), Adam, class-balanced sample weights, 200 epochs and patience 25.
MLP_WIDTHS = (64, 128)
MLP_SETTINGS = {
    "activation": "relu",
    "solver": "adam",
    "alpha": 1e-4,
    "batch_size": 256,
    "learning_rate_init": 1e-3,
    "early_stopping": False,
    "shuffle": True,
}


def _train_mlp(x_train, y_train, x_val, y_val, classes, settings, epochs, patience):
    """Notebook 02's MLP loop: keep the epoch with the best validation macro-F1."""
    scaler = StandardScaler().fit(x_train)
    xt = scaler.transform(x_train).astype(np.float32)
    xv = scaler.transform(x_val).astype(np.float32)
    weights = compute_sample_weight("balanced", y_train)
    settings = {**settings, "batch_size": min(settings["batch_size"], len(x_train))}
    model = MLPClassifier(**settings)
    best, best_score, stale = None, -np.inf, 0
    for _ in range(epochs):
        model.partial_fit(xt, y_train, classes=classes, sample_weight=weights)
        value = f1_score(
            y_val, model.predict(xv), labels=classes, average="macro", zero_division=0
        )
        if value > best_score + 1e-8:
            best, best_score, stale = copy.deepcopy(model), float(value), 0
        else:
            stale += 1
        if stale >= patience:
            break
    return best, scaler, best_score


def mlp_seed_predictions(
    cache: DataCache,
    budgets: Iterable[str],
    seeds: Iterable[int],
    cache_dir: str | Path,
    epochs: int = 200,
    patience: int = 25,
    threads: int = 4,
) -> dict[str, dict[int, np.ndarray]]:
    """
    Notebook 02's MLP baseline for every budget and seed, predicted for all cells.

    Only ``random_state`` changes with the seed. Masks are the GNN runs' masks, so
    seed 42 at 5pct and full reproduces notebook 02's saved predictions. Results
    are cached as ``{budget}_seed{seed}.npy`` in ``cache_dir``.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    pcs = np.load(cache.export_root / "X_pca.npy", allow_pickle=False)
    pcs = pcs.astype(np.float32)
    runs: dict[str, dict[int, np.ndarray]] = {}
    for budget in budgets:
        data = cache.get(budget, "knn")
        classes = np.arange(len(data.class_names))
        train, val = data.train_mask, data.validation_mask
        runs[budget] = {}
        for seed in seeds:
            path = cache_dir / f"{budget}_seed{seed}.npy"
            if not path.is_file():
                fits = []
                with threadpool_limits(limits=threads):
                    for width in MLP_WIDTHS:
                        settings = {
                            **MLP_SETTINGS,
                            "random_state": seed,
                            "hidden_layer_sizes": (width,),
                        }
                        fits.append(
                            _train_mlp(
                                pcs[train],
                                data.labels[train],
                                pcs[val],
                                data.validation_labels[val],
                                classes,
                                settings,
                                epochs,
                                patience,
                            )
                        )
                    model, scaler, _ = max(fits, key=lambda fit: fit[2])
                    scaled = scaler.transform(pcs).astype(np.float32)
                    np.save(path, model.predict(scaled).astype(np.int64))
            runs[budget][seed] = np.load(path, allow_pickle=False)
    return runs


def load_mlp_baseline(
    predictions: dict[str, dict[int, np.ndarray]], cache: DataCache
) -> pd.DataFrame:
    """
    Validation scores of the MLP, one row per budget and seed.
    """
    rows = []
    for budget, seeds in predictions.items():
        data = cache.get(budget, "knn")
        mask = data.validation_mask
        targets = all_labels(data)
        for seed, predicted in seeds.items():
            metrics = score(predicted[mask], targets[mask], data.class_names)
            per_class = metrics.pop("per_class_f1")
            rows.append(
                {
                    "budget": budget,
                    "model": "mlp",
                    "seed": seed,
                    **metrics,
                    **{f"f1::{name}": value for name, value in per_class.items()},
                }
            )
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
            summary["experiment"],
            ["main", "scarcity", "depth", "neighbourhood"],
            ordered=True,
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
    mlp_stats = mlp.groupby("budget")["macro_f1"].agg(["mean", "std"])
    mlp_text = (
        mlp_stats["mean"].map("{:.3f}".format)
        + " ± "
        + mlp_stats["std"].map("{:.3f}".format)
    )
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


def mean_confusion(runs: dict[int, RunOutput], class_names: list[str]) -> pd.DataFrame:
    """
    Row-normalised confusion matrix averaged over seeds.
    """
    tables = [
        confusion_table(run.predictions[run.mask], run.targets[run.mask], class_names)
        for run in runs.values()
    ]
    return sum(tables) / len(tables)


def granularity_table(
    outputs: dict[RunSpec, RunOutput], cache: DataCache
) -> pd.DataFrame:
    """
    Fine vs T-merged macro-F1 and within-T error share for every setting and
    model, averaged over seeds.
    """
    rows = []
    for spec, run in outputs.items():
        names = cache.get(spec.budget, spec.graph).class_names
        predictions, targets = run.predictions[run.mask], run.targets[run.mask]
        rows.append(
            {
                **Setting.of(spec).columns,
                "model": spec.model,
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
        .groupby([*SETTING_KEYS, "model"], sort=False)[
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
    setting: Setting,
    mlp: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Seed-mean per-class F1 of each model in one setting next to the class
    homophily of that setting's graph. With ``mlp`` (from
    ``load_mlp_baseline``) the MLP of the same budget is added as a column.
    """
    scores = results[
        (results["experiment"] == setting.experiment)
        & (results["budget"] == setting.budget)
        & (results["graph"] == setting.graph)
        & (results["depth"] == setting.depth)
    ]
    per_class = (
        scores.groupby("model", sort=False)[[f"f1::{c}" for c in class_names]].mean().T
    )
    per_class.index = class_names
    if mlp is not None and (mlp["budget"] == setting.budget).any():
        rows = mlp[mlp["budget"] == setting.budget]
        per_class["mlp"] = [rows[f"f1::{c}"].mean() for c in class_names]
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
    Mean gradient of each class logit with respect to the scaled input PCs.
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
    Rank of each canonical marker in its class's attribution. ``in_data`` is
    False for genes removed in QC.
    """
    pool = attributions.loc[eligible[eligible].index]
    rows = []
    for gene, classes in markers.items():
        for name in classes:
            if name not in pool.columns:
                continue
            row = {
                "marker": gene,
                "celltype": name,
                "in_data": gene in attributions.index,
                "in_hvg": gene in pool.index,
            }
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
    mlp_predictions: dict[int, np.ndarray] | np.ndarray | None,
    n_genes: np.ndarray,
    cell_types: np.ndarray,
    n_bins: int = 4,
) -> pd.DataFrame:
    """
    Validation accuracy of each GNN and the MLP per dropout bin.

    Bins are quantiles of ``n_genes`` within each cell type, so every bin has
    the same cell-type mix. Bin 1 has the fewest genes detected (most dropout).
    With MLP predictions for several seeds the MLP accuracy is their mean.
    Without MLP predictions for the budget the MLP columns are NaN.
    """
    first = next(iter(next(iter(runs.values())).values()))
    mask, targets = first.mask, first.targets
    bins = np.zeros(len(targets), dtype=int)
    for cell_type in np.unique(cell_types[mask]):
        members = mask & (cell_types == cell_type)
        ranks = pd.Series(n_genes[members]).rank(method="first", pct=True)
        bins[members] = np.ceil(ranks.to_numpy() * n_bins).astype(int)

    if mlp_predictions is None:
        mlp_correct = np.full(len(targets), np.nan)
    else:
        if isinstance(mlp_predictions, np.ndarray):
            mlp_predictions = {0: mlp_predictions}
        mlp_correct = np.mean(
            [(predicted == targets) for predicted in mlp_predictions.values()], axis=0
        )
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
    return table.groupby(["bin", "model"], sort=False).agg(
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
    # Reuse the output folder safely when the PCs, cell order or sample changes.
    digest = sha256()
    for array in (labels, pcs):
        array = np.ascontiguousarray(array)
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    digest.update(str((n_cells, seed)).encode())
    fingerprint = digest.hexdigest()
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cached:
            if "fingerprint" in cached and str(cached["fingerprint"]) == fingerprint:
                return cached["subset"], cached["coords"]
    import umap

    subset = stratified_subsample(labels, n_cells, seed=seed)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=seed)
    coords = reducer.fit_transform(pcs[subset])
    np.savez(cache_path, subset=subset, coords=coords, fingerprint=fingerprint)
    return subset, coords


# ---------------------------------------------------------------------------
# Plots: each returns the figure so the notebook decides where to save it
# ---------------------------------------------------------------------------


def plot_overview(
    results: pd.DataFrame, mlp: pd.DataFrame, budgets: list[str]
) -> plt.Figure:
    """
    One main-comparison panel per graph, then label scarcity, depth and
    neighbourhood size, with the MLP baseline.
    """
    results = _with_labels(results)
    mlp = _with_labels(mlp)
    palette = {model_label(m): c for m, c in MODEL_COLOURS.items()}
    graphs = list(dict.fromkeys(results.loc[results["experiment"] == "main", "graph"]))
    has_neighbourhood = (results["experiment"] == "neighbourhood").any()
    panels = len(graphs) + 2 + int(has_neighbourhood)
    fig, axes = plt.subplots(1, panels, figsize=(7 * panels, 5))
    for axis, graph in zip(axes, graphs):
        main = results[(results["experiment"] == "main") & (results["graph"] == graph)]
        sns.barplot(
            data=pd.concat([main, mlp[mlp["budget"].isin(main["budget"])]]),
            x="Budget",
            y="macro_f1",
            hue="Model",
            palette=palette,
            errorbar="sd",
            ax=axis,
        )
        axis.set_title(f"Main comparison ({graph_label(graph)}) vs MLP")

    scarcity = pd.concat([results[results["experiment"] == "scarcity"], mlp])
    scarcity["Budget"] = pd.Categorical(
        scarcity["Budget"], [budget_label(b) for b in budgets], ordered=True
    )
    sns.lineplot(
        data=scarcity,
        x="Budget",
        y="macro_f1",
        hue="Model",
        palette=palette,
        marker="o",
        errorbar="sd",
        ax=axes[len(graphs)],
    )
    axes[len(graphs)].set_title("Label scarcity ($k$-NN) vs MLP")

    depth = results[results["experiment"] == "depth"]
    sns.lineplot(
        data=depth,
        x="depth",
        y="macro_f1",
        hue="Model",
        palette=palette,
        marker="o",
        errorbar="sd",
        ax=axes[len(graphs) + 1],
    )
    axes[len(graphs) + 1].set_title("Depth (5% labels, $k$-NN)")
    axes[len(graphs) + 1].set_xticks(sorted(depth["depth"].unique()))
    axes[len(graphs) + 1].set_xlabel("Message-passing layers")

    if has_neighbourhood:
        neighbourhood = results[results["experiment"] == "neighbourhood"].copy()
        neighbourhood["k"] = neighbourhood["graph"].str.removeprefix("knn").astype(int)
        sns.lineplot(
            data=neighbourhood,
            x="k",
            y="macro_f1",
            hue="Model",
            palette=palette,
            style="Budget",
            marker="o",
            errorbar="sd",
            ax=axes[-1],
        )
        axes[-1].set_title("Neighbourhood size $k$ ($k$-NN)")
        axes[-1].set_xticks(sorted(neighbourhood["k"].unique()))
    for axis in axes:
        axis.set_ylabel("Validation macro-F1 (mean ± sd over seeds)")
    fig.tight_layout()
    return fig


def _with_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy of a results table with report-style Model and Budget columns."""
    return frame.assign(
        Model=frame["model"].map(model_label),
        Budget=frame["budget"].map(budget_label),
    )


def plot_confusions(tables: dict[str, pd.DataFrame], title: str) -> plt.Figure:
    """
    One row-normalised confusion matrix per model, side by side.
    """
    fig, axes = plt.subplots(1, len(tables), figsize=(8 * len(tables), 7))
    for axis, (model, table) in zip(np.atleast_1d(axes), tables.items()):
        names = [class_label(name) for name in table.index]
        sns.heatmap(
            table.to_numpy(),
            vmin=0,
            vmax=1,
            cmap="Blues",
            annot=True,
            fmt=".2f",
            cbar=False,
            xticklabels=names,
            yticklabels=names,
            ax=axis,
        )
        axis.set_title(f"{model_label(model)} | {title}")
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
    fig.tight_layout()
    return fig


def plot_per_class_vs_homophily(
    per_class: pd.DataFrame, models: list[str], title: str
) -> plt.Figure:
    """
    Per-class F1 of every model and the gated-minus-GCN gain against class
    homophily.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for model in models:
        axes[0].scatter(
            per_class["endpoint_homophily"],
            per_class[model],
            label=model_label(model),
            color=MODEL_COLOURS.get(model),
        )
    if "mlp" in per_class:
        axes[0].scatter(
            per_class["endpoint_homophily"],
            per_class["mlp"],
            label="MLP",
            color="black",
            marker="x",
        )
    axes[1].axhline(0, color="grey", lw=1)
    axes[1].scatter(per_class["endpoint_homophily"], per_class["gate_gain"])
    for name, row in per_class.iterrows():
        axes[0].annotate(
            class_label(name),
            (row["endpoint_homophily"], row["gated_gcn"]),
            fontsize=7,
        )
        axes[1].annotate(
            class_label(name),
            (row["endpoint_homophily"], row["gate_gain"]),
            fontsize=7,
        )
    axes[0].set_ylabel("Validation F1 (mean over seeds)")
    axes[0].legend()
    axes[1].set_ylabel("F1 gain: Gated GCN - GCN")
    for axis in axes:
        axis.set_xlabel("Class endpoint homophily (training graph)")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_gate_vs_homophily(
    run: RunOutput,
    homophily: np.ndarray,
    class_means: pd.DataFrame,
    seed: int,
    title: str = "",
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
        axes[1].annotate(class_label(name), (row["homophily"], row["gate"]), fontsize=7)
    axes[1].set_xlabel("Class mean local homophily")
    axes[1].set_ylabel("Class mean gate")
    axes[1].set_title("Class level (mean over seeds)")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_umap(
    coords: np.ndarray,
    colourings: dict[str, np.ndarray],
    title: str,
    n_columns: int = 3,
) -> plt.Figure:
    """
    One UMAP panel per colouring: categorical arrays get a legend, floats a colorbar.
    """
    n_rows = int(np.ceil(len(colourings) / n_columns))
    fig, axes = plt.subplots(
        n_rows, n_columns, figsize=(5.5 * n_columns, 4.5 * n_rows), dpi=72
    )
    axes = np.atleast_1d(axes).ravel()
    for axis, (name, values) in zip(axes, colourings.items()):
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
    for axis in axes[len(colourings) :]:
        axis.set_visible(False)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Report figures: one graph per image, sized for one column of the report
# ---------------------------------------------------------------------------

REPORT_STYLE = {
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}
REPORT_SIZE = (6.4, 4.8)


def _place_labels(
    axis: plt.Axes,
    x: Iterable[float],
    y: Iterable[float],
    names: Iterable[str],
    size: int = 8,
) -> None:
    """
    Label points and push labels apart so they overlap neither each other nor
    the other points. Call once the axis limits and layout are final.
    """
    fig = axis.figure
    fig.canvas.draw()
    to_px = axis.transData
    points = to_px.transform(np.column_stack([np.asarray(x), np.asarray(y)]))
    height = size * fig.dpi / 72 * 1.25
    widths = [len(name) * size * fig.dpi / 72 * 0.55 for name in names]
    text = points + np.array([6.0, 0.0])
    for _ in range(300):
        moved = False
        for i in range(len(text)):
            for j in range(i + 1, len(text)):
                left, right = (i, j) if text[i, 0] <= text[j, 0] else (j, i)
                overlap_x = text[left, 0] + widths[left] - text[right, 0]
                overlap_y = height - abs(text[i, 1] - text[j, 1])
                if overlap_x > 0 and overlap_y > 0:
                    up, down = (i, j) if text[i, 1] >= text[j, 1] else (j, i)
                    text[up, 1] += overlap_y / 2 + 0.5
                    text[down, 1] -= overlap_y / 2 + 0.5
                    moved = True
        for i in range(len(text)):
            for k in range(len(points)):
                if k == i:
                    continue
                inside_x = text[i, 0] - 6 < points[k, 0] < text[i, 0] + widths[i] + 6
                gap = text[i, 1] - points[k, 1]
                need = height / 2 + 6
                if inside_x and abs(gap) < need:
                    text[i, 1] += (need - abs(gap) + 0.5) * (1 if gap >= 0 else -1)
                    moved = True
        if not moved:
            break
    inverse = to_px.inverted()
    for point, label_px, name in zip(points, text, names):
        shifted = abs(label_px[1] - point[1]) > 3
        axis.annotate(
            name,
            xy=inverse.transform(point),
            xytext=inverse.transform(label_px),
            fontsize=size,
            va="center",
            arrowprops={"arrowstyle": "-", "lw": 0.5, "color": "grey"}
            if shifted
            else None,
        )


def report_class_f1_vs_homophily(
    per_class: pd.DataFrame, models: list[str], title: str
) -> plt.Figure:
    """Per-class F1 of each model (and the MLP if present) against homophily."""
    with plt.rc_context(REPORT_STYLE):
        fig, axis = plt.subplots(figsize=REPORT_SIZE)
        h = per_class["endpoint_homophily"]
        for model in models:
            axis.scatter(
                h,
                per_class[model],
                s=30,
                color=MODEL_COLOURS.get(model),
                label=model_label(model),
            )
        if "mlp" in per_class:
            axis.scatter(
                h,
                per_class["mlp"],
                s=45,
                color="black",
                marker="x",
                label="MLP",
            )
        axis.set_xlim(-0.03, 0.95)
        axis.set_ylim(-0.03, 0.9)
        axis.set_xlabel("Class endpoint homophily")
        axis.set_ylabel("Validation F1 (mean over seeds)")
        axis.set_title(title)
        axis.legend(loc="lower right")
        fig.tight_layout()
        label_model = "gated_gcn" if "gated_gcn" in per_class else models[0]
        _place_labels(
            axis,
            h,
            per_class[label_model],
            [class_label(n) for n in per_class.index],
        )
    return fig


def report_gate_gain_by_class(
    per_class: dict[str, pd.DataFrame], title: str
) -> plt.Figure:
    """Gated GCN minus GCN F1 per class, one bar per budget, sorted by homophily."""
    with plt.rc_context(REPORT_STYLE):
        fig, axis = plt.subplots(figsize=(REPORT_SIZE[0], REPORT_SIZE[1] * 1.15))
        first = next(iter(per_class.values()))
        order = first.sort_values("endpoint_homophily").index
        y = np.arange(len(order))
        height = 0.8 / len(per_class)
        colours = ["tab:purple", "tab:cyan", "tab:olive", "tab:brown"]
        for index, (budget, table) in enumerate(per_class.items()):
            axis.barh(
                y - 0.4 + height * (index + 0.5),
                table.loc[order, "gate_gain"],
                height=height,
                color=colours[index % len(colours)],
                label=f"{budget_label(budget)} labels",
            )
        axis.axvline(0, color="black", lw=1)
        axis.set_yticks(
            y,
            [
                f"{class_label(n)} ({first.loc[n, 'endpoint_homophily']:.2f})"
                for n in order
            ],
        )
        axis.set_xlabel("F1 difference (Gated GCN - GCN)")
        axis.set_title(title)
        axis.legend(loc="lower left")
        fig.tight_layout()
    return fig


def report_mean_gate_by_budget(results: pd.DataFrame, budgets: list[str]) -> plt.Figure:
    """Mean validation gate of the Gated GCN by label budget (scarcity runs)."""
    rows = results[
        (results["experiment"] == "scarcity") & (results["model"] == "gated_gcn")
    ]
    stats = rows.groupby("budget")["mean_gate"].agg(["mean", "std"]).reindex(budgets)
    with plt.rc_context(REPORT_STYLE):
        fig, axis = plt.subplots(figsize=REPORT_SIZE)
        x = np.arange(len(budgets))
        axis.errorbar(
            x,
            stats["mean"],
            yerr=stats["std"],
            marker="o",
            capsize=4,
            color=MODEL_COLOURS["gated_gcn"],
        )
        axis.set_xticks(x, [budget_label(b) for b in budgets])
        axis.set_xlabel("Label budget")
        axis.set_ylabel("Mean gate (1 = keep own features)")
        axis.set_title("Gated GCN mean gate by label budget ($k$-NN)")
        fig.tight_layout()
    return fig


def report_gate_cells(run: RunOutput, homophily: np.ndarray, title: str) -> plt.Figure:
    """Gate against local homophily for the validation cells of one run."""
    keep = run.mask & np.isfinite(homophily)
    with plt.rc_context(REPORT_STYLE):
        fig, axis = plt.subplots(figsize=REPORT_SIZE)
        hexes = axis.hexbin(
            homophily[keep], run.gate[keep], gridsize=35, mincnt=1, cmap="viridis"
        )
        fig.colorbar(hexes, ax=axis, label="Cells")
        axis.set_xlabel("Local homophily of the cell")
        axis.set_ylabel("Mean gate (1 = keep own features)")
        axis.set_title(title)
        fig.tight_layout()
    return fig


def report_gate_class_means(
    class_means: pd.DataFrame, rho: float, p_value: float, title: str
) -> plt.Figure:
    """Class mean gate against class mean local homophily."""
    with plt.rc_context(REPORT_STYLE):
        fig, axis = plt.subplots(figsize=REPORT_SIZE)
        axis.scatter(
            class_means["homophily"],
            class_means["gate"],
            s=35,
            color=MODEL_COLOURS["gated_gcn"],
        )
        axis.set_xlim(-0.03, 0.95)
        axis.set_xlabel("Class mean local homophily")
        axis.set_ylabel("Class mean gate")
        axis.set_title(f"{title}\nSpearman $\\rho$ = {rho:.2f}, p = {p_value:.2f}")
        fig.tight_layout()
        _place_labels(
            axis,
            class_means["homophily"],
            class_means["gate"],
            [class_label(n) for n in class_means.index],
        )
    return fig


def report_confusion(table: pd.DataFrame, title: str) -> plt.Figure:
    """One row-normalised confusion matrix (rows: true class)."""
    names = [class_label(name) for name in table.index]
    with plt.rc_context(REPORT_STYLE):
        fig, axis = plt.subplots(figsize=(7, 6.4))
        sns.heatmap(
            table.to_numpy(),
            vmin=0,
            vmax=1,
            cmap="Blues",
            annot=True,
            fmt=".2f",
            annot_kws={"size": 8},
            cbar=False,
            xticklabels=names,
            yticklabels=names,
            ax=axis,
        )
        axis.tick_params(axis="x", rotation=60)
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        axis.set_title(title)
        fig.tight_layout()
    return fig


def report_scarcity(
    results: pd.DataFrame, mlp: pd.DataFrame, budgets: list[str]
) -> plt.Figure:
    """Macro-F1 by label budget for the scarcity models and the MLP."""
    scarcity = results[results["experiment"] == "scarcity"]
    curves = {"mlp": mlp.groupby("budget")["macro_f1"].agg(["mean", "std"])}
    for model in scarcity["model"].unique():
        rows = scarcity[scarcity["model"] == model]
        curves[model] = rows.groupby("budget")["macro_f1"].agg(["mean", "std"])
    with plt.rc_context(REPORT_STYLE):
        fig, axis = plt.subplots(figsize=REPORT_SIZE)
        x = np.arange(len(budgets))
        for model, curve in curves.items():
            curve = curve.reindex(budgets)
            axis.errorbar(
                x,
                curve["mean"],
                yerr=curve["std"],
                marker="o",
                capsize=3,
                color=MODEL_COLOURS.get(model),
                label=model_label(model),
            )
        axis.set_xticks(x, [budget_label(b) for b in budgets])
        axis.set_xlabel("Label budget")
        axis.set_ylabel("Validation macro-F1")
        axis.set_title("Label scarcity ($k$-NN, depth 2)")
        axis.legend()
        fig.tight_layout()
    return fig


def report_neighbourhood(results: pd.DataFrame) -> plt.Figure:
    """Macro-F1 against neighbourhood size k at each budget."""
    rows = results[results["experiment"] == "neighbourhood"].copy()
    rows["k"] = rows["graph"].str.removeprefix("knn").astype(int)
    styles = {"5pct": "--", "full": "-"}
    with plt.rc_context(REPORT_STYLE):
        fig, axis = plt.subplots(figsize=REPORT_SIZE)
        for model in rows["model"].unique():
            for budget in rows["budget"].unique():
                part = rows[(rows["model"] == model) & (rows["budget"] == budget)]
                curve = part.groupby("k")["macro_f1"].agg(["mean", "std"])
                axis.errorbar(
                    curve.index,
                    curve["mean"],
                    yerr=curve["std"],
                    marker="o",
                    capsize=3,
                    ls=styles.get(budget, "-"),
                    color=MODEL_COLOURS.get(model),
                    label=f"{model_label(model)}, {budget_label(budget)} labels",
                )
        axis.set_xticks(sorted(rows["k"].unique()))
        low, high = axis.get_ylim()
        axis.set_ylim(low, high + (high - low) * 0.35)
        axis.set_xlabel("Neighbours per cell $k$")
        axis.set_ylabel("Validation macro-F1")
        axis.set_title("Neighbourhood size ($k$-NN, depth 2)")
        axis.legend(loc="upper center", ncol=2, fontsize=9)
        fig.tight_layout()
    return fig


def report_depth(results: pd.DataFrame) -> plt.Figure:
    """Macro-F1 of every seed against depth, with the seed mean as a line."""
    rows = results[results["experiment"] == "depth"]
    models = list(rows["model"].unique())
    shifts = np.linspace(-0.08, 0.08, len(models))
    with plt.rc_context(REPORT_STYLE):
        fig, axis = plt.subplots(figsize=REPORT_SIZE)
        for model, shift in zip(models, shifts):
            part = rows[rows["model"] == model]
            axis.scatter(
                part["depth"] + shift,
                part["macro_f1"],
                s=30,
                color=MODEL_COLOURS.get(model),
                label=f"{model_label(model)} (one dot per seed)",
            )
            mean = part.groupby("depth")["macro_f1"].mean()
            axis.plot(
                mean.index + shift,
                mean.values,
                color=MODEL_COLOURS.get(model),
                alpha=0.5,
            )
        axis.set_xticks(sorted(rows["depth"].unique()))
        low, high = axis.get_ylim()
        axis.set_ylim(low, high + (high - low) * 0.3)
        axis.set_xlabel("Message-passing layers")
        axis.set_ylabel("Validation macro-F1")
        axis.set_title("Depth (5% labels, $k$-NN)")
        axis.legend(loc="upper left")
        fig.tight_layout()
    return fig


def report_dropout_gap(dropout: pd.DataFrame, title: str) -> plt.Figure:
    """GNN minus MLP accuracy per dropout bin for one setting."""
    gap = dropout["gnn_minus_mlp_mean"].unstack("model")
    gap = gap[sorted(gap.columns, key=list(MODEL_LABELS).index)]
    with plt.rc_context(REPORT_STYLE):
        fig, axis = plt.subplots(figsize=REPORT_SIZE)
        for model in gap.columns:
            axis.plot(
                gap.index,
                gap[model],
                marker="o",
                color=MODEL_COLOURS.get(model),
                label=model_label(model),
            )
        axis.axhline(0, color="black", lw=1, label="MLP")
        axis.set_xticks(sorted(gap.index))
        low, high = axis.get_ylim()
        axis.set_ylim(low - (high - low) * 0.45, high)
        axis.set_xlabel("Genes-detected bin (1 = most dropout)")
        axis.set_ylabel("Accuracy difference to MLP")
        axis.set_title(title)
        axis.legend(loc="lower center", ncol=3)
        fig.tight_layout()
    return fig
