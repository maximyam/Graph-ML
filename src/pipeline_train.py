"""Training and experiment utilities for Step 3 GNN experiments.

The pipeline keeps the graph transductive: every node participates in message
passing, while only ``labelled_train`` contributes to the supervised loss.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch import Tensor
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

try:
	from .models import GNNClassifier
except ImportError:
	from models import GNNClassifier


@dataclass
class ExperimentData:
	features: np.ndarray
	edge_index: np.ndarray
	edge_weight: np.ndarray
	cell_ids: np.ndarray
	cell_types: np.ndarray
	local_homophily: np.ndarray
	train_mask: np.ndarray
	validation_mask: np.ndarray
	test_mask: np.ndarray
	labels: np.ndarray
	validation_labels: np.ndarray
	class_names: list[str]


def set_seed(seed: int) -> None:
	"""Seed every random source used by the pipeline."""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
	torch.use_deterministic_algorithms(True, warn_only=True)
	if torch.backends.cudnn.is_available():
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False


def _read_class_names(path: Path, labels: pd.Series) -> list[str]:
	if path.is_file():
		return [str(value) for value in json.loads(path.read_text())]
	return sorted(labels.dropna().astype(str).unique().tolist())


def _sample_label_mask(
	train_pool: np.ndarray,
	labels: np.ndarray,
	fraction: Optional[float],
	seed: int,
) -> np.ndarray:
	if fraction is None or fraction >= 1.0:
		return train_pool.copy()
	if not 0 < fraction <= 1:
		raise ValueError("label fraction must be in (0, 1]")
	generator = np.random.default_rng(seed)
	selected = np.zeros_like(train_pool, dtype=bool)
	for class_id in np.unique(labels[train_pool]):
		candidates = np.flatnonzero(train_pool & (labels == class_id))
		count = min(len(candidates), max(1, int(np.ceil(fraction * len(candidates)))))
		selected[generator.choice(candidates, size=count, replace=False)] = True
	return selected


def _budget_fraction(budget: str) -> Optional[float]:
	if budget == "full":
		return None
	if budget.endswith("pct"):
		return float(budget[:-3]) / 100.0
	raise ValueError(f"Budget must be 'full' or '<percent>pct', got {budget!r}")


def _adjacency_to_edges(adjacency: sp.spmatrix) -> tuple[np.ndarray, np.ndarray]:
	coordinate_graph = adjacency.tocoo()
	edge_index = np.stack((coordinate_graph.row, coordinate_graph.col)).astype(np.int64)
	edge_weight = coordinate_graph.data.astype(np.float32)
	return edge_index, edge_weight


def load_data(
	export_root: str | Path,
	budget: str = "5pct",
	graph_name: str = "knn",
	seed: int = 42,
) -> ExperimentData:
	"""Load an existing Step 2 export or construct a missing scarcity budget."""
	export_root = Path(export_root)
	metadata = pd.read_csv(export_root / "cell_ids.csv", index_col=0)
	cell_ids = metadata.index.to_numpy(dtype=str)
	if "celltype" not in metadata.columns:
		metadata = pd.read_csv(export_root / "../processed/cell_metadata.csv", index_col=0)
	labels_text = metadata["celltype"].astype(str).to_numpy()
	local_homophily = metadata.get(
		"local_homophily", pd.Series(np.nan, index=metadata.index)
	).to_numpy(dtype=np.float32)
	class_names = _read_class_names(export_root / "class_names.json", metadata["celltype"])
	class_to_id = {name: index for index, name in enumerate(class_names)}
	labels = np.array([class_to_id[name] for name in labels_text], dtype=np.int64)

	masks_path = export_root / "labels_full" / "masks.npz"
	if not masks_path.is_file():
		raise FileNotFoundError(f"Missing Step 2 export: {masks_path}")
	shared_masks = np.load(masks_path)
	train_pool = shared_masks["train_pool"].astype(bool)
	validation_mask = shared_masks["validation"].astype(bool)
	test_mask = shared_masks["test"].astype(bool)

	budget_folder = export_root / f"labels_{budget}"
	exported_masks = budget_folder / "masks.npz"
	if exported_masks.is_file():
		budget_masks = np.load(exported_masks)
		labelled_train = budget_masks["labelled_train"].astype(bool)
	else:
		labelled_train = _sample_label_mask(
			train_pool, labels, _budget_fraction(budget), seed
		)

	raw_features_path = export_root / "X_pca.npy"
	if raw_features_path.is_file():
		features = np.load(raw_features_path, allow_pickle=False).astype(np.float32)
		scaler = StandardScaler().fit(features[labelled_train])
		features = scaler.transform(features).astype(np.float32)
	else:
		scaled_path = budget_folder / "X_scaled.npy"
		if not scaled_path.is_file():
			raise FileNotFoundError(f"Missing features: {raw_features_path} or {scaled_path}")
		features = np.load(scaled_path, allow_pickle=False).astype(np.float32)

	adjacency_path = export_root / "graphs" / f"{graph_name}_adjacency.npz"
	if not adjacency_path.is_file():
		raise FileNotFoundError(f"Missing graph export: {adjacency_path}")
	edge_index, edge_weight = _adjacency_to_edges(sp.load_npz(adjacency_path))

	training_labels = np.full(len(labels), -1, dtype=np.int64)
	training_labels[labelled_train] = labels[labelled_train]
	validation_labels = np.full(len(labels), -1, dtype=np.int64)
	validation_labels[validation_mask] = labels[validation_mask]
	if not np.all(labelled_train <= train_pool):
		raise ValueError("labelled_train must be a subset of train_pool")
	if np.any(train_pool & validation_mask) or np.any(train_pool & test_mask):
		raise ValueError("split masks overlap")
	return ExperimentData(
		features=features,
		edge_index=edge_index,
		edge_weight=edge_weight,
		cell_ids=cell_ids,
		cell_types=labels_text,
		local_homophily=local_homophily,
		train_mask=labelled_train,
		validation_mask=validation_mask,
		test_mask=test_mask,
		labels=training_labels,
		validation_labels=validation_labels,
		class_names=class_names,
	)


def _metrics(predictions: np.ndarray, targets: np.ndarray, class_names: list[str]) -> dict[str, Any]:
	per_class_scores = np.asarray(
		f1_score(
			targets,
			predictions,
			labels=np.arange(len(class_names)),
			average=None,
			zero_division=0,
		),
		dtype=float,
	)
	return {
		"accuracy": float(accuracy_score(targets, predictions)),
		"macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
		"per_class_f1": dict(zip(class_names, per_class_scores.tolist())),
	}


def train_one(
	data: ExperimentData,
	model_name: str,
	hidden_features: int = 64,
	depth: int = 2,
	seed: int = 42,
	epochs: int = 200,
	patience: int = 25,
	learning_rate: float = 1e-3,
	weight_decay: float = 1e-4,
	device: str = "cpu",
) -> tuple[dict[str, Any], dict[str, Tensor]]:
	"""Train one model and return validation-selected results and its state."""
	set_seed(seed)
	torch_device = torch.device(device)
	features = torch.from_numpy(data.features).to(torch_device)
	edge_index = torch.from_numpy(data.edge_index).to(torch_device)
	edge_weight = torch.from_numpy(data.edge_weight).to(torch_device)
	labels = torch.from_numpy(data.labels).to(torch_device)
	train_mask = torch.from_numpy(data.train_mask).to(torch_device)
	validation_mask = torch.from_numpy(data.validation_mask).to(torch_device)
	class_count = len(data.class_names)
	train_labels = labels[train_mask]
	counts = torch.bincount(train_labels, minlength=class_count).float()
	class_weights = counts.sum() / (class_count * counts.clamp_min(1.0))
	model = GNNClassifier(
		model_name,
		features.shape[1],
		hidden_features,
		class_count,
		depth=depth,
	).to(torch_device)
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
	best_score = -np.inf
	best_epoch = 0
	best_state: Optional[dict[str, Tensor]] = None
	history: list[dict[str, float]] = []
	stale_epochs = 0
	validation_targets = data.validation_labels[data.validation_mask]
	for epoch in range(1, epochs + 1):
		model.train()
		optimizer.zero_grad(set_to_none=True)
		logits = model(features, edge_index, edge_weight)
		loss = torch.nn.functional.cross_entropy(logits[train_mask], labels[train_mask], weight=class_weights)
		loss.backward()
		optimizer.step()

		model.eval()
		with torch.no_grad():
			validation_logits = model(features, edge_index, edge_weight)
			validation_predictions = validation_logits[validation_mask].argmax(dim=1).cpu().numpy()
		validation_score = float(
			f1_score(validation_targets, validation_predictions, average="macro", zero_division=0)
		)
		history.append({"epoch": float(epoch), "loss": float(loss.item()), "val_macro_f1": validation_score})
		if validation_score > best_score:
			best_score = validation_score
			best_epoch = epoch
			best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
			stale_epochs = 0
		else:
			stale_epochs += 1
			if stale_epochs >= patience:
				break
	if best_state is None:
		raise RuntimeError("Training produced no checkpoint")
	model.load_state_dict(best_state)
	model.eval()
	with torch.no_grad():
		logits = model(features, edge_index, edge_weight)
	validation_predictions = logits[validation_mask].argmax(dim=1).cpu().numpy()
	result = _metrics(validation_predictions, validation_targets, data.class_names)
	result.update(
		{
			"model": model_name,
			"seed": seed,
			"depth": depth,
			"hidden_features": hidden_features,
			"best_epoch": best_epoch,
			"epochs_run": len(history),
			"history": history,
		}
	)
	gate = model.gate_values()
	if gate is not None:
		result["mean_gate"] = float(gate.mean().item())
		result["validation_mean_gate"] = float(gate[validation_mask].mean().item())
	return result, best_state


def export_gate_values(
	data: ExperimentData,
	state: dict[str, Tensor],
	model_name: str,
	hidden_features: int = 64,
	depth: int = 2,
	output_path: str | Path | None = None,
) -> np.ndarray | None:
	"""Run a saved checkpoint and optionally persist its node-wise gate values."""
	if model_name != "gated_gcn":
		return None
	model = GNNClassifier(
		model_name,
		data.features.shape[1],
		hidden_features,
		len(data.class_names),
		depth=depth,
	)
	model.load_state_dict(state)
	model.eval()
	features = torch.from_numpy(data.features)
	edge_index = torch.from_numpy(data.edge_index)
	edge_weight = torch.from_numpy(data.edge_weight)
	with torch.no_grad():
		model(features, edge_index, edge_weight)
	gate = model.gate_values()
	if gate is None:
		return None
	gate_values = gate.cpu().numpy().astype(np.float32, copy=False)
	if output_path is not None:
		output_path = Path(output_path)
		output_path.parent.mkdir(parents=True, exist_ok=True)
		np.save(output_path, gate_values)
		metadata = pd.DataFrame(
			{
				"cell_id": data.cell_ids,
				"celltype": data.cell_types,
				"local_homophily": data.local_homophily,
				"train_mask": data.train_mask,
				"validation_mask": data.validation_mask,
				"test_mask": data.test_mask,
			}
		)
		metadata.to_csv(output_path.with_name(output_path.stem + "_metadata.csv"), index=False)
	return gate_values


def run_experiments(
	export_root: str | Path,
	output_root: str | Path,
	budgets: Iterable[str] = ("5pct", "full"),
	graph_names: Iterable[str] = ("knn", "snn"),
	model_names: Iterable[str] = ("gcn", "graphsage", "gat", "gated_gcn"),
	depths: Iterable[int] = (2,),
	seeds: Iterable[int] = (42,),
	epochs: int = 200,
	device: str = "cpu",
) -> pd.DataFrame:
	output_root = Path(output_root)
	output_root.mkdir(parents=True, exist_ok=True)
	rows: list[dict[str, Any]] = []
	for budget in budgets:
		for graph_name in graph_names:
			data = load_data(export_root, budget, graph_name, seed=42)
			for model_name in model_names:
				for depth in depths:
					for seed in seeds:
						result, state = train_one(
							data,
							model_name,
							depth=depth,
							seed=seed,
							epochs=epochs,
							device=device,
						)
						result.update({"budget": budget, "graph": graph_name})
						checkpoint_name = f"{model_name}_{budget}_{graph_name}_depth{depth}_seed{seed}.pt"
						torch.save(state, output_root / checkpoint_name)
						if model_name == "gated_gcn":
							export_gate_values(
								data,
								state,
								model_name,
								depth=depth,
								output_path=output_root / checkpoint_name.replace(
									".pt", "_gate_values.npy"
								),
							)
						rows.append({key: value for key, value in result.items() if key != "history"})
						(output_root / checkpoint_name.replace(".pt", ".json")).write_text(
							json.dumps(result, indent=2)
						)
	results = pd.DataFrame(rows)
	results.to_csv(output_root / "validation_results.csv", index=False)
	return results


def main() -> None:
	parser = argparse.ArgumentParser(description="Run Step 3 sparse GNN experiments")
	parser.add_argument("--export-root", default="data/graph_engineering_50pc")
	parser.add_argument("--output-root", default="data/gnn_training_50pc")
	parser.add_argument("--budgets", nargs="+", default=["5pct", "full"])
	parser.add_argument("--graphs", nargs="+", default=["knn", "snn"])
	parser.add_argument("--models", nargs="+", default=["gcn", "graphsage", "gat", "gated_gcn"])
	parser.add_argument("--depths", nargs="+", type=int, default=[2])
	parser.add_argument("--seeds", nargs="+", type=int, default=[42])
	parser.add_argument("--epochs", type=int, default=200)
	parser.add_argument("--device", default="cpu")
	arguments = parser.parse_args()
	results = run_experiments(
		arguments.export_root,
		arguments.output_root,
		arguments.budgets,
		arguments.graphs,
		arguments.models,
		arguments.depths,
		arguments.seeds,
		arguments.epochs,
		arguments.device,
	)
	print(results.to_string(index=False))


if __name__ == "__main__":
	main()
