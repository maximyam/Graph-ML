"""First-principles sparse GNN layers used by the Step 3 experiments."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


def _validate_graph(edge_index: Tensor, edge_weight: Optional[Tensor], n_nodes: int) -> Tensor:
	if edge_index.ndim != 2 or edge_index.shape[0] != 2:
		raise ValueError("edge_index must have shape (2, n_edges)")
	if edge_index.dtype not in (torch.int32, torch.int64):
		raise TypeError("edge_index must contain integer indices")
	if edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= n_nodes):
		raise ValueError("edge_index contains an out-of-range node")
	if edge_weight is None:
		return torch.ones(edge_index.shape[1], device=edge_index.device)
	if edge_weight.ndim != 1 or edge_weight.shape[0] != edge_index.shape[1]:
		raise ValueError("edge_weight must have one value per edge")
	return edge_weight.to(dtype=torch.float32)


def _add_self_loops(
	edge_index: Tensor, edge_weight: Tensor, n_nodes: int
) -> tuple[Tensor, Tensor]:
	loops = torch.arange(n_nodes, device=edge_index.device, dtype=edge_index.dtype)
	loop_index = torch.stack((loops, loops))
	loop_weight = torch.ones(n_nodes, device=edge_weight.device, dtype=edge_weight.dtype)
	return (
		torch.cat((edge_index, loop_index), dim=1),
		torch.cat((edge_weight, loop_weight)),
	)


def _aggregate(
	x: Tensor,
	edge_index: Tensor,
	edge_weight: Optional[Tensor] = None,
	*,
	normalize: bool = False,
	add_self_loops: bool = False,
) -> Tensor:
	"""Aggregate source node features into destination nodes with index_add."""
	n_nodes = x.shape[0]
	weights = _validate_graph(edge_index, edge_weight, n_nodes).to(x.device)
	if add_self_loops:
		edge_index, weights = _add_self_loops(edge_index, weights, n_nodes)

	source, destination = edge_index
	messages = x[source] * weights.unsqueeze(-1).to(x.dtype)
	result = torch.zeros_like(x)
	result.index_add_(0, destination, messages)
	if normalize:
		degree = torch.zeros(n_nodes, device=x.device, dtype=x.dtype)
		degree.index_add_(0, destination, weights.to(x.dtype))
		result = result / degree.clamp_min(torch.finfo(x.dtype).eps).unsqueeze(-1)
	return result


def _scatter_softmax(values: Tensor, groups: Tensor, n_groups: int) -> Tensor:
	"""Compute a numerically stable softmax independently for each destination."""
	maximum = torch.full(
		(n_groups,), -torch.inf, device=values.device, dtype=values.dtype
	)
	maximum.scatter_reduce_(0, groups, values, reduce="amax", include_self=True)
	exponent = torch.exp(values - maximum[groups])
	denominator = torch.zeros(n_groups, device=values.device, dtype=values.dtype)
	denominator.index_add_(0, groups, exponent)
	return exponent / denominator[groups].clamp_min(torch.finfo(values.dtype).eps)


class GCNLayer(nn.Module):
	"""A weighted GCN layer with symmetric normalized adjacency."""

	def __init__(self, in_features: int, out_features: int) -> None:
		super().__init__()
		self.linear = nn.Linear(in_features, out_features, bias=False)

	def forward(
		self, x: Tensor, edge_index: Tensor, edge_weight: Optional[Tensor] = None
	) -> Tensor:
		edge_weight = _validate_graph(edge_index, edge_weight, x.shape[0]).to(x.device)
		edge_index, edge_weight = _add_self_loops(edge_index, edge_weight, x.shape[0])
		source, destination = edge_index
		degree = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
		degree.index_add_(0, destination, edge_weight.to(x.dtype))
		normalized = edge_weight.to(x.dtype) / (
			degree[source].clamp_min(torch.finfo(x.dtype).eps).sqrt()
			* degree[destination].clamp_min(torch.finfo(x.dtype).eps).sqrt()
		)
		messages = self.linear(x[source]) * normalized.unsqueeze(-1)
		output = torch.zeros(x.shape[0], self.linear.out_features, device=x.device, dtype=x.dtype)
		output.index_add_(0, destination, messages)
		return output


class GraphSAGELayer(nn.Module):
	"""Mean-aggregation GraphSAGE layer."""

	def __init__(self, in_features: int, out_features: int) -> None:
		super().__init__()
		self.linear = nn.Linear(in_features * 2, out_features)

	def forward(
		self, x: Tensor, edge_index: Tensor, edge_weight: Optional[Tensor] = None
	) -> Tensor:
		neighbor_mean = _aggregate(x, edge_index, edge_weight, normalize=True)
		return self.linear(torch.cat((x, neighbor_mean), dim=-1))


class GATLayer(nn.Module):
	"""Multi-head graph attention using scatter-based destination softmax."""

	def __init__(
		self, in_features: int, out_features: int, heads: int = 2, concat: bool = True
	) -> None:
		super().__init__()
		self.heads = heads
		self.out_features = out_features
		self.concat = concat
		self.linear = nn.Linear(in_features, heads * out_features, bias=False)
		self.attention_source = nn.Parameter(torch.empty(heads, out_features))
		self.attention_destination = nn.Parameter(torch.empty(heads, out_features))
		nn.init.xavier_uniform_(self.attention_source)
		nn.init.xavier_uniform_(self.attention_destination)

	def forward(
		self, x: Tensor, edge_index: Tensor, edge_weight: Optional[Tensor] = None
	) -> Tensor:
		n_nodes = x.shape[0]
		weights = _validate_graph(edge_index, edge_weight, n_nodes).to(x.device)
		edge_index, weights = _add_self_loops(edge_index, weights, n_nodes)
		source, destination = edge_index
		transformed = self.linear(x).view(n_nodes, self.heads, self.out_features)
		scores = (
			(transformed[source] * self.attention_source).sum(-1)
			+ (transformed[destination] * self.attention_destination).sum(-1)
		)
		scores = torch.where(scores >= 0, scores, 0.2 * scores)
		attention = torch.stack(
			[_scatter_softmax(scores[:, head], destination, n_nodes) for head in range(self.heads)],
			dim=1,
		)
		attention = attention * weights.to(attention.dtype).unsqueeze(-1)
		messages = transformed[source] * attention.unsqueeze(-1)
		output = torch.zeros(
			n_nodes, self.heads, self.out_features, device=x.device, dtype=x.dtype
		)
		output.index_add_(0, destination, messages)
		return output.reshape(n_nodes, -1) if self.concat else output.mean(dim=1)


class HomophilyGatedLayer(nn.Module):
	"""Learn a node-wise balance between self features and neighbor features."""

	def __init__(self, in_features: int, out_features: int) -> None:
		super().__init__()
		self.gate = nn.Linear(in_features * 2, in_features)
		self.linear = nn.Linear(in_features, out_features)
		self.last_gate: Optional[Tensor] = None

	def forward(
		self, x: Tensor, edge_index: Tensor, edge_weight: Optional[Tensor] = None
	) -> Tensor:
		neighbor_mean = _aggregate(x, edge_index, edge_weight, normalize=True)
		gate = torch.sigmoid(self.gate(torch.cat((x, neighbor_mean), dim=-1)))
		self.last_gate = gate.detach()
		mixed = gate * x + (1.0 - gate) * neighbor_mean
		return self.linear(mixed)


class GNNClassifier(nn.Module):
	"""Configurable node classifier for the four Step 3 architectures."""

	def __init__(
		self,
		model_name: str,
		in_features: int,
		hidden_features: int,
		n_classes: int,
		depth: int = 2,
		dropout: float = 0.2,
		attention_heads: int = 2,
	) -> None:
		super().__init__()
		if depth < 1:
			raise ValueError("depth must be at least 1")
		if model_name not in {"gcn", "graphsage", "gat", "gated_gcn"}:
			raise ValueError(f"Unknown model: {model_name}")
		self.model_name = model_name
		self.dropout = dropout
		layers: list[nn.Module] = []
		input_size = in_features
		for layer_number in range(depth):
			output_size = n_classes if layer_number == depth - 1 else hidden_features
			if model_name == "gcn":
				layer = GCNLayer(input_size, output_size)
			elif model_name == "graphsage":
				layer = GraphSAGELayer(input_size, output_size)
			elif model_name == "gat":
				layer = GATLayer(
					input_size,
					output_size,
					heads=attention_heads if layer_number < depth - 1 else 1,
					concat=layer_number < depth - 1,
				)
				output_size *= attention_heads if layer_number < depth - 1 else 1
			else:
				layer = HomophilyGatedLayer(input_size, output_size)
			layers.append(layer)
			input_size = output_size
		self.layers = nn.ModuleList(layers)

	def forward(
		self, x: Tensor, edge_index: Tensor, edge_weight: Optional[Tensor] = None
	) -> Tensor:
		for layer_number, layer in enumerate(self.layers):
			x = layer(x, edge_index, edge_weight)
			if layer_number < len(self.layers) - 1:
				x = torch.relu(x)
				x = nn.functional.dropout(x, p=self.dropout, training=self.training)
		return x

	def gate_values(self) -> Optional[Tensor]:
		gates = [layer.last_gate for layer in self.layers if isinstance(layer, HomophilyGatedLayer)]
		return gates[-1] if gates else None
