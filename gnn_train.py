"""
GNN Training Module for Node Value Prediction

Trains a four-layer Graph Neural Network (GCN or GAT) to predict missing node values
from partially observed synthetic graphs. Saves trained models and layer activations
for downstream interpretability analysis.
"""

import argparse
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data, Batch
from tqdm import tqdm


MOTIF_LABELS = [
    "cascade",
    "feedback_loop",
    "feedforward_loop",
    "single_input_module",
    "mixed_motif",
    "unknown"
]
MOTIF_TO_ID = {label: idx for idx, label in enumerate(MOTIF_LABELS)}


def _extract_graph_id(graph_path: Path) -> Optional[int]:
    """
    Extract original graph ID from filename.

    Filenames follow pattern: graph_<ID>.pkl
    Graph IDs encode motif type:
    - 0-999:       feedforward_loop
    - 1000-1999:   feedback_loop
    - 2000-2999:   single_input_module
    - 3000-3999:   cascade
    - 4000-4999:   mixed_motif

    Args:
        graph_path: Path to graph file

    Returns:
        Original graph ID, or None if cannot be extracted
    """
    try:
        graph_id = int(graph_path.stem.split('_')[1])
        return graph_id
    except (IndexError, ValueError):
        return None


def _infer_motif_label(graph_path: Path) -> str:
    """
    Infer motif label from a path using directory structure or graph ID.

    Supports both old structure (single_motif_graphs/motif_type/) and
    new structure (all_graphs/raw_graphs/ with graph IDs).

    For new structure, graph IDs map to motifs as follows:
    - IDs 0-999:       feedforward_loop
    - IDs 1000-1999:   feedback_loop
    - IDs 2000-2999:   single_input_module
    - IDs 3000-3999:   cascade
    - IDs 4000-4999:   mixed_motif
    """
    parts = graph_path.parts

    # Try old directory structure first
    if "single_motif_graphs" in parts:
        idx = parts.index("single_motif_graphs")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    if "mixed_motif_graphs" in parts:
        return "mixed_motif"

    # For new structure, infer from graph ID
    try:
        graph_id = int(graph_path.stem.split('_')[1])

        # Mixed-motif graphs (IDs 4000-4999)
        if graph_id >= 4000:
            return "mixed_motif"

        # Single-motif graphs (IDs 0-3999) - use graph ID ranges
        # Order matches GraphMotifGenerator.motif_types initialization
        if graph_id < 1000:
            return "feedforward_loop"
        elif graph_id < 2000:
            return "feedback_loop"
        elif graph_id < 3000:
            return "single_input_module"
        else:  # graph_id < 4000
            return "cascade"

    except (IndexError, ValueError):
        return "unknown"


class GraphDataset(Dataset):
    """
    Dataset for loading synthetic graphs with expression data.

    Attributes:
        graph_paths: List of paths to graph files
        mask_prob: Probability of masking each node during training
        base_seed: Base random seed (each graph gets base_seed + graph_id)
    """

    def __init__(self, graph_paths: List[Path], mask_prob: float = 0.2, seed: int = 42):
        """
        Initialize the graph dataset.

        Args:
            graph_paths: List of paths to pickled graph files
            mask_prob: Probability of masking nodes for prediction
            seed: Random seed for reproducibility
        """
        self.graph_paths = graph_paths
        self.mask_prob = mask_prob
        self.base_seed = seed

    def __len__(self) -> int:
        return len(self.graph_paths)

    def __getitem__(self, idx: int) -> Data:
        """
        Load a graph and simulate expression values.

        Args:
            idx: Index of graph to load

        Returns:
            PyG Data object with node features, edge indices, and labels
        """
        # Load graph
        with open(self.graph_paths[idx], 'rb') as f:
            G = pickle.load(f)

        motif_label = _infer_motif_label(self.graph_paths[idx])
        motif_id = MOTIF_TO_ID.get(motif_label, MOTIF_TO_ID["unknown"])

        # Extract graph ID from filename (e.g., "graph_123.pkl" -> 123)
        graph_id = _extract_graph_id(self.graph_paths[idx])
        if graph_id is None:
            # Fallback to index if extraction fails
            graph_id = idx

        # Create local RNG for this graph for deterministic, reproducible randomness
        local_rng = np.random.default_rng(self.base_seed + graph_id)

        # Get adjacency matrix and simulate expression
        import networkx as nx
        n_nodes = len(G.nodes())
        W = nx.to_numpy_array(G, weight='weight')

        # Simulate expression dynamics
        expression = self._simulate_expression(W, rng=local_rng)

        # Create mask for training (which nodes to predict)
        mask = local_rng.random(n_nodes) < self.mask_prob
        mask = torch.tensor(mask, dtype=torch.bool)

        # Create node features: [normalized_expression, mask_flag]
        # For masked nodes, set expression to 0
        expression_tensor = torch.tensor(expression, dtype=torch.float32)
        masked_expression = expression_tensor.clone()
        masked_expression[mask] = 0.0

        # Normalize to [0, 1] range
        if masked_expression.max() > 0:
            masked_expression = masked_expression / (masked_expression.max() + 1e-8)

        # Stack features
        mask_flag = (~mask).float().unsqueeze(1)  # 1 if observed, 0 if masked
        x = torch.cat([masked_expression.unsqueeze(1), mask_flag], dim=1)

        # Create edge index from adjacency matrix
        edge_index = torch.tensor(np.array(np.nonzero(W)), dtype=torch.long)
        edge_weight = torch.tensor(W[W != 0], dtype=torch.float32)

        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_weight,
            y=expression_tensor,
            mask=mask,
            num_nodes=n_nodes
        )
        data.motif_id = torch.tensor([motif_id], dtype=torch.long)
        data.graph_id = torch.tensor([graph_id], dtype=torch.long)

        return data

    def _simulate_expression(self, W: np.ndarray, rng: np.random.Generator,
                            steps: int = 50, gamma: float = 0.3, noise_std: float = 0.01) -> np.ndarray:
        """
        Simulate gene expression dynamics.

        Args:
            W: Weighted adjacency matrix
            rng: Random number generator for this graph
            steps: Number of simulation steps
            gamma: Update rate parameter
            noise_std: Standard deviation of Gaussian noise

        Returns:
            Final expression values
        """
        n_nodes = W.shape[0]
        x = rng.uniform(0, 1, size=n_nodes)

        for _ in range(steps):
            weighted_input = W @ x
            sigmoid_input = 1.0 / (1.0 + np.exp(-np.clip(weighted_input, -10, 10)))
            noise = rng.normal(0, noise_std, size=n_nodes)
            x = (1 - gamma) * x + gamma * sigmoid_input + noise
            x = np.clip(x, 0, 1)

        return x


class GCNModel(nn.Module):
    """
    Four-layer Graph Convolutional Network for node value prediction.

    Architecture:
        - Layer 1: GCNConv(2 -> 128) + ReLU + Dropout (1-hop neighborhoods)
        - Layer 2: GCNConv(128 -> 128) + ReLU + Dropout (2-hop neighborhoods)
        - Layer 3: GCNConv(128 -> 64) + ReLU + Dropout (3-hop neighborhoods)
        - Layer 4: GCNConv(64 -> 1) (4-hop neighborhoods, final prediction)
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 128,
                 output_dim: int = 1, dropout: float = 0.2):
        """
        Initialize the GCN model.

        Args:
            input_dim: Input feature dimension
            hidden_dim: First and second hidden layer dimension
            output_dim: Output dimension
            dropout: Dropout probability
        """
        super(GCNModel, self).__init__()

        # Disable internal normalization to support signed edge weights
        self.conv1 = GCNConv(input_dim, hidden_dim, normalize=False)
        self.conv2 = GCNConv(hidden_dim, hidden_dim, normalize=False)
        self.conv3 = GCNConv(hidden_dim, 64, normalize=False)
        self.conv4 = GCNConv(64, output_dim, normalize=False)
        self.dropout = dropout

        # Storage for activations
        self.layer1_activations = None
        self.layer2_activations = None
        self.layer3_activations = None
        self.layer4_activations = None

    def forward(self, data: Data, store_activations: bool = False) -> torch.Tensor:
        """
        Forward pass through the 4-layer GCN.

        Args:
            data: PyG Data object
            store_activations: Whether to store layer activations

        Returns:
            Predicted node values
        """
        x, edge_index = data.x, data.edge_index
        edge_weight = getattr(data, 'edge_attr', None)

        # Layer 1: GCNConv + ReLU + Dropout (1-hop neighborhoods)
        h1 = self.conv1(x, edge_index, edge_weight=edge_weight)
        h1 = F.relu(h1)

        if store_activations:
            self.layer1_activations = h1.detach()

        h1 = F.dropout(h1, p=self.dropout, training=self.training)

        # Layer 2: GCNConv + ReLU + Dropout (2-hop neighborhoods)
        h2 = self.conv2(h1, edge_index, edge_weight=edge_weight)
        h2 = F.relu(h2)

        if store_activations:
            self.layer2_activations = h2.detach()

        h2 = F.dropout(h2, p=self.dropout, training=self.training)

        # Layer 3: GCNConv + ReLU + Dropout (3-hop neighborhoods)
        h3 = self.conv3(h2, edge_index, edge_weight=edge_weight)
        h3 = F.relu(h3)

        if store_activations:
            self.layer3_activations = h3.detach()

        h3 = F.dropout(h3, p=self.dropout, training=self.training)

        # Layer 4: GCNConv (4-hop neighborhoods, final prediction)
        h4 = self.conv4(h3, edge_index, edge_weight=edge_weight)

        if store_activations:
            self.layer4_activations = h4.detach()

        return h4.squeeze(-1)

    def get_activations(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get stored layer activations.

        Returns:
            Tuple of (layer1_activations, layer2_activations, layer3_activations, layer4_activations)
        """
        return self.layer1_activations, self.layer2_activations, self.layer3_activations, self.layer4_activations


class GATModel(nn.Module):
    """
    Four-layer Graph Attention Network for node value prediction.

    Architecture:
        - Layer 1: Multi-head GATConv (2 -> 128) + ELU + Dropout (captures 1-hop)
        - Layer 2: Multi-head GATConv (128 -> 128) + ELU + Dropout (captures 2-hop)
        - Layer 3: Multi-head GATConv (128 -> 64) + ELU + Dropout (captures 3-hop)
        - Layer 4: Single-head GATConv (64 -> 1) (captures 4-hop, final prediction)
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 32,
                 output_dim: int = 1, dropout: float = 0.2,
                 num_heads: int = 4, edge_dim: int = 1):
        super(GATModel, self).__init__()

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim

        # Layer 1: Multi-head attention (2 -> 32*4 = 128)
        # to represent first hop and add hyperparameter tuning freedom
        self.conv1 = GATConv(
            input_dim,
            hidden_dim,
            heads=num_heads,
            dropout=dropout,
            edge_dim=edge_dim
        )

        # Layer 2: Multi-head attention (128 -> 128)
        self.conv2 = GATConv(
            hidden_dim * num_heads,
            hidden_dim,
            heads=num_heads,
            dropout=dropout,
            edge_dim=edge_dim
        )

        # Layer 3: Multi-head attention (128 -> 64) --> To get 64 activations (same as GCN) for standardized and comparable SAE analysis
        # Og GAT paper uses 8 hidden_dim and 8 heads (giving 64) so we are fixing third layer based on that
        self.conv3 = GATConv(
            hidden_dim * num_heads,
            8,
            8,
            dropout=dropout,
            edge_dim=edge_dim
        )

        # Layer 4: Single-head attention for output (64 -> 1)
        self.conv4 = GATConv(
            64,
            output_dim,
            heads=1,
            concat=False,
            dropout=dropout,
            edge_dim=edge_dim
        )

        self.dropout = dropout

        # Storage for activations
        self.layer1_activations = None
        self.layer2_activations = None
        self.layer3_activations = None
        self.layer4_activations = None

    def forward(self, data: Data, store_activations: bool = False) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, 'edge_attr', None)

        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        # Layer 1
        h1 = self.conv1(
            x,
            edge_index,
            edge_attr=edge_attr
        )
        h1 = F.elu(h1)

        if store_activations:
            self.layer1_activations = h1.detach()

        h1 = F.dropout(h1, p=self.dropout, training=self.training)

        # Layer 2
        h2 = self.conv2(
            h1,
            edge_index,
            edge_attr=edge_attr
        )
        h2 = F.elu(h2)

        if store_activations:
            self.layer2_activations = h2.detach()

        h2 = F.dropout(h2, p=self.dropout, training=self.training)

        # Layer 3
        h3 = self.conv3(
            h2,
            edge_index,
            edge_attr=edge_attr
        )
        h3 = F.elu(h3)

        if store_activations:
            self.layer3_activations = h3.detach()

        h3 = F.dropout(h3, p=self.dropout, training=self.training)

        # Layer 4
        h4 = self.conv4(
            h3,
            edge_index,
            edge_attr=edge_attr
        )

        if store_activations:
            self.layer4_activations = h4.detach()

        return h4.squeeze(-1)

    def get_activations(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.layer1_activations, self.layer2_activations, self.layer3_activations, self.layer4_activations


class GNNTrainer:
    """
    Trainer class for GCN model on synthetic graphs.

    Handles training loop, validation, and saving of models and activations.
    """

    def __init__(self, model: GCNModel, device: str = 'cuda',
                 learning_rate: float = 1e-3, seed: int = 42):
        """
        Initialize the trainer.

        Args:
            model: GCN model to train
            device: Device to train on ('cuda' or 'cpu')
            learning_rate: Learning rate for optimizer
            seed: Random seed for reproducibility
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss(reduction='none')

        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.

        Args:
            train_loader: DataLoader for training data

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            batch = batch.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            pred = self.model(batch)

            # Compute loss only on masked nodes
            loss_per_node = self.criterion(pred, batch.y)
            masked_loss = loss_per_node[batch.mask]

            if len(masked_loss) > 0:
                loss = masked_loss.mean()

                # Backward pass
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(num_batches, 1)

    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate the model.

        Args:
            val_loader: DataLoader for validation data

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)

                # Forward pass
                pred = self.model(batch)

                # Compute loss only on masked nodes
                loss_per_node = self.criterion(pred, batch.y)
                masked_loss = loss_per_node[batch.mask]

                if len(masked_loss) > 0:
                    loss = masked_loss.mean()
                    total_loss += loss.item()
                    num_batches += 1

        return total_loss / max(num_batches, 1)

    def save_model(self, path: str):
        """
        Save model weights.

        Args:
            path: Path to save model
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def extract_and_save_activations(self, data_loader: DataLoader,
                                     output_dir: str, split_name: str, layer: int = 3):
        """
        Extract and save layer activations for all graphs.

        Args:
            data_loader: DataLoader for graphs
            output_dir: Base output directory
            split_name: Name of data split (train/val/test)
            layer: Which layer to save (1, 2, 3, or 4). Default: 3 (third hidden layer, 64-dim)
        """
        self.model.eval()

        layer_dirs = {
            1: Path(output_dir) / "activations" / "layer1_new" / split_name,
            2: Path(output_dir) / "activations" / "layer2_new" / split_name,
            3: Path(output_dir) / "activations" / "layer3_new" / split_name,
            4: Path(output_dir) / "activations" / "layer4_new" / split_name,
        }

        if layer not in [1, 2, 3, 4]:
            raise ValueError(f"layer must be 1, 2, 3, or 4, got {layer}")

        layer_dir = layer_dirs[layer]
        layer_dir.mkdir(parents=True, exist_ok=True)

        saved_count = 0

        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Extracting {split_name} activations (layer {layer})"):
                batch = batch.to(self.device)

                # Forward pass with activation storage
                _ = self.model(batch, store_activations=True)
                h1, h2, h3, h4 = self.model.get_activations()

                # Select which layer to save
                activations_map = {1: h1, 2: h2, 3: h3, 4: h4}
                h_layer = activations_map[layer]

                # Split batch back into individual graphs
                if hasattr(batch, 'batch'):
                    # Handle batched data
                    batch_indices = batch.batch
                    unique_batches = torch.unique(batch_indices)

                    for b in unique_batches:
                        mask = batch_indices == b
                        h_graph = h_layer[mask].cpu()

                        # Get original graph ID from batch
                        graph_id = int(batch.graph_id[b].item())

                        # Save activations with original graph ID
                        torch.save(h_graph, layer_dir / f"graph_{graph_id}.pt")
                        saved_count += 1
                else:
                    # Single graph
                    graph_id = int(batch.graph_id.item())
                    torch.save(h_layer.cpu(), layer_dir / f"graph_{graph_id}.pt")
                    saved_count += 1

        print(f"Saved layer {layer} activations for {saved_count} graphs to {output_dir}/activations/layer{layer}/")


def collate_fn(batch: List[Data]) -> Data:
    """
    Custom collate function for batching PyG Data objects.

    Args:
        batch: List of Data objects

    Returns:
        Batched Data object
    """
    return Batch.from_data_list(batch)


def load_all_graphs(data_dir: str = "./virtual_graphs/data", single_motif_only: bool = True) -> List[Path]:
    """
    Load all graph paths from the data directory.

    Args:
        data_dir: Base directory containing graph data
        single_motif_only: If True, only load single-motif graphs

    Returns:
        List of paths to graph pickle files
    """
    data_path = Path(data_dir)
    graph_paths = []

    # First try to load from new all_graphs/raw_graphs/ directory
    all_graphs_dir = data_path / "all_graphs" / "raw_graphs"
    if all_graphs_dir.exists():
        # Load all graphs from the new all_graphs directory
        all_pkl_files = sorted(all_graphs_dir.glob("*.pkl"))

        if single_motif_only:
            # Filter to only single-motif graphs (IDs 0-3999)
            graph_paths.extend([f for f in all_pkl_files if _is_single_motif_graph(f)])
        else:
            # Load all graphs (both single and mixed motif)
            graph_paths.extend(all_pkl_files)
    else:
        # Fall back to old directory structure for backward compatibility
        # Load single-motif graphs
        single_motif_dir = data_path / "single_motif_graphs"
        if single_motif_dir.exists():
            for motif_type_dir in single_motif_dir.iterdir():
                if motif_type_dir.is_dir():
                    graph_paths.extend(sorted(motif_type_dir.glob("*.pkl")))

        # Load mixed-motif graphs only if single_motif_only is False
        if not single_motif_only:
            mixed_motif_dir = data_path / "mixed_motif_graphs"
            if mixed_motif_dir.exists():
                graph_paths.extend(sorted(mixed_motif_dir.glob("*.pkl")))

    return graph_paths


def _is_single_motif_graph(graph_path: Path) -> bool:
    """
    Check if a graph is from the single-motif set based on its filename.
    Single-motif graphs have IDs 0-3999, mixed-motif graphs have IDs 4000-4999.

    Args:
        graph_path: Path to graph file

    Returns:
        True if the graph is single-motif, False otherwise
    """
    try:
        # Extract the number from filename like "graph_123.pkl"
        graph_id = int(graph_path.stem.split('_')[1])
        return graph_id < 4000
    except (IndexError, ValueError):
        # If we can't parse the ID, assume it's not single-motif
        return False


def _count_motif_distribution(paths: List[Path]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for path in paths:
        motif = _infer_motif_label(path)
        counts[motif] = counts.get(motif, 0) + 1
    return counts


def _print_split_stats(name: str, paths: List[Path]):
    counts = _count_motif_distribution(paths)
    motif_strings = [f"{motif}: {count}" for motif, count in sorted(counts.items())]
    details = ", ".join(motif_strings)
    print(f"{name} split -> total: {len(paths)} ({details})")


def _compute_motif_metrics(trainer: "GNNTrainer", data_loader: DataLoader) -> Dict[str, Dict[str, float]]:
    """Compute average masked MSE per motif for a given loader."""
    trainer.model.eval()
    motif_losses: Dict[str, List[float]] = defaultdict(list)
    motif_counts: Dict[str, int] = defaultdict(int)

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(trainer.device)
            pred = trainer.model(batch)
            loss_per_node = trainer.criterion(pred, batch.y)

            if hasattr(batch, 'batch'):
                batch_indices = batch.batch
            else:
                batch_indices = torch.zeros(batch.y.size(0), dtype=torch.long, device=batch.y.device)

            unique_graphs = torch.unique(batch_indices)

            for g in unique_graphs:
                node_mask = batch_indices == g
                masked_nodes = batch.mask[node_mask]
                node_losses = loss_per_node[node_mask]
                masked_losses = node_losses[masked_nodes]

                if masked_losses.numel() == 0:
                    continue

                graph_loss = masked_losses.mean().item()

                motif_label = "unknown"
                if hasattr(batch, 'motif_id'):
                    motif_tensor = batch.motif_id[g]
                    motif_idx = int(motif_tensor.item())
                    if 0 <= motif_idx < len(MOTIF_LABELS):
                        motif_label = MOTIF_LABELS[motif_idx]

                motif_losses[motif_label].append(graph_loss)
                motif_counts[motif_label] += 1

    metrics: Dict[str, Dict[str, float]] = {}
    for motif_label in sorted(motif_losses.keys()):
        losses = motif_losses[motif_label]
        metrics[motif_label] = {
            "num_graphs": motif_counts[motif_label],
            "mean_masked_mse": float(np.mean(losses)),
            "std_masked_mse": float(np.std(losses)) if len(losses) > 1 else 0.0
        }

    return metrics


def _save_json(data: Dict, path: str):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def _save_split_metadata(graph_paths: List[Path], split_name: str) -> None:
    """
    Save metadata mapping for a data split.

    Maps split indices (0-N) to original graph IDs and motif types.
    This enables tracking which activation belongs to which original graph
    when activations are stored sequentially per split (0-500 for test),
    but original graph IDs contain motif information (0-3999 single, 4000-4999 mixed).

    Args:
        graph_paths: List of graph paths in the order they appear in split
        split_name: Name of split (train/val/test)
    """
    metadata = {
        "split": split_name,
        "num_graphs": len(graph_paths),
        "mappings": []
    }

    for split_idx, graph_path in enumerate(graph_paths):
        graph_id = _extract_graph_id(graph_path)
        motif_type = _infer_motif_label(graph_path)

        metadata["mappings"].append({
            "split_idx": split_idx,
            "graph_id": graph_id,
            "motif_type": motif_type,
            "graph_path": str(graph_path)
        })

    # Save metadata
    output_dir = Path("outputs/activation_metadata")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{split_name}_metadata.json"

    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved {split_name} split metadata to {output_path}")

    # Print summary
    motif_counts = {}
    for mapping in metadata["mappings"]:
        motif = mapping["motif_type"]
        motif_counts[motif] = motif_counts.get(motif, 0) + 1

    print(f"  {split_name} motif distribution: {motif_counts}")


def split_data(graph_paths: List[Path], train_ratio: float = 0.8,
               val_ratio: float = 0.1, seed: int = 42,
               stratify_by_motif: bool = True,
               equal_counts_per_motif: bool = False) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Split graph paths into train/val/test sets.

    Args:
        graph_paths: List of all graph paths
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        seed: Random seed
        stratify_by_motif: Whether to split within motif groups
        equal_counts_per_motif: If True, enforce equal per-motif counts in each
            split (drops excess graphs per motif)

    Returns:
        Tuple of (train_paths, val_paths, test_paths)
    """
    rng = np.random.default_rng(seed)

    if not stratify_by_motif:
        n_graphs = len(graph_paths)
        indices = rng.permutation(n_graphs)

        n_train = int(n_graphs * train_ratio)
        n_val = int(n_graphs * val_ratio)

        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]

        train_paths = [graph_paths[i] for i in train_indices]
        val_paths = [graph_paths[i] for i in val_indices]
        test_paths = [graph_paths[i] for i in test_indices]

        return train_paths, val_paths, test_paths

    motif_groups: Dict[str, List[Path]] = {}
    for path in graph_paths:
        motif = _infer_motif_label(path)
        motif_groups.setdefault(motif, []).append(path)

    if equal_counts_per_motif and motif_groups:
        non_empty_counts = [len(paths) for paths in motif_groups.values() if paths]
        usable_count = min(non_empty_counts) if non_empty_counts else 0
    else:
        usable_count = None

    train_paths: List[Path] = []
    val_paths: List[Path] = []
    test_paths: List[Path] = []

    for motif_paths in motif_groups.values():
        if not motif_paths:
            continue

        shuffled = list(np.array(motif_paths)[rng.permutation(len(motif_paths))])

        if equal_counts_per_motif and usable_count is not None:
            shuffled = shuffled[:usable_count]

        n = len(shuffled)
        if n == 0:
            continue

        n_train = int(np.floor(n * train_ratio))
        n_val = int(np.floor(n * val_ratio))
        n_test = n - n_train - n_val

        train_paths.extend(shuffled[:n_train])
        val_paths.extend(shuffled[n_train:n_train + n_val])
        test_paths.extend(shuffled[n_train + n_val:n_train + n_val + n_test])

    train_paths = list(np.array(train_paths)[rng.permutation(len(train_paths))]) if train_paths else []
    val_paths = list(np.array(val_paths)[rng.permutation(len(val_paths))]) if val_paths else []
    test_paths = list(np.array(test_paths)[rng.permutation(len(test_paths))]) if test_paths else []

    return train_paths, val_paths, test_paths


def main(model_type: str = "GCN"):
    """Main training pipeline."""
    # Configuration
    SEED = 42
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 128  # From MISATO paper for QM graph tasks 
    NUM_EPOCHS = 100  # Standard epochs with early stopping (patience=25) to prevent overfitting
    LEARNING_RATE = 0.036  # Standard learning rate for Adam optimizer with GNNs
    MASK_PROB = 0.2  # Node masking probability for inductive learning task
                      # Selected to create moderate sparsity while maintaining sufficient signal

    print(f"Using device: {DEVICE}")
    print(f"Model type: {model_type}")
    print(f"Loading graphs from ./virtual_graphs/data/")

    # Load and split data
    all_graph_paths = load_all_graphs(single_motif_only=True)
    print(f"Found {len(all_graph_paths)} graphs")

    if len(all_graph_paths) == 0:
        print("Error: No graphs found. Please run graph_motif_generator.py first.")
        return

    train_paths, val_paths, test_paths = split_data(
        all_graph_paths,
        seed=SEED,
        stratify_by_motif=True,
        equal_counts_per_motif=False
    )
    print(f"Split: {len(train_paths)} train, {len(val_paths)} val, {len(test_paths)} test")
    print("Motif distribution per split:")
    _print_split_stats("  Train", train_paths)
    _print_split_stats("  Val", val_paths)
    _print_split_stats("  Test", test_paths)

    # Save metadata mapping for activations
    print("\nSaving activation metadata...")
    _save_split_metadata(train_paths, "train")
    _save_split_metadata(val_paths, "val")
    _save_split_metadata(test_paths, "test")

    # Create datasets
    train_dataset = GraphDataset(train_paths, mask_prob=MASK_PROB, seed=SEED)
    val_dataset = GraphDataset(val_paths, mask_prob=MASK_PROB, seed=SEED + 1)
    test_dataset = GraphDataset(test_paths, mask_prob=MASK_PROB, seed=SEED + 2)

    # Create data loaders (pin_memory=True for GPU, num_workers for parallel loading)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, collate_fn=collate_fn, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                           shuffle=False, collate_fn=collate_fn, pin_memory=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=collate_fn, pin_memory=True, num_workers=2)
    train_eval_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                   shuffle=False, collate_fn=collate_fn, pin_memory=True, num_workers=2)

    # Initialize model and trainer
    model_type_upper = model_type.upper()
    if model_type_upper == "GAT":
        model = GATModel(
            input_dim=2,
            hidden_dim=16,
            output_dim=1,
            dropout=0.2,
            num_heads=4,
            edge_dim=1
        )
        model_name = "gat_model.pt"
        print("\nInitialized GAT model (multi-head attention)")
        print("  - Layer 1: 2 -> 16 x 4 heads = 64 dims (1-hop)")
        print("  - Layer 2: 64 -> 16 x 4 heads = 64 dims (2-hop)")
        print("  - Layer 3: 64 -> 8 x 8 heads = 64 dims (3-hop)")
        print("  - Layer 4: 64 -> 1 (single-head, 4-hop)")
    elif model_type_upper == "GCN":
        model = GCNModel(input_dim=2, hidden_dim=80, output_dim=1, dropout=0.3)
        model_name = "gnn_model.pt"
        print("\nInitialized GCN model")
        print("  - Layer 1: 2 -> 88 (1-hop)")
        print("  - Layer 2: 88 -> 88 (2-hop)")
        print("  - Layer 3: 88 -> 64 (3-hop)")
        print("  - Layer 4: 64 -> 1 (4-hop)")
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'GCN' or 'GAT'.")

    trainer = GNNTrainer(model, device=DEVICE, learning_rate=LEARNING_RATE, seed=SEED)

    # Training loop
    print(f"\nTraining {model_type_upper} model...")
    best_val_loss = float('inf')
    patience = 25
    patience_counter = 0
    best_epoch = -1
    training_metrics = {
        "model_type": model_type_upper,
        "train_loss": [],
        "val_loss": []
    }

    for epoch in tqdm(range(NUM_EPOCHS), desc="Training"):
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.validate(val_loader)

        training_metrics["train_loss"].append(train_loss)
        training_metrics["val_loss"].append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_epoch = epoch
            trainer.save_model(f"checkpoints/{model_name}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss = trainer.validate(test_loader)
    print(f"Test Loss: {test_loss:.4f}")

    training_metrics["best_val_loss"] = float(best_val_loss)
    training_metrics["best_epoch"] = best_epoch + 1 if best_epoch >= 0 else None
    training_metrics["test_loss"] = float(test_loss)
    _save_json(training_metrics, "outputs/training_metrics.json")
    print("Saved training metrics to outputs/training_metrics.json")

    # Extract and save activations from all three hidden layers
    print("\nExtracting activations from layers 1, 2, and 3...")
    for layer in [1, 2, 3]:
        print(f"  Extracting layer {layer} activations...")
        trainer.extract_and_save_activations(train_loader, "outputs", "train", layer=layer)
        trainer.extract_and_save_activations(val_loader, "outputs", "val", layer=layer)
        trainer.extract_and_save_activations(test_loader, "outputs", "test", layer=layer)

    motif_metrics = {
        "train": _compute_motif_metrics(trainer, train_eval_loader),
        "val": _compute_motif_metrics(trainer, val_loader),
        "test": _compute_motif_metrics(trainer, test_loader)
    }
    _save_json(motif_metrics, "outputs/motif_metrics.json")
    print("Saved motif metrics to outputs/motif_metrics.json")

    print("\nTraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GNN (GCN or GAT) for motif prediction.")
    parser.add_argument(
        "--model_type",
        type=str,
        default="GCN",
        choices=["GCN", "GAT", "gcn", "gat"],
        help="Model architecture to train."
    )
    args = parser.parse_args()
    main(model_type=args.model_type)
