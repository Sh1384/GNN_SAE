"""
Comprehensive Benchmarking Script for GNN Interpretability Study

Implements:
1. Baseline models (mean/median predictor, MLP)
2. Data sensitivity analysis (timesteps and noise levels)
3. Multi-seed training with statistical analysis
4. Results tables and visualizations
"""

import argparse
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy import stats

# Import from existing code
from gnn_train import (
    GraphDataset, GCNModel, GATModel, GNNTrainer,
    load_all_graphs, split_data, collate_fn, MOTIF_LABELS, MOTIF_TO_ID
)


# ============================================================================
# BASELINE MODELS
# ============================================================================

class MeanMedianBaseline(nn.Module):
    """
    Baseline that predicts mean or median of observed node values.
    """
    def __init__(self, statistic: str = 'mean'):
        super().__init__()
        self.statistic = statistic
        self.observed_values = None

    def fit(self, train_loader: DataLoader):
        """
        Compute mean or median from training data.
        """
        all_observed = []

        for batch in train_loader:
            observed_mask = ~batch.mask
            observed_values = batch.y[observed_mask]
            # Ensure we extract numpy array correctly, handling potential device placement
            all_observed.extend(observed_values.detach().cpu().numpy())

        all_observed = np.array(all_observed)

        if self.statistic == 'mean':
            self.observed_values = np.mean(all_observed)
        else:  # median
            self.observed_values = np.median(all_observed)

    def forward(self, data: Data) -> torch.Tensor:
        """Predict same value for all masked nodes."""
        n_nodes = data.x.shape[0]
        predictions = torch.full((n_nodes,), self.observed_values, dtype=torch.float32)
        return predictions


class MLPBaseline(nn.Module):
    """
    Simple MLP baseline that uses only node features (no graph structure).
    Architecture: 2 -> 128 -> 64 -> 1
    """
    def __init__(self, input_dim: int = 2, hidden_dim1: int = 128,
                 hidden_dim2: int = 64, output_dim: int = 1, dropout: float = 0.2):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data: Data) -> torch.Tensor:
        """Process through MLP using only node features."""
        x = data.x

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        x = self.fc3(x)
        return x.squeeze(-1)


# ============================================================================
# DATA GENERATION WITH SENSITIVITY ANALYSIS
# ============================================================================

class DataVariationDataset(Dataset):
    """
    Extended GraphDataset with configurable timesteps and noise levels
    for sensitivity analysis.
    """
    def __init__(self, graph_paths: List[Path], mask_prob: float = 0.2,
                 seed: int = 42, steps: int = 50, noise_std: float = 0.01):
        """
        Args:
            graph_paths: List of paths to pickled graph files
            mask_prob: Probability of masking nodes
            seed: Random seed
            steps: Number of simulation steps
            noise_std: Standard deviation of Gaussian noise
        """
        self.graph_paths = graph_paths
        self.mask_prob = mask_prob
        self.rng = np.random.default_rng(seed)
        self.steps = steps
        self.noise_std = noise_std

    def __len__(self) -> int:
        return len(self.graph_paths)

    def __getitem__(self, idx: int) -> Data:
        """Load graph and simulate expression with custom parameters."""
        with open(self.graph_paths[idx], 'rb') as f:
            G = pickle.load(f)

        # Get adjacency matrix
        import networkx as nx
        n_nodes = len(G.nodes())
        W = nx.to_numpy_array(G, weight='weight')

        # Simulate expression with custom parameters
        expression = self._simulate_expression(W)

        # Create mask
        mask = self.rng.random(n_nodes) < self.mask_prob
        mask = torch.tensor(mask, dtype=torch.bool)

        # Create features
        expression_tensor = torch.tensor(expression, dtype=torch.float32)
        masked_expression = expression_tensor.clone()
        masked_expression[mask] = 0.0

        if masked_expression.max() > 0:
            masked_expression = masked_expression / (masked_expression.max() + 1e-8)

        mask_flag = (~mask).float().unsqueeze(1)
        x = torch.cat([masked_expression.unsqueeze(1), mask_flag], dim=1)

        # Create edge index
        edge_index = torch.tensor(np.array(np.nonzero(W)), dtype=torch.long)
        edge_weight = torch.tensor(W[W != 0], dtype=torch.float32)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_weight,
            y=expression_tensor,
            mask=mask,
            num_nodes=n_nodes
        )

        return data

    def _simulate_expression(self, W: np.ndarray) -> np.ndarray:
        """Simulate expression with custom parameters."""
        n_nodes = W.shape[0]
        x = self.rng.uniform(0, 1, size=n_nodes)

        for _ in range(self.steps):
            weighted_input = W @ x
            sigmoid_input = 1.0 / (1.0 + np.exp(-np.clip(weighted_input, -10, 10)))
            noise = self.rng.normal(0, self.noise_std, size=n_nodes)
            x = (1 - 0.3) * x + 0.3 * sigmoid_input + noise
            x = np.clip(x, 0, 1)

        return x


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

class BaselineTrainer:
    """Trainer for baseline models."""

    def __init__(self, model: nn.Module, device: str = 'cuda',
                 learning_rate: float = 1e-3, seed: int = 42):
        self.model = model.to(device)
        self.device = device
        # Only create optimizer if model has parameters (skip for non-parametric baselines)
        model_params = list(model.parameters())
        self.optimizer = torch.optim.Adam(model_params, lr=learning_rate) if model_params else None
        self.criterion = nn.MSELoss(reduction='none')

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        for batch in train_loader:
            batch = batch.to(self.device)

            # Skip optimization for non-parametric models (no optimizer)
            if self.optimizer is not None:
                self.optimizer.zero_grad()

            pred = self.model(batch)

            loss_per_node = self.criterion(pred, batch.y)
            masked_loss = loss_per_node[batch.mask]
            loss = masked_loss.mean()

            if self.optimizer is not None:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_loader: DataLoader) -> float:
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                pred = self.model(batch)

                loss_per_node = self.criterion(pred, batch.y)
                masked_loss = loss_per_node[batch.mask]
                loss = masked_loss.mean()

                total_loss += loss.item()

        return total_loss / len(val_loader)

    def save_model(self, save_path: str):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)


# ============================================================================
# BENCHMARKING PIPELINE
# ============================================================================

class BenchmarkExperiment:
    """Manages a complete benchmarking experiment."""

    def __init__(self, output_dir: str = "outputs/benchmark"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = defaultdict(dict)

    def run_multi_seed_training(self,
                               model_type: str = "GCN",
                               n_seeds: int = 5,
                               num_epochs: int = 100,
                               batch_size: int = 128,
                               learning_rate: float = 1e-3) -> Dict:
        """
        Train GNN/baseline models across multiple seeds.

        Args:
            model_type: 'GCN', 'GAT', 'MLP', 'MeanMedian'
            n_seeds: Number of random seeds to run
            num_epochs: Maximum number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate

        Returns:
            Dictionary with results across seeds
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load and split data
        all_graph_paths = load_all_graphs(single_motif_only=True)
        if not all_graph_paths:
            print("Error: No graphs found")
            return {}

        # Storage for results
        seed_results = {
            'train_loss': [],
            'val_loss': [],
            'test_loss': [],
            'best_epoch': [],
            'motif_metrics': []  # Store motif metrics per seed
        }

        print(f"\nRunning {n_seeds} seeds for {model_type}...")

        for seed in range(n_seeds):
            print(f"\n--- Seed {seed} ---")

            # Set random seeds for reproducibility and to control stochasticity
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Split data with different seed
            train_paths, val_paths, test_paths = split_data(
                all_graph_paths,
                seed=seed,
                stratify_by_motif=True,
                equal_counts_per_motif=True
            )

            # Create datasets and loaders
            train_dataset = GraphDataset(train_paths, mask_prob=0.2, seed=seed)
            val_dataset = GraphDataset(val_paths, mask_prob=0.2, seed=seed + 1)
            test_dataset = GraphDataset(test_paths, mask_prob=0.2, seed=seed + 2)

            train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                     shuffle=True, collate_fn=collate_fn, pin_memory=True, num_workers=2)
            val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                   shuffle=False, collate_fn=collate_fn, pin_memory=True, num_workers=2)
            test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                    shuffle=False, collate_fn=collate_fn, pin_memory=True, num_workers=2)

            # Initialize model
            if model_type == 'GCN':
                model = GCNModel(input_dim=2, hidden_dim=128,
                               output_dim=1, dropout=0.2)
            elif model_type == 'GAT':
                model = GATModel(input_dim=2, hidden_dim=32, output_dim=1,
                               dropout=0.2, num_heads=4, edge_dim=1)
            elif model_type == 'MLP':
                model = MLPBaseline(input_dim=2, hidden_dim1=128, hidden_dim2=64,
                                  output_dim=1, dropout=0.2)
            elif model_type == 'MeanMedian':
                model = MeanMedianBaseline(statistic='mean')
                model.fit(train_loader)
                # Skip training for mean/median baseline
                trainer = BaselineTrainer(model, device=device)
                val_loss = trainer.validate(val_loader)
                test_loss = trainer.validate(test_loader)
                # Compute motif-specific metrics
                motif_metrics = self.compute_motif_metrics(trainer.model, test_loader, device=device)
                seed_results['train_loss'].append(val_loss)
                seed_results['val_loss'].append(val_loss)
                seed_results['test_loss'].append(test_loss)
                seed_results['best_epoch'].append(1)
                seed_results['motif_metrics'].append(motif_metrics)
                continue
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            trainer = BaselineTrainer(model, device=device, learning_rate=learning_rate)

            # Training loop
            best_val_loss = float('inf')
            patience = 25
            patience_counter = 0
            best_epoch = -1

            for epoch in tqdm(range(num_epochs), desc=f"{model_type} training"):
                train_loss = trainer.train_epoch(train_loader)
                val_loss = trainer.validate(val_loader)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_epoch = epoch
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

            # Test evaluation
            test_loss = trainer.validate(test_loader)

            # Compute motif-specific metrics
            motif_metrics = self.compute_motif_metrics(trainer.model, test_loader, device=device)

            # Store results
            seed_results['train_loss'].append(train_loss)
            seed_results['val_loss'].append(best_val_loss)
            seed_results['test_loss'].append(test_loss)
            seed_results['best_epoch'].append(best_epoch + 1)
            seed_results['motif_metrics'].append(motif_metrics)

            print(f"Best epoch: {best_epoch + 1}, Val Loss: {best_val_loss:.4f}, Test Loss: {test_loss:.4f}")

        return seed_results

    def run_sensitivity_analysis(self,
                                timestep_values: List[int] = [25, 50, 75],
                                noise_values: List[float] = [0.005, 0.01, 0.05],
                                model_type: str = "GCN",
                                num_epochs: int = 100,
                                batch_size: int = 32) -> Dict:
        """
        Run sensitivity analysis on timesteps and noise levels.
        GCN and GAT are trained on both single-motif and mixed-motif graphs.
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load all graphs (single-motif + mixed-motif) for GNN methods
        all_graph_paths = load_all_graphs(single_motif_only=False)
        if not all_graph_paths:
            return {}

        results = {}

        print(f"\nRunning sensitivity analysis for {model_type}...")

        # Analyze timesteps
        print("\nTesting timesteps...")
        timestep_results = []

        for steps in timestep_values:
            print(f"  Timesteps: {steps}")

            # Set random seeds for reproducibility
            torch.manual_seed(42)
            np.random.seed(42)

            train_paths, val_paths, test_paths = split_data(
                all_graph_paths, seed=42, stratify_by_motif=True, equal_counts_per_motif=True
            )

            train_dataset = DataVariationDataset(train_paths, steps=steps, noise_std=0.01, seed=42)
            val_dataset = DataVariationDataset(val_paths, steps=steps, noise_std=0.01, seed=43)
            test_dataset = DataVariationDataset(test_paths, steps=steps, noise_std=0.01, seed=44)

            train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                     shuffle=True, collate_fn=collate_fn, pin_memory=True, num_workers=2)
            val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                   shuffle=False, collate_fn=collate_fn, pin_memory=True, num_workers=2)
            test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                    shuffle=False, collate_fn=collate_fn, pin_memory=True, num_workers=2)

            # Train model
            if model_type == 'GCN':
                model = GCNModel(input_dim=2, hidden_dim=128,
                               output_dim=1, dropout=0.2)
            elif model_type == 'MLP':
                model = MLPBaseline()
            else:
                model = GATModel(input_dim=2, hidden_dim=32, output_dim=1,
                               dropout=0.2, num_heads=4, edge_dim=1)

            trainer = BaselineTrainer(model, device=device)

            best_val_loss = float('inf')
            patience_counter = 0

            for epoch in range(num_epochs):
                train_loss = trainer.train_epoch(train_loader)
                val_loss = trainer.validate(val_loader)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= 25:
                        break

            test_loss = trainer.validate(test_loader)
            timestep_results.append({
                'timesteps': steps,
                'val_loss': best_val_loss,
                'test_loss': test_loss
            })

        results['timesteps'] = timestep_results

        # Analyze noise levels
        print("\nTesting noise levels...")
        noise_results = []

        for noise_std in noise_values:
            print(f"  Noise std: {noise_std}")

            # Set random seeds for reproducibility
            torch.manual_seed(42)
            np.random.seed(42)

            train_dataset = DataVariationDataset(train_paths, steps=50, noise_std=noise_std, seed=42)
            val_dataset = DataVariationDataset(val_paths, steps=50, noise_std=noise_std, seed=43)
            test_dataset = DataVariationDataset(test_paths, steps=50, noise_std=noise_std, seed=44)

            train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                     shuffle=True, collate_fn=collate_fn, pin_memory=True, num_workers=2)
            val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                   shuffle=False, collate_fn=collate_fn, pin_memory=True, num_workers=2)
            test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                    shuffle=False, collate_fn=collate_fn, pin_memory=True, num_workers=2)

            # Train model
            if model_type == 'GCN':
                model = GCNModel(input_dim=2, hidden_dim=128,
                               output_dim=1, dropout=0.2)
            elif model_type == 'MLP':
                model = MLPBaseline()
            else:
                model = GATModel(input_dim=2, hidden_dim=32, output_dim=1,
                               dropout=0.2, num_heads=4, edge_dim=1)

            trainer = BaselineTrainer(model, device=device)

            best_val_loss = float('inf')
            patience_counter = 0

            for epoch in range(num_epochs):
                train_loss = trainer.train_epoch(train_loader)
                val_loss = trainer.validate(val_loader)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= 25:
                        break

            test_loss = trainer.validate(test_loader)
            noise_results.append({
                'noise_std': noise_std,
                'val_loss': best_val_loss,
                'test_loss': test_loss
            })

        results['noise'] = noise_results

        return results

    def generate_results_table(self, seed_results: Dict, model_name: str) -> pd.DataFrame:
        """
        Generate comprehensive statistics table from multi-seed results.
        Includes mean, median, percentiles, std dev, CI, and min/max.
        """
        test_losses = np.array(seed_results['test_loss'])
        val_losses = np.array(seed_results['val_loss'])

        n_seeds = len(test_losses)

        # Compute all statistics
        mean_loss = np.mean(test_losses)
        median_loss = np.median(test_losses)
        std_loss = np.std(test_losses)
        p25 = np.percentile(test_losses, 25)
        p75 = np.percentile(test_losses, 75)
        iqr = p75 - p25

        # Confidence interval (95%)
        se = std_loss / np.sqrt(n_seeds)
        ci_95 = 1.96 * se

        # Also compute for validation losses
        mean_val_loss = np.mean(val_losses)
        median_val_loss = np.median(val_losses)

        results_df = pd.DataFrame({
            'Model': [model_name],
            'Test Loss (Mean)': [f"{mean_loss:.4f}"],
            'Test Loss (Median)': [f"{median_loss:.4f}"],
            'P25': [f"{p25:.4f}"],
            'P75': [f"{p75:.4f}"],
            'IQR': [f"{iqr:.4f}"],
            'Std Dev': [f"{std_loss:.4f}"],
            '95% CI': [f"±{ci_95:.4f}"],
            'Min': [f"{np.min(test_losses):.4f}"],
            'Max': [f"{np.max(test_losses):.4f}"],
            'N Seeds': [n_seeds]
        })

        return results_df

    def generate_detailed_statistics(self, seed_results: Dict, model_name: str) -> Dict:
        """
        Generate detailed statistics dictionary for further analysis and plotting.
        """
        test_losses = np.array(seed_results['test_loss'])
        val_losses = np.array(seed_results['val_loss'])
        train_losses = np.array(seed_results['train_loss'])
        best_epochs = np.array(seed_results['best_epoch'])

        n_seeds = len(test_losses)
        se = np.std(test_losses) / np.sqrt(n_seeds)

        stats = {
            'model': model_name,
            'n_seeds': n_seeds,

            # Test Loss Statistics
            'test_loss': {
                'mean': np.mean(test_losses),
                'median': np.median(test_losses),
                'std': np.std(test_losses),
                'p25': np.percentile(test_losses, 25),
                'p75': np.percentile(test_losses, 75),
                'iqr': np.percentile(test_losses, 75) - np.percentile(test_losses, 25),
                'min': np.min(test_losses),
                'max': np.max(test_losses),
                'ci_95': 1.96 * se,
                'se': se,
                'values': test_losses.tolist()
            },

            # Val Loss Statistics
            'val_loss': {
                'mean': np.mean(val_losses),
                'median': np.median(val_losses),
                'std': np.std(val_losses),
                'p25': np.percentile(val_losses, 25),
                'p75': np.percentile(val_losses, 75),
                'iqr': np.percentile(val_losses, 75) - np.percentile(val_losses, 25),
                'min': np.min(val_losses),
                'max': np.max(val_losses),
                'values': val_losses.tolist()
            },

            # Train Loss Statistics
            'train_loss': {
                'mean': np.mean(train_losses),
                'median': np.median(train_losses),
                'std': np.std(train_losses),
                'values': train_losses.tolist()
            },

            # Best Epoch Statistics
            'best_epoch': {
                'mean': np.mean(best_epochs),
                'median': np.median(best_epochs),
                'values': best_epochs.tolist()
            }
        }

        return stats

    def compute_pairwise_comparisons(self, all_seed_results: Dict[str, Dict]) -> Dict:
        """
        Compute pairwise comparisons between models using Wilcoxon signed-rank test.
        Comparisons: (GCN, MeanMedian), (GAT, MeanMedian), (GCN, MLP), (GAT, MLP)

        Args:
            all_seed_results: Dictionary with seed results for all models
                Format: {model_name: {'test_loss': [values across seeds], ...}}

        Returns:
            Dictionary with pairwise comparison results including:
            - statistic: Wilcoxon test statistic
            - p_value: Two-tailed p-value
            - rank_biserial: Rank-biserial correlation (effect size)
            - mean_diff: Mean difference between groups
            - direction: Which model performed better
        """
        # Define comparison pairs: GNN vs non-GNN methods
        comparison_pairs = [
            ('GCN', 'MeanMedian'),
            ('GAT', 'MeanMedian'),
            ('GCN', 'MLP'),
            ('GAT', 'MLP')
        ]

        comparisons = {}

        for model_a, model_b in comparison_pairs:
            if model_a not in all_seed_results or model_b not in all_seed_results:
                continue

            # Get test loss values for both models
            losses_a = np.array(all_seed_results[model_a]['test_loss'])
            losses_b = np.array(all_seed_results[model_b]['test_loss'])

            # Ensure same number of seeds
            if len(losses_a) != len(losses_b):
                print(f"Warning: {model_a} and {model_b} have different number of seeds")
                continue

            # Wilcoxon signed-rank test (paired, non-parametric)
            statistic, p_value = stats.wilcoxon(losses_a, losses_b)

            # Compute rank-biserial correlation (effect size for Wilcoxon)
            # Formula: r = 1 - (2R / (n * (n + 1)))
            # where R is sum of ranks for positive differences
            differences = losses_a - losses_b
            n = len(differences)

            # Compute ranks of absolute differences
            abs_diff = np.abs(differences)
            ranks = stats.rankdata(abs_diff)

            # Sum ranks for positive differences
            positive_diff_mask = differences > 0
            r_plus = np.sum(ranks[positive_diff_mask])

            # Rank-biserial correlation
            rank_biserial = 1 - (2 * r_plus) / (n * (n + 1))

            # Bootstrap 95% CI for rank-biserial effect size
            n_bootstrap = 10000
            bootstrap_rank_biserials = []

            np.random.seed(42)  # For reproducibility
            for _ in range(n_bootstrap):
                # Resample differences with replacement
                boot_indices = np.random.choice(n, size=n, replace=True)
                boot_differences = differences[boot_indices]

                # Compute rank-biserial for bootstrap sample
                boot_abs_diff = np.abs(boot_differences)
                boot_ranks = stats.rankdata(boot_abs_diff)
                boot_positive_mask = boot_differences > 0
                boot_r_plus = np.sum(boot_ranks[boot_positive_mask])
                boot_rank_biserial = 1 - (2 * boot_r_plus) / (n * (n + 1))
                bootstrap_rank_biserials.append(boot_rank_biserial)

            bootstrap_rank_biserials = np.array(bootstrap_rank_biserials)
            rank_biserial_ci_lower = np.percentile(bootstrap_rank_biserials, 2.5)
            rank_biserial_ci_upper = np.percentile(bootstrap_rank_biserials, 97.5)

            # Determine which model is better
            mean_a = np.mean(losses_a)
            mean_b = np.mean(losses_b)
            better_model = model_a if mean_a < mean_b else model_b
            mean_diff = abs(mean_a - mean_b)

            # Interpret significance at alpha=0.05
            is_significant = p_value < 0.05

            comparisons[f"{model_a} vs {model_b}"] = {
                'wilcoxon_statistic': float(statistic),
                'p_value': float(p_value),
                'rank_biserial': float(rank_biserial),
                'rank_biserial_ci_lower': float(rank_biserial_ci_lower),
                'rank_biserial_ci_upper': float(rank_biserial_ci_upper),
                'rank_biserial_ci_str': f"{rank_biserial:.4f} [95% CI: {rank_biserial_ci_lower:.4f} to {rank_biserial_ci_upper:.4f}]",
                'mean_diff': float(mean_diff),
                'mean_loss_a': float(mean_a),
                'mean_loss_b': float(mean_b),
                'better_model': better_model,
                'is_significant': bool(is_significant),
                'interpretation': f"{better_model} performs significantly better" if is_significant
                                else f"No significant difference (p={p_value:.4f})",
                'n_seeds': int(n)
            }

        return comparisons

    def compute_motif_metrics(self, model: nn.Module, test_loader: DataLoader,
                             device: str = 'cuda') -> Dict[str, Dict[str, float]]:
        """
        Compute MSE and MAE metrics per motif type for post-training evaluation.
        Adapted from gnn_train.py but also computes MAE for comprehensive comparison.

        Args:
            model: Trained model to evaluate
            test_loader: DataLoader with test data
            device: Device to run on

        Returns:
            Dict with per-motif metrics: {motif_label: {mean_mse, std_mse, mean_mae, std_mae, num_graphs}}
        """
        model.to(device)
        model.eval()
        criterion_mse = nn.MSELoss(reduction='none')

        motif_losses = defaultdict(list)
        motif_counts = defaultdict(int)

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                pred = model(batch)

                # Compute per-node losses
                loss_per_node_mse = criterion_mse(pred, batch.y)

                # Extract motif information from batch
                if hasattr(batch, 'batch'):
                    batch_indices = batch.batch
                else:
                    batch_indices = torch.zeros(batch.y.size(0), dtype=torch.long, device=device)

                unique_graphs = torch.unique(batch_indices)

                for g in unique_graphs:
                    node_mask = batch_indices == g
                    masked_nodes = batch.mask[node_mask]
                    node_losses_mse = loss_per_node_mse[node_mask]
                    masked_losses_mse = node_losses_mse[masked_nodes]

                    if masked_losses_mse.numel() == 0:
                        continue

                    # Compute MSE
                    graph_mse = masked_losses_mse.mean().item()

                    # Compute MAE
                    masked_preds = pred[node_mask][masked_nodes]
                    masked_targets = batch.y[node_mask][masked_nodes]
                    graph_mae = torch.abs(masked_preds - masked_targets).mean().item()

                    # Get motif label from batch
                    motif_label = "unknown"
                    if hasattr(batch, 'motif_id'):
                        try:
                            motif_tensor = batch.motif_id[g]
                            motif_idx = int(motif_tensor.item())
                            from gnn_train import MOTIF_LABELS
                            if 0 <= motif_idx < len(MOTIF_LABELS):
                                motif_label = MOTIF_LABELS[motif_idx]
                        except (AttributeError, IndexError, ValueError, ImportError) as e:
                            # Could not extract motif label from batch
                            print(f"Warning: Failed to extract motif label for graph {g}: {type(e).__name__}: {str(e)}")
                            pass

                    # Store metrics
                    motif_losses[motif_label].append({
                        'mse': graph_mse,
                        'mae': graph_mae
                    })
                    motif_counts[motif_label] += 1

        # Aggregate metrics per motif
        metrics = {}
        for motif_label in sorted(motif_losses.keys()):
            losses = motif_losses[motif_label]
            mses = [l['mse'] for l in losses]
            maes = [l['mae'] for l in losses]

            metrics[motif_label] = {
                'num_graphs': motif_counts[motif_label],
                'mean_mse': float(np.mean(mses)),
                'std_mse': float(np.std(mses)) if len(mses) > 1 else 0.0,
                'mean_mae': float(np.mean(maes)),
                'std_mae': float(np.std(maes)) if len(maes) > 1 else 0.0
            }

        return metrics

    def save_results(self, results: Dict, filename: str):
        """Save results to JSON."""
        save_path = self.output_dir / filename
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {save_path}")

    # ========================================================================
    # VISUALIZATION METHODS
    # ========================================================================

    def plot_baseline_comparison(self, all_detailed_stats: Dict):
        """
        Plot comparison of test loss across baseline and GNN models.
        Shows mean, median, and 25th/75th percentile bounds.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        models = list(all_detailed_stats.keys())
        means = [all_detailed_stats[m]['test_loss']['mean'] for m in models]
        medians = [all_detailed_stats[m]['test_loss']['median'] for m in models]
        p25s = [all_detailed_stats[m]['test_loss']['p25'] for m in models]
        p75s = [all_detailed_stats[m]['test_loss']['p75'] for m in models]
        mins = [all_detailed_stats[m]['test_loss']['min'] for m in models]
        maxs = [all_detailed_stats[m]['test_loss']['max'] for m in models]

        x = np.arange(len(models))
        width = 0.35

        # Plot 1: Mean vs Median
        axes[0].bar(x - width/2, means, width, label='Mean', alpha=0.8, color='steelblue')
        axes[0].bar(x + width/2, medians, width, label='Median', alpha=0.8, color='coral')
        axes[0].set_ylabel('Test Loss', fontsize=12)
        axes[0].set_title('Mean vs Median Test Loss Across Models', fontsize=13, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models, rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)

        # Plot 2: Distribution with quartiles
        for i, model in enumerate(models):
            # Plot min-max range
            axes[1].plot([i, i], [mins[i], maxs[i]], 'k-', linewidth=2, alpha=0.3)

            # Plot 25th-75th percentile (IQR)
            axes[1].plot([i, i], [p25s[i], p75s[i]], 'o-', linewidth=8, markersize=10,
                        label=f'{model}' if i < len(models) else '', alpha=0.6)

            # Plot mean
            axes[1].scatter(i, means[i], marker='D', s=150, color='red', zorder=5, alpha=0.8)

            # Plot median
            axes[1].scatter(i, medians[i], marker='s', s=100, color='green', zorder=5, alpha=0.8)

        axes[1].set_ylabel('Test Loss', fontsize=12)
        axes[1].set_title('Test Loss Distribution: Min-Max, IQR (P25-P75), Mean, Median', fontsize=13, fontweight='bold')
        axes[1].set_xticks(range(len(models)))
        axes[1].set_xticklabels(models, rotation=45, ha='right')
        axes[1].grid(axis='y', alpha=0.3)

        # Add legend for markers
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='k', linewidth=2, alpha=0.3, label='Min-Max'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='C0', markersize=10, alpha=0.6, label='IQR (P25-P75)'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='red', markersize=8, label='Mean'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='green', markersize=8, label='Median')
        ]
        axes[1].legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()
        fig.savefig(self.output_dir / 'baseline_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Saved plot to {self.output_dir / 'baseline_comparison.png'}")
        plt.close()

    def plot_seed_variance(self, all_detailed_stats: Dict):
        """
        Plot test loss across seeds for each model (box plot with individual points).
        """
        fig, ax = plt.subplots(figsize=(14, 7))

        models = list(all_detailed_stats.keys())
        seed_losses = [all_detailed_stats[m]['test_loss']['values'] for m in models]

        # Box plot with individual points
        bp = ax.boxplot(seed_losses, labels=models, patch_artist=True,
                       widths=0.6, showmeans=True,
                       meanprops=dict(marker='D', markerfacecolor='red', markersize=8))

        # Color the boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Overlay individual seed points
        for i, (model, losses) in enumerate(zip(models, seed_losses)):
            y = losses
            x = np.random.normal(i + 1, 0.04, size=len(y))
            ax.scatter(x, y, alpha=0.6, s=80, color='darkblue')

        ax.set_ylabel('Test Loss', fontsize=12)
        ax.set_title('Test Loss Distribution Across Seeds (Box Plot + Individual Runs)', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='D', color='w', markerfacecolor='red', markersize=8, label='Mean'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='darkblue', markersize=8, label='Individual Seeds')
        ]
        ax.legend(handles=legend_elements)

        plt.tight_layout()
        fig.savefig(self.output_dir / 'seed_variance.png', dpi=300, bbox_inches='tight')
        print(f"Saved plot to {self.output_dir / 'seed_variance.png'}")
        plt.close()

    def plot_sensitivity_analysis(self, sensitivity_results: Dict):
        """
        Plot sensitivity analysis for timesteps and noise levels.
        Creates a 2x3 grid with timesteps (top) and noise (bottom) for GCN, GAT, and MLP.
        """
        n_models = len(sensitivity_results)
        fig, axes = plt.subplots(2, n_models, figsize=(6*n_models, 12))

        # Handle case of single model
        if n_models == 1:
            axes = axes.reshape(2, 1)

        for model_idx, (model_name, results) in enumerate(sensitivity_results.items()):
            # Timestep analysis (top row)
            if 'timesteps' in results:
                timesteps_data = results['timesteps']
                timesteps = [d['timesteps'] for d in timesteps_data]
                test_losses = [d['test_loss'] for d in timesteps_data]
                val_losses = [d['val_loss'] for d in timesteps_data]

                axes[0, model_idx].plot(timesteps, test_losses, 'o-', linewidth=2.5, markersize=8,
                                       label='Test Loss', color='steelblue')
                axes[0, model_idx].plot(timesteps, val_losses, 's-', linewidth=2.5, markersize=8,
                                       label='Val Loss', color='coral', alpha=0.7)
                axes[0, model_idx].axvline(50, color='red', linestyle='--', linewidth=2, alpha=0.6, label='Original (50)')
                axes[0, model_idx].set_xlabel('Number of Timesteps', fontsize=12, fontweight='bold')
                axes[0, model_idx].set_ylabel('Loss', fontsize=12, fontweight='bold')
                axes[0, model_idx].set_title(f'{model_name}: Timestep Sensitivity', fontsize=13, fontweight='bold')
                axes[0, model_idx].legend(fontsize=10, loc='best')
                axes[0, model_idx].grid(alpha=0.3)

            # Noise analysis (bottom row)
            if 'noise' in results:
                noise_data = results['noise']
                noise_levels = [d['noise_std'] for d in noise_data]
                test_losses_noise = [d['test_loss'] for d in noise_data]
                val_losses_noise = [d['val_loss'] for d in noise_data]

                axes[1, model_idx].plot(noise_levels, test_losses_noise, 'o-', linewidth=2.5, markersize=8,
                                       label='Test Loss', color='steelblue')
                axes[1, model_idx].plot(noise_levels, val_losses_noise, 's-', linewidth=2.5, markersize=8,
                                       label='Val Loss', color='coral', alpha=0.7)
                axes[1, model_idx].axvline(0.01, color='red', linestyle='--', linewidth=2, alpha=0.6, label='Original (0.01)')
                axes[1, model_idx].set_xlabel('Noise Std Dev', fontsize=12, fontweight='bold')
                axes[1, model_idx].set_ylabel('Loss', fontsize=12, fontweight='bold')
                axes[1, model_idx].set_title(f'{model_name}: Noise Sensitivity', fontsize=13, fontweight='bold')
                axes[1, model_idx].legend(fontsize=10, loc='best')
                axes[1, model_idx].grid(alpha=0.3)

        plt.suptitle('Data Sensitivity Analysis: Timesteps and Noise Level Impact',
                    fontsize=15, fontweight='bold', y=1.00)
        plt.tight_layout()
        fig.savefig(self.output_dir / 'sensitivity_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Saved plot to {self.output_dir / 'sensitivity_analysis.png'}")
        plt.close()

    def plot_individual_sensitivity_analysis(self, sensitivity_results: Dict):
        """
        Create individual sensitivity analysis plots for each model separately.
        Useful for detailed inspection of each model's behavior.
        """
        for model_name, results in sensitivity_results.items():
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # Timestep analysis
            if 'timesteps' in results:
                timesteps_data = results['timesteps']
                timesteps = [d['timesteps'] for d in timesteps_data]
                test_losses = [d['test_loss'] for d in timesteps_data]
                val_losses = [d['val_loss'] for d in timesteps_data]

                axes[0].plot(timesteps, test_losses, 'o-', linewidth=3, markersize=10,
                            label='Test Loss', color='steelblue')
                axes[0].plot(timesteps, val_losses, 's-', linewidth=3, markersize=10,
                            label='Val Loss', color='coral', alpha=0.7)
                axes[0].axvline(50, color='red', linestyle='--', linewidth=2.5, alpha=0.7, label='Original (50)')
                axes[0].fill_between(timesteps, test_losses, val_losses, alpha=0.1, color='gray')
                axes[0].set_xlabel('Number of Timesteps', fontsize=13, fontweight='bold')
                axes[0].set_ylabel('Loss', fontsize=13, fontweight='bold')
                axes[0].set_title(f'{model_name}: Timestep Sensitivity Analysis', fontsize=14, fontweight='bold')
                axes[0].legend(fontsize=11)
                axes[0].grid(alpha=0.3)

            # Noise analysis
            if 'noise' in results:
                noise_data = results['noise']
                noise_levels = [d['noise_std'] for d in noise_data]
                test_losses_noise = [d['test_loss'] for d in noise_data]
                val_losses_noise = [d['val_loss'] for d in noise_data]

                axes[1].plot(noise_levels, test_losses_noise, 'o-', linewidth=3, markersize=10,
                            label='Test Loss', color='steelblue')
                axes[1].plot(noise_levels, val_losses_noise, 's-', linewidth=3, markersize=10,
                            label='Val Loss', color='coral', alpha=0.7)
                axes[1].axvline(0.01, color='red', linestyle='--', linewidth=2.5, alpha=0.7, label='Original (0.01)')
                axes[1].fill_between(noise_levels, test_losses_noise, val_losses_noise, alpha=0.1, color='gray')
                axes[1].set_xlabel('Noise Std Dev', fontsize=13, fontweight='bold')
                axes[1].set_ylabel('Loss', fontsize=13, fontweight='bold')
                axes[1].set_title(f'{model_name}: Noise Sensitivity Analysis', fontsize=14, fontweight='bold')
                axes[1].legend(fontsize=11)
                axes[1].grid(alpha=0.3)

            plt.suptitle(f'{model_name} - Data Sensitivity Analysis', fontsize=15, fontweight='bold')
            plt.tight_layout()

            # Save individual model sensitivity plot
            fig.savefig(self.output_dir / f'{model_name.lower()}_sensitivity_detailed.png',
                       dpi=300, bbox_inches='tight')
            print(f"Saved detailed sensitivity plot to {self.output_dir / f'{model_name.lower()}_sensitivity_detailed.png'}")
            plt.close()

    def plot_statistical_summary_table(self, all_detailed_stats: Dict):
        """
        Create a detailed visual summary table showing all statistics per model.
        """
        fig, ax = plt.subplots(figsize=(18, 8))
        ax.axis('tight')
        ax.axis('off')

        # Build table data
        table_data = [['Model', 'Mean', 'Median', 'P25', 'P75', 'IQR', 'Std Dev', '95% CI', 'Min', 'Max', 'N Seeds']]

        for model_name, stats in all_detailed_stats.items():
            test_loss = stats['test_loss']
            row = [
                model_name,
                f"{test_loss['mean']:.4f}",
                f"{test_loss['median']:.4f}",
                f"{test_loss['p25']:.4f}",
                f"{test_loss['p75']:.4f}",
                f"{test_loss['iqr']:.4f}",
                f"{test_loss['std']:.4f}",
                f"±{test_loss['ci_95']:.4f}",
                f"{test_loss['min']:.4f}",
                f"{test_loss['max']:.4f}",
                f"{stats['n_seeds']}"
            ]
            table_data.append(row)

        # Create table
        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                        colWidths=[0.1, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.1, 0.08, 0.08, 0.08])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)

        # Format header
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Alternate row colors
        for i in range(1, len(table_data)):
            color = '#f0f0f0' if i % 2 == 0 else 'white'
            for j in range(len(table_data[0])):
                table[(i, j)].set_facecolor(color)

        plt.title('Multi-Seed Statistical Summary: Test Loss Metrics', fontsize=14, fontweight='bold', pad=20)
        fig.savefig(self.output_dir / 'statistical_summary_table.png', dpi=300, bbox_inches='tight')
        print(f"Saved plot to {self.output_dir / 'statistical_summary_table.png'}")
        plt.close()

    def plot_train_val_test_progression(self, all_detailed_stats: Dict):
        """
        Plot the progression of train, val, and test losses across seeds.
        """
        fig, axes = plt.subplots(1, len(all_detailed_stats), figsize=(5*len(all_detailed_stats), 6))
        if len(all_detailed_stats) == 1:
            axes = [axes]

        for idx, (model_name, stats) in enumerate(all_detailed_stats.items()):
            train_losses = stats['train_loss']['values']
            val_losses = stats['val_loss']['values']
            test_losses = stats['test_loss']['values']

            # Plot for each seed
            seeds = range(1, len(train_losses) + 1)
            axes[idx].scatter(seeds, train_losses, s=100, alpha=0.6, label='Train', marker='o', color='green')
            axes[idx].scatter(seeds, val_losses, s=100, alpha=0.6, label='Val', marker='s', color='orange')
            axes[idx].scatter(seeds, test_losses, s=100, alpha=0.6, label='Test', marker='^', color='red')

            # Add horizontal lines for means
            axes[idx].axhline(np.mean(train_losses), color='green', linestyle='--', alpha=0.3)
            axes[idx].axhline(np.mean(val_losses), color='orange', linestyle='--', alpha=0.3)
            axes[idx].axhline(np.mean(test_losses), color='red', linestyle='--', alpha=0.3)

            axes[idx].set_xlabel('Seed', fontsize=11)
            axes[idx].set_ylabel('Loss', fontsize=11)
            axes[idx].set_title(f'{model_name}', fontsize=12, fontweight='bold')
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)
            axes[idx].set_xticks(seeds)

        plt.suptitle('Train/Val/Test Loss Progression Across Seeds', fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        fig.savefig(self.output_dir / 'train_val_test_progression.png', dpi=300, bbox_inches='tight')
        print(f"Saved plot to {self.output_dir / 'train_val_test_progression.png'}")
        plt.close()

    def plot_pairwise_comparisons(self, pairwise_results: Dict):
        """
        Create a visual table showing pairwise comparison results.
        Shows p-values, rank-biserial effect sizes with bootstrap 95% CIs, and significance indicators.
        """
        fig, ax = plt.subplots(figsize=(20, 8))
        ax.axis('tight')
        ax.axis('off')

        # Build table data
        table_data = [['Comparison', 'Model A', 'Model B', 'Mean Loss A', 'Mean Loss B',
                       'Mean Diff', 'Wilcoxon p-value', 'Effect Size (r) [95% CI]', 'Significant?']]

        for comparison_label, results in pairwise_results.items():
            # Extract model names from comparison label
            parts = comparison_label.split(' vs ')
            model_a = parts[0]
            model_b = parts[1] if len(parts) > 1 else ''

            # Build effect size string with CI
            effect_size_str = results.get('rank_biserial_ci_str',
                                        f"{results['rank_biserial']:.4f}")

            row = [
                comparison_label,
                model_a,
                model_b,
                f"{results['mean_loss_a']:.4f}",
                f"{results['mean_loss_b']:.4f}",
                f"{results['mean_diff']:.4f}",
                f"{results['p_value']:.4f}",
                effect_size_str,
                f"{results['better_model']} *" if results['is_significant'] else 'No'
            ]
            table_data.append(row)

        # Create table
        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                        colWidths=[0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.15])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.8)

        # Format header
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Alternate row colors and highlight significant results
        for i in range(1, len(table_data)):
            color = '#f0f0f0' if i % 2 == 0 else 'white'
            # Highlight rows with significant results
            if i <= len(pairwise_results):
                results_list = list(pairwise_results.values())
                if results_list[i-1]['is_significant']:
                    color = '#ffffcc'  # Light yellow for significant

            for j in range(len(table_data[0])):
                table[(i, j)].set_facecolor(color)

        plt.title('Pairwise Comparisons: Wilcoxon Signed-Rank Test with Bootstrap 95% CIs\n(* indicates p < 0.05; Effect Size values with 95% CI from 10,000 bootstrap resamples)',
                  fontsize=13, fontweight='bold', pad=20)
        fig.savefig(self.output_dir / 'pairwise_comparisons.png', dpi=300, bbox_inches='tight')
        print(f"Saved plot to {self.output_dir / 'pairwise_comparisons.png'}")
        plt.close()

    def plot_motif_comparison(self, motif_metrics_all_models: Dict[str, Dict]):
        """
        Plot motif-specific MSE and MAE comparison across models.

        Args:
            motif_metrics_all_models: Dict with structure {model_name: {motif_label: {mean_mse, std_mse, mean_mae, std_mae}}}
        """
        motif_labels = set()
        for model_metrics in motif_metrics_all_models.values():
            motif_labels.update(model_metrics.keys())
        motif_labels = sorted([m for m in motif_labels if m != 'unknown'])

        if not motif_labels:
            print("Warning: No valid motif labels found for comparison")
            return

        models = list(motif_metrics_all_models.keys())
        n_motifs = len(motif_labels)

        # Create figure with 2 rows (MSE and MAE) and n_motifs columns
        fig, axes = plt.subplots(2, n_motifs, figsize=(5*n_motifs, 12))
        if n_motifs == 1:
            axes = axes.reshape(2, 1)

        # Extract metrics for plotting
        for motif_idx, motif in enumerate(motif_labels):
            mse_means = []
            mse_stds = []
            mae_means = []
            mae_stds = []

            for model in models:
                if motif in motif_metrics_all_models[model]:
                    metrics = motif_metrics_all_models[model][motif]
                    mse_means.append(metrics['mean_mse'])
                    mse_stds.append(metrics['std_mse'])
                    mae_means.append(metrics['mean_mae'])
                    mae_stds.append(metrics['std_mae'])
                else:
                    mse_means.append(0)
                    mse_stds.append(0)
                    mae_means.append(0)
                    mae_stds.append(0)

            # MSE plot (top row)
            x_pos = np.arange(len(models))
            axes[0, motif_idx].bar(x_pos, mse_means, yerr=mse_stds, capsize=5,
                                  alpha=0.8, color='steelblue', edgecolor='black', linewidth=1.5)
            axes[0, motif_idx].set_ylabel('Mean Squared Error (MSE)', fontsize=11, fontweight='bold')
            axes[0, motif_idx].set_title(f'{motif.replace("_", " ").title()} - MSE',
                                        fontsize=12, fontweight='bold')
            axes[0, motif_idx].set_xticks(x_pos)
            axes[0, motif_idx].set_xticklabels(models, rotation=45, ha='right')
            axes[0, motif_idx].grid(axis='y', alpha=0.3)

            # MAE plot (bottom row)
            axes[1, motif_idx].bar(x_pos, mae_means, yerr=mae_stds, capsize=5,
                                  alpha=0.8, color='coral', edgecolor='black', linewidth=1.5)
            axes[1, motif_idx].set_ylabel('Mean Absolute Error (MAE)', fontsize=11, fontweight='bold')
            axes[1, motif_idx].set_title(f'{motif.replace("_", " ").title()} - MAE',
                                        fontsize=12, fontweight='bold')
            axes[1, motif_idx].set_xticks(x_pos)
            axes[1, motif_idx].set_xticklabels(models, rotation=45, ha='right')
            axes[1, motif_idx].grid(axis='y', alpha=0.3)

        plt.suptitle('Motif-Specific Performance: MSE and MAE Comparison Across Models',
                    fontsize=15, fontweight='bold')
        plt.tight_layout()
        fig.savefig(self.output_dir / 'motif_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Saved plot to {self.output_dir / 'motif_comparison.png'}")
        plt.close()

    def plot_motif_heatmap(self, motif_metrics_all_models: Dict[str, Dict]):
        """
        Create heatmaps showing motif-specific performance across models.
        Separate heatmaps for MSE and MAE.
        """
        motif_labels = set()
        for model_metrics in motif_metrics_all_models.values():
            motif_labels.update(model_metrics.keys())
        motif_labels = sorted([m for m in motif_labels if m != 'unknown'])

        if not motif_labels:
            print("Warning: No valid motif labels found for heatmap")
            return

        models = list(motif_metrics_all_models.keys())

        # Build MSE matrix
        mse_matrix = np.zeros((len(models), len(motif_labels)))
        mae_matrix = np.zeros((len(models), len(motif_labels)))

        for model_idx, model in enumerate(models):
            for motif_idx, motif in enumerate(motif_labels):
                if motif in motif_metrics_all_models[model]:
                    mse_matrix[model_idx, motif_idx] = motif_metrics_all_models[model][motif]['mean_mse']
                    mae_matrix[model_idx, motif_idx] = motif_metrics_all_models[model][motif]['mean_mae']

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # MSE heatmap
        sns.heatmap(mse_matrix, annot=True, fmt='.4f', cmap='YlOrRd', ax=axes[0],
                   xticklabels=[m.replace('_', '\n') for m in motif_labels],
                   yticklabels=models, cbar_kws={'label': 'MSE'})
        axes[0].set_title('Mean Squared Error (MSE) by Motif and Model', fontsize=13, fontweight='bold')
        axes[0].set_xlabel('Network Motif', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Model', fontsize=12, fontweight='bold')

        # MAE heatmap
        sns.heatmap(mae_matrix, annot=True, fmt='.4f', cmap='YlGnBu', ax=axes[1],
                   xticklabels=[m.replace('_', '\n') for m in motif_labels],
                   yticklabels=models, cbar_kws={'label': 'MAE'})
        axes[1].set_title('Mean Absolute Error (MAE) by Motif and Model', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('Network Motif', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Model', fontsize=12, fontweight='bold')

        plt.suptitle('Motif-Specific Performance Heatmaps', fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        fig.savefig(self.output_dir / 'motif_heatmap.png', dpi=300, bbox_inches='tight')
        print(f"Saved plot to {self.output_dir / 'motif_heatmap.png'}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run benchmarking experiments")
    parser.add_argument('--seeds', type=int, default=20, help='Number of random seeds')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--sensitivity', action='store_true', help='Run sensitivity analysis')
    parser.add_argument('--output-dir', default='outputs/benchmark', help='Output directory')

    args = parser.parse_args()

    benchmark = BenchmarkExperiment(output_dir=args.output_dir)

    # Run multi-seed training for different models
    models_to_test = ['GCN', 'GAT', 'MLP', 'MeanMedian']

    all_results = {}
    all_detailed_stats = {}

    for model in models_to_test:
        print(f"\n{'='*60}")
        print(f"Training {model}")
        print(f"{'='*60}")

        try:
            results = benchmark.run_multi_seed_training(
                model_type=model,
                n_seeds=args.seeds,
                num_epochs=args.epochs,
                batch_size=args.batch_size
            )

            all_results[model] = results

            # Save individual model results
            benchmark.save_results(results, f"{model.lower()}_results.json")

            # Generate statistics and print table
            detailed_stats = benchmark.generate_detailed_statistics(results, model)
            all_detailed_stats[model] = detailed_stats

            results_df = benchmark.generate_results_table(results, model)
            print(f"\n{model} Results:")
            print(results_df.to_string(index=False))

        except Exception as e:
            print(f"Error training {model}: {e}")
            import traceback
            traceback.print_exc()

    # Generate comprehensive statistics table and save
    if all_detailed_stats:
        print(f"\n{'='*60}")
        print("Generating Comprehensive Statistical Summary")
        print(f"{'='*60}")

        # Save detailed statistics as JSON
        detailed_stats_dict = {
            model: {k: v for k, v in stats.items() if k != 'test_loss' or isinstance(v, dict)}
            for model, stats in all_detailed_stats.items()
        }
        benchmark.save_results(detailed_stats_dict, "detailed_statistics.json")

    # Aggregate motif metrics across seeds
    motif_metrics_all_models = {}
    for model, results in all_results.items():
        if 'motif_metrics' in results and results['motif_metrics']:
            # Average motif metrics across seeds
            aggregated_motif_metrics = {}
            for motif_label in results['motif_metrics'][0].keys():
                mses = [seed_metrics[motif_label]['mean_mse'] for seed_metrics in results['motif_metrics'] if motif_label in seed_metrics]
                maes = [seed_metrics[motif_label]['mean_mae'] for seed_metrics in results['motif_metrics'] if motif_label in seed_metrics]
                num_graphs = results['motif_metrics'][0][motif_label]['num_graphs']

                aggregated_motif_metrics[motif_label] = {
                    'num_graphs': num_graphs,
                    'mean_mse': float(np.mean(mses)),
                    'std_mse': float(np.std(mses)) if len(mses) > 1 else 0.0,
                    'mean_mae': float(np.mean(maes)),
                    'std_mae': float(np.std(maes)) if len(maes) > 1 else 0.0
                }
            motif_metrics_all_models[model] = aggregated_motif_metrics

    # Save aggregated motif metrics
    if motif_metrics_all_models:
        benchmark.save_results(motif_metrics_all_models, "motif_metrics_summary.json")

    # Run sensitivity analysis if requested
    sensitivity_results = {}
    if args.sensitivity:
        print(f"\n{'='*60}")
        print("Running Sensitivity Analysis")
        print(f"{'='*60}")

        for model in ['GCN', 'GAT', 'MLP']:
            try:
                results = benchmark.run_sensitivity_analysis(model_type=model)
                sensitivity_results[model] = results
                benchmark.save_results(results, f"{model.lower()}_sensitivity.json")
            except Exception as e:
                print(f"Error in sensitivity analysis for {model}: {e}")
                import traceback
                traceback.print_exc()

    # Generate visualizations
    if all_detailed_stats:
        print(f"\n{'='*60}")
        print("Generating Visualizations")
        print(f"{'='*60}")

        try:
            print("Generating baseline comparison plot...")
            benchmark.plot_baseline_comparison(all_detailed_stats)
        except Exception as e:
            print(f"Error generating baseline comparison plot: {e}")

        try:
            print("Generating seed variance plot...")
            benchmark.plot_seed_variance(all_detailed_stats)
        except Exception as e:
            print(f"Error generating seed variance plot: {e}")

        try:
            print("Generating statistical summary table...")
            benchmark.plot_statistical_summary_table(all_detailed_stats)
        except Exception as e:
            print(f"Error generating statistical summary table: {e}")

        try:
            print("Generating train/val/test progression plot...")
            benchmark.plot_train_val_test_progression(all_detailed_stats)
        except Exception as e:
            print(f"Error generating train/val/test progression plot: {e}")

        try:
            print("Computing pairwise comparisons (Wilcoxon signed-rank test)...")
            pairwise_comparisons = benchmark.compute_pairwise_comparisons(all_results)
            if pairwise_comparisons:
                benchmark.plot_pairwise_comparisons(pairwise_comparisons)
                # Save pairwise comparison results
                benchmark.save_results(pairwise_comparisons, "pairwise_comparisons.json")
        except Exception as e:
            print(f"Error in pairwise comparisons: {e}")
            import traceback
            traceback.print_exc()

    # Generate motif-specific comparison plots if available
    if motif_metrics_all_models:
        print(f"\n{'='*60}")
        print("Generating Motif-Specific Comparison Plots")
        print(f"{'='*60}")

        try:
            print("Generating motif comparison plot...")
            benchmark.plot_motif_comparison(motif_metrics_all_models)
        except Exception as e:
            print(f"Error generating motif comparison plot: {e}")

        try:
            print("Generating motif heatmap plot...")
            benchmark.plot_motif_heatmap(motif_metrics_all_models)
        except Exception as e:
            print(f"Error generating motif heatmap plot: {e}")

    # Generate sensitivity analysis plots if available
    if sensitivity_results:
        try:
            print("Generating combined sensitivity analysis plot...")
            benchmark.plot_sensitivity_analysis(sensitivity_results)
        except Exception as e:
            print(f"Error generating sensitivity analysis plot: {e}")

        try:
            print("Generating individual model sensitivity analysis plots...")
            benchmark.plot_individual_sensitivity_analysis(sensitivity_results)
        except Exception as e:
            print(f"Error generating individual sensitivity analysis plots: {e}")

    # Save summary
    benchmark.save_results(all_results, "multi_seed_summary.json")

    # Print final summary
    print(f"\n{'='*60}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*60}")
    print(f"All results saved to {benchmark.output_dir}")
    print("\nGenerated files:")
    print(f"\n  Visualizations:")
    print(f"    - Baseline comparison plot: baseline_comparison.png")
    print(f"    - Seed variance plot: seed_variance.png")
    print(f"    - Statistical summary table: statistical_summary_table.png")
    print(f"    - Train/Val/Test progression: train_val_test_progression.png")
    print(f"    - Pairwise comparisons (Wilcoxon): pairwise_comparisons.png")
    if motif_metrics_all_models:
        print(f"\n  Motif-Specific Analysis:")
        print(f"    - Motif comparison plot: motif_comparison.png")
        print(f"    - Motif heatmap: motif_heatmap.png")
    if sensitivity_results:
        print(f"\n  Sensitivity Analysis:")
        print(f"    - Combined sensitivity analysis: sensitivity_analysis.png")
        print(f"    - Individual GCN sensitivity: gcn_sensitivity_detailed.png")
        print(f"    - Individual GAT sensitivity: gat_sensitivity_detailed.png")
        print(f"    - Individual MLP sensitivity: mlp_sensitivity_detailed.png")
    print(f"\n  Data:")
    print(f"    - Detailed statistics: detailed_statistics.json")
    print(f"    - Pairwise comparisons: pairwise_comparisons.json")
    print(f"    - Individual model results: {{model}}_results.json")
    if motif_metrics_all_models:
        print(f"    - Motif metrics summary: motif_metrics_summary.json")
    if sensitivity_results:
        print(f"    - Sensitivity results: {{model}}_sensitivity.json")


if __name__ == "__main__":
    main()
