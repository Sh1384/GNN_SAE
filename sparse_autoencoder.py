"""
Sparse Autoencoder Module for GNN Layer2 Activation Analysis

Trains a sparse autoencoder on frozen GNN layer2 activations (64-dim) to discover
interpretable features in an expanded 512-dimensional latent space (8x expansion).

Architecture:
    - Input: 64-dimensional GNN layer2 activations
    - Latent: 512-dimensional sparse representation (TopK activation)
    - Output: 64-dimensional reconstruction

Sparsity:
    - TopK activation: Only top K neurons are kept active per sample
    - Rest are set to zero, enforcing exact sparsity

Loss:
    L = ||x - x_hat||_2^2  (reconstruction only, sparsity enforced by TopK)
"""

import json
import os
from pathlib import Path
from typing import Tuple, List, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder for discovering interpretable features in GNN activations.

    Expands 64-dim activations to 512-dim latent space with TopK sparsity.
    Only the top K neurons are kept active, rest are set to zero.
    """

    def __init__(self, input_dim: int = 64, latent_dim: int = 512, k: int = 32):
        """
        Initialize the sparse autoencoder.

        Args:
            input_dim: Dimension of input activations (64 for layer2)
            latent_dim: Dimension of latent representation (512)
            k: Number of top activations to keep (rest set to zero)
        """
        super(SparseAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.k = k

        # Encoder: 64 -> 512 with ReLU + TopK
        self.encoder = nn.Linear(input_dim, latent_dim)

        # Decoder: 512 -> 64
        self.decoder = nn.Linear(latent_dim, input_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier initialization."""
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation with TopK sparsity.

        Args:
            x: Input activations (batch_size, 64)

        Returns:
            Latent representation (batch_size, 512) with only top-k active
        """
        z = self.encoder(x)
        z = F.relu(z)  # Enforce non-negativity for interpretability

        # Apply TopK: keep only top k activations, set rest to zero
        if self.k < self.latent_dim:
            # Get top k values and indices for each sample
            topk_values, topk_indices = torch.topk(z, self.k, dim=1)

            # Create sparse tensor with only top k activations
            z_sparse = torch.zeros_like(z)
            z_sparse.scatter_(1, topk_indices, topk_values)

            return z_sparse
        else:
            return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to reconstruction.

        Args:
            z: Latent representation (batch_size, 512)

        Returns:
            Reconstructed activations (batch_size, 64)
        """
        x_hat = self.decoder(z)
        return x_hat

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through autoencoder.

        Args:
            x: Input activations (batch_size, 64)

        Returns:
            Tuple of (reconstructed_x, latent_z)
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def compute_loss(self, x: torch.Tensor, x_hat: torch.Tensor,
                     z: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute SAE loss (reconstruction only, sparsity enforced by TopK).

        Args:
            x: Original activations
            x_hat: Reconstructed activations
            z: Latent representation

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Reconstruction loss (MSE) - only loss term with TopK
        recon_loss = F.mse_loss(x_hat, x)

        # Total loss is just reconstruction (sparsity handled by TopK)
        total_loss = recon_loss

        # Track sparsity metrics for monitoring
        l0_sparsity = (z > 0).float().mean().item()  # Fraction of active neurons
        l1_sparsity = torch.mean(torch.abs(z)).item()  # Average absolute activation

        loss_dict = {
            'total': total_loss.item(),
            'reconstruction': recon_loss.item(),
            'sparsity': l1_sparsity,  # For monitoring only
            'l0_sparsity': l0_sparsity  # Should be approximately k/latent_dim
        }

        return total_loss, loss_dict


class ActivationDataset(Dataset):
    """
    Dataset for loading GNN layer2 activations from saved .pt files.

    Each graph's activations are stored as a tensor of shape [num_nodes, 64].
    This dataset flattens all node activations across all graphs into individual samples.
    """

    def __init__(self, activation_dir: Path):
        """
        Initialize dataset by loading all activations from directory.

        Args:
            activation_dir: Directory containing graph_*.pt files
        """
        self.activation_files = sorted(activation_dir.glob("graph_*.pt"))

        # Load all activations into memory
        print(f"Loading activations from {activation_dir}...")
        all_activations = []

        for act_file in tqdm(self.activation_files, desc="Loading"):
            activations = torch.load(act_file, weights_only=True)  # Shape: [num_nodes, 64]
            all_activations.append(activations)

        # Concatenate all node activations: [total_nodes, 64]
        self.activations = torch.cat(all_activations, dim=0)

        print(f"Loaded {len(self.activation_files)} graphs with {self.activations.shape[0]} total nodes")
        print(f"Activation shape: {self.activations.shape}")

    def __len__(self) -> int:
        return self.activations.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.activations[idx]


class SAETrainer:
    """
    Trainer for Sparse Autoencoder on GNN activations.
    """

    def __init__(self, model: SparseAutoencoder, device: str = 'cuda',
                 learning_rate: float = 1e-3):
        """
        Initialize SAE trainer.

        Args:
            model: Sparse autoencoder model
            device: Device to train on
            learning_rate: Learning rate for optimizer
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Training history
        self.history = {
            'train_loss': [],
            'train_recon': [],
            'train_sparsity': [],
            'train_l0': [],
            'val_loss': [],
            'val_recon': [],
            'val_sparsity': [],
            'val_l0': []
        }

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: DataLoader for training data

        Returns:
            Dictionary of average losses
        """
        self.model.train()

        epoch_losses = {
            'total': [],
            'reconstruction': [],
            'sparsity': [],
            'l0_sparsity': []
        }

        for batch in tqdm(train_loader, desc="Training", leave=False):
            batch = batch.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            x_hat, z = self.model(batch)

            # Compute loss
            loss, loss_dict = self.model.compute_loss(batch, x_hat, z)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Track losses
            for key in epoch_losses:
                epoch_losses[key].append(loss_dict[key])

        # Return average losses
        return {key: np.mean(values) for key, values in epoch_losses.items()}

    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate on validation/test data.

        Args:
            val_loader: DataLoader for validation data

        Returns:
            Dictionary of average losses
        """
        self.model.eval()

        epoch_losses = {
            'total': [],
            'reconstruction': [],
            'sparsity': [],
            'l0_sparsity': []
        }

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating", leave=False):
                batch = batch.to(self.device)

                # Forward pass
                x_hat, z = self.model(batch)

                # Compute loss
                _, loss_dict = self.model.compute_loss(batch, x_hat, z)

                # Track losses
                for key in epoch_losses:
                    epoch_losses[key].append(loss_dict[key])

        # Return average losses
        return {key: np.mean(values) for key, values in epoch_losses.items()}

    def save_model(self, path: str):
        """Save model checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        print(f"Model loaded from {path}")


def save_json(data: Dict, path: str):
    """Save dictionary to JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def _run_single_config(
    train_dataset: ActivationDataset,
    val_dataset: ActivationDataset,
    test_dataset: ActivationDataset,
    input_dim: int,
    latent_dim: int,
    k: int,
    device: str,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    seed: int,
) -> None:
    """
    Train and evaluate a single SAE configuration and save results.

    Saves checkpoints to checkpoints/sae_latent{latent_dim}_k{k}.pt and
    metrics to outputs/sae_metrics_latent{latent_dim}_k{k}.json.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    print("=" * 60)
    print(f"Sparse Autoencoder Training (latent_dim={latent_dim}, k={k})")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Architecture: {input_dim} -> {latent_dim} -> {input_dim}")
    print(f"Sparsity method: TopK (k={k}, {100*k/latent_dim:.1f}% active)")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print()

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=4)

    model = SparseAutoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        k=k
    )
    trainer = SAETrainer(model, device=device, learning_rate=learning_rate)

    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0

    ckpt_path = f"checkpoints/sae_latent{latent_dim}_k{k}.pt"

    print("Training Sparse Autoencoder...")
    print("-" * 60)

    for epoch in range(num_epochs):
        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.evaluate(val_loader)

        trainer.history['train_loss'].append(train_metrics['total'])
        trainer.history['train_recon'].append(train_metrics['reconstruction'])
        trainer.history['train_sparsity'].append(train_metrics['sparsity'])
        trainer.history['train_l0'].append(train_metrics['l0_sparsity'])

        trainer.history['val_loss'].append(val_metrics['total'])
        trainer.history['val_recon'].append(val_metrics['reconstruction'])
        trainer.history['val_sparsity'].append(val_metrics['sparsity'])
        trainer.history['val_l0'].append(val_metrics['l0_sparsity'])

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"[latent={latent_dim}, k={k}] Epoch {epoch+1}/{num_epochs}")
            print(f"  Train - Loss: {train_metrics['total']:.6f}, "
                  f"Recon: {train_metrics['reconstruction']:.6f}, "
                  f"L0: {train_metrics['l0_sparsity']:.3f}")
            print(f"  Val   - Loss: {val_metrics['total']:.6f}, "
                  f"Recon: {val_metrics['reconstruction']:.6f}, "
                  f"L0: {val_metrics['l0_sparsity']:.3f}")

        if val_metrics['total'] < best_val_loss:
            best_val_loss = val_metrics['total']
            patience_counter = 0
            trainer.save_model(ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    print("\nTraining complete!")
    print(f"Best validation loss (latent={latent_dim}, k={k}): {best_val_loss:.6f}")

    # Evaluate best checkpoint on test set
    trainer.load_model(ckpt_path)
    test_metrics = trainer.evaluate(test_loader)

    print(f"Test Loss: {test_metrics['total']:.6f}")
    print(f"Test Reconstruction: {test_metrics['reconstruction']:.6f}")
    print(f"Test L0 Sparsity: {test_metrics['l0_sparsity']:.3f}")

    metrics_path = f"outputs/sae_metrics_latent{latent_dim}_k{k}.json"
    final_metrics = {
        'best_val_loss': float(best_val_loss),
        'test_loss': float(test_metrics['total']),
        'test_reconstruction': float(test_metrics['reconstruction']),
        'test_sparsity': float(test_metrics['sparsity']),
        'test_l0_sparsity': float(test_metrics['l0_sparsity']),
        'train_history': trainer.history,
        'config': {
            'input_dim': input_dim,
            'latent_dim': latent_dim,
            'k': k,
            'sparsity_method': 'topk',
            'target_sparsity': k / latent_dim,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'seed': seed
        }
    }
    save_json(final_metrics, metrics_path)
    print(f"Metrics saved to {metrics_path}")
    print("=" * 60)


def main():
    """Main training pipeline for SAE with (latent_dim, k) sweep."""
    SEED = 42
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 1024
    NUM_EPOCHS = 200  # Increased from 100
    LEARNING_RATE = 5e-4

    INPUT_DIM = 64  # Layer 3 activations (updated from 80)

    latent_dims = [128, 256, 512]
    k_values = [4, 8, 16, 32]

    print("=" * 60)
    print("Sparse Autoencoder Training for GNN Layer3 Activations")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Input dim: {INPUT_DIM} (layer 3 hidden activations)")
    print(f"Latent dims to sweep: {latent_dims}")
    print(f"k values to sweep: {k_values}")
    print()

    print("Loading activation datasets...")
    train_dir = Path("outputs/activations/layer3_new/train")
    val_dir = Path("outputs/activations/layer3_new/val")
    test_dir = Path("outputs/activations/layer3_new/test")

    if not train_dir.exists():
        print(f"Error: {train_dir} not found. Please run gnn_train.py first.")
        return

    train_dataset = ActivationDataset(train_dir)
    val_dataset = ActivationDataset(val_dir)
    test_dataset = ActivationDataset(test_dir)

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)} node activations")
    print(f"  Val: {len(val_dataset)} node activations")
    print(f"  Test: {len(test_dataset)} node activations")
    print()

    for latent_dim in latent_dims:
        for k in k_values:
            if k > latent_dim:
                continue
            _run_single_config(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                input_dim=INPUT_DIM,
                latent_dim=latent_dim,
                k=k,
                device=DEVICE,
                batch_size=BATCH_SIZE,
                num_epochs=NUM_EPOCHS,
                learning_rate=LEARNING_RATE,
                seed=SEED,
            )


if __name__ == "__main__":
    main()
