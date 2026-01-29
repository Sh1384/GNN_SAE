"""
Sparse Autoencoder Module for GNN Layer2 Activation Analysis

Implements multiple SAE variants for discovering interpretable features in
GNN layer3 activations (64-dim expanded to latent space).

Variants:
  - TopK SAE: Standard approach with top-K sparsity
  - Gated SAE: Separates feature detection from magnitude estimation
  - JumpReLU SAE: Uses discontinuous activation with straight-through estimators
  - Switch SAE: Mixture of experts routing to expert SAEs
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ============================================================================
# Abstract Base Class
# ============================================================================

class BaseSAE(nn.Module, ABC):
    """
    Abstract base class for all SAE variants.

    Defines the common interface that all variants must implement.
    """

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse latent representation."""
        pass

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to reconstruction."""
        pass

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returns (reconstruction, latents)."""
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    @abstractmethod
    def compute_loss(self, x: torch.Tensor, x_hat: torch.Tensor,
                     z: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss and return (total_loss, loss_dict)."""
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return configuration dict for saving."""
        pass


# ============================================================================
# TopK SAE (Refactored from existing SparseAutoencoder)
# ============================================================================

class TopKSAE(BaseSAE):
    """
    TopK Sparse Autoencoder with fixed sparsity.

    Only the top K neurons are kept active, rest are set to zero.
    """

    def __init__(self, input_dim: int = 64, latent_dim: int = 512, k: int = 32):
        super().__init__(input_dim, latent_dim)
        self.k = k

        # Encoder: input_dim -> latent_dim with ReLU + TopK
        self.encoder = nn.Linear(input_dim, latent_dim)

        # Decoder: latent_dim -> input_dim
        self.decoder = nn.Linear(latent_dim, input_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier initialization."""
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode with ReLU + TopK sparsity."""
        z = self.encoder(x)
        z = F.relu(z)

        if self.k < self.latent_dim:
            topk_values, topk_indices = torch.topk(z, self.k, dim=1)
            z_sparse = torch.zeros_like(z)
            z_sparse.scatter_(1, topk_indices, topk_values)
            return z_sparse
        else:
            return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to reconstruction."""
        return self.decoder(z)

    def compute_loss(self, x: torch.Tensor, x_hat: torch.Tensor,
                     z: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute MSE reconstruction loss only."""
        recon_loss = F.mse_loss(x_hat, x)
        l0_sparsity = (z > 0).float().mean().item()
        l1_sparsity = torch.mean(torch.abs(z)).item()

        loss_dict = {
            'total': recon_loss.item(),
            'reconstruction': recon_loss.item(),
            'sparsity': l1_sparsity,
            'l0_sparsity': l0_sparsity
        }

        return recon_loss, loss_dict

    def get_config(self) -> Dict[str, Any]:
        """Return configuration."""
        return {
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'k': self.k,
            'sparsity_method': 'topk',
            'target_sparsity': self.k / self.latent_dim
        }


# ============================================================================
# Gated SAE
# ============================================================================

class GatedSAE(BaseSAE):
    """
    Gated Sparse Autoencoder that decouples feature detection from magnitude.

    Uses separate gating network (determines WHICH features) and magnitude
    network (determines HOW MUCH), solving shrinkage issues.
    """

    def __init__(self, input_dim: int = 64, latent_dim: int = 512,
                 sparsity_coef: float = 1e-3, aux_coef: float = 1/32):
        super().__init__(input_dim, latent_dim)

        # Gating network (feature detection)
        self.encoder_gate = nn.Linear(input_dim, latent_dim)

        # Magnitude network (strength estimation)
        self.encoder_mag = nn.Linear(input_dim, latent_dim)

        # Decoder
        self.decoder = nn.Linear(latent_dim, input_dim)

        self.sparsity_coef = sparsity_coef
        self.aux_coef = aux_coef

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier initialization."""
        nn.init.xavier_uniform_(self.encoder_gate.weight)
        nn.init.zeros_(self.encoder_gate.bias)
        nn.init.xavier_uniform_(self.encoder_mag.weight)
        nn.init.zeros_(self.encoder_mag.bias)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode with gated mechanism."""
        gate = F.relu(self.encoder_gate(x))
        mag = F.relu(self.encoder_mag(x))
        pi = (gate > 0).float()  # Heaviside (binary mask)
        z = pi * mag
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to reconstruction."""
        return self.decoder(z)

    def compute_loss(self, x: torch.Tensor, x_hat: torch.Tensor,
                     z: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute gated SAE loss: reconstruction + L1(gate) + auxiliary."""
        # Reconstruction loss
        recon_loss = F.mse_loss(x_hat, x)

        # Sparsity loss on gating network only
        gate = F.relu(self.encoder_gate(x))
        sparsity_loss = torch.mean(torch.abs(gate))

        # Auxiliary decoder loss (helps magnitude network)
        mag = F.relu(self.encoder_mag(x))
        aux_recon = self.decoder(mag)
        aux_loss = F.mse_loss(aux_recon, x)

        total_loss = (recon_loss +
                      self.sparsity_coef * sparsity_loss +
                      self.aux_coef * aux_loss)

        l0_sparsity = (z > 0).float().mean().item()

        loss_dict = {
            'total': total_loss.item(),
            'reconstruction': recon_loss.item(),
            'sparsity': sparsity_loss.item(),
            'auxiliary': aux_loss.item(),
            'l0_sparsity': l0_sparsity
        }

        return total_loss, loss_dict

    def get_config(self) -> Dict[str, Any]:
        """Return configuration."""
        return {
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'sparsity_coef': self.sparsity_coef,
            'aux_coef': self.aux_coef,
            'sparsity_method': 'gated',
            'target_sparsity': None  # Variable
        }


# ============================================================================
# JumpReLU SAE with Straight-Through Estimator
# ============================================================================

class JumpReLUFunction(torch.autograd.Function):
    """Custom autograd for JumpReLU with straight-through estimator."""

    @staticmethod
    def forward(ctx, z_pre, threshold, bandwidth):
        """Forward pass: discontinuous JumpReLU."""
        ctx.save_for_backward(z_pre, threshold)
        ctx.bandwidth = bandwidth
        return (z_pre > threshold).float() * z_pre

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass: smooth STE approximation."""
        z_pre, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth

        # Gaussian approximation of Dirac delta
        diff = z_pre - threshold
        ste_mask = torch.exp(-diff**2 / (2 * bandwidth**2))
        ste_mask = ste_mask / (bandwidth * np.sqrt(2 * np.pi))

        grad_z_pre = grad_output.clone()
        grad_threshold = -grad_output * ste_mask

        return grad_z_pre, grad_threshold, None


class JumpReLUSAE(BaseSAE):
    """
    JumpReLU Sparse Autoencoder using discontinuous activation.

    Uses straight-through estimators for backprop through discontinuous
    activation function. Directly optimizes L0 sparsity.
    """

    def __init__(self, input_dim: int = 64, latent_dim: int = 512,
                 threshold_init: float = 0.01, bandwidth: float = 0.01,
                 l0_coef: float = 1e-3):
        super().__init__(input_dim, latent_dim)

        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

        # Learnable per-feature thresholds
        self.threshold = nn.Parameter(torch.ones(latent_dim) * threshold_init)

        self.bandwidth = bandwidth
        self.l0_coef = l0_coef

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier initialization."""
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode with JumpReLU activation."""
        z_pre = self.encoder(x)
        # Use custom autograd function for STE
        z = JumpReLUFunction.apply(z_pre, self.threshold, self.bandwidth)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to reconstruction."""
        return self.decoder(z)

    def compute_loss(self, x: torch.Tensor, x_hat: torch.Tensor,
                     z: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute JumpReLU loss: reconstruction + L0 sparsity."""
        recon_loss = F.mse_loss(x_hat, x)

        # Direct L0 penalty
        l0_loss = (z > 0).float().mean()

        total_loss = recon_loss + self.l0_coef * l0_loss

        # L1 sparsity for monitoring
        l1_sparsity = torch.mean(torch.abs(z)).item()

        loss_dict = {
            'total': total_loss.item(),
            'reconstruction': recon_loss.item(),
            'sparsity': l1_sparsity,  # L1 sparsity for monitoring
            'l0_sparsity': l0_loss.item(),
            'threshold_mean': self.threshold.mean().item()
        }

        return total_loss, loss_dict

    def get_config(self) -> Dict[str, Any]:
        """Return configuration."""
        return {
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'threshold_init': float(self.threshold.mean().item()),
            'bandwidth': self.bandwidth,
            'l0_coef': self.l0_coef,
            'sparsity_method': 'jumprelu',
            'target_sparsity': None  # Variable
        }


# ============================================================================
# Switch SAE with Mixture of Experts
# ============================================================================

class ExpertSAE(nn.Module):
    """Individual expert SAE with TopK sparsity."""

    def __init__(self, input_dim: int, latent_dim: int, k: int):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)
        self.k = k
        self.latent_dim = latent_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode with TopK sparsity."""
        z = F.relu(self.encoder(x))
        if self.k < self.latent_dim:
            topk_values, topk_indices = torch.topk(z, self.k, dim=-1)
            z_sparse = torch.zeros_like(z)
            z_sparse.scatter_(-1, topk_indices, topk_values)
            return z_sparse
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent."""
        return self.decoder(z)


class SwitchSAE(BaseSAE):
    """
    Switch Sparse Autoencoder with mixture of experts routing.

    Routes inputs to expert SAEs using learned router network.
    More sample-efficient than single large SAE.
    """

    def __init__(self, input_dim: int = 64, num_experts: int = 8,
                 latent_per_expert: int = 128, k_per_expert: int = 8,
                 router_temp: float = 1.0):
        # Total latent capacity
        latent_dim = num_experts * latent_per_expert
        super().__init__(input_dim, latent_dim)

        # Router network
        self.router = nn.Linear(input_dim, num_experts)
        self.router_temp = router_temp

        # Experts
        self.num_experts = num_experts
        self.latent_per_expert = latent_per_expert
        self.experts = nn.ModuleList([
            ExpertSAE(input_dim, latent_per_expert, k_per_expert)
            for _ in range(num_experts)
        ])

        # Track expert usage for load balancing
        self.register_buffer('expert_usage', torch.zeros(num_experts))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode by routing to expert SAEs."""
        batch_size = x.shape[0]

        # Route each sample
        router_logits = self.router(x) / self.router_temp
        router_probs = F.softmax(router_logits, dim=-1)
        expert_indices = torch.argmax(router_probs, dim=-1)

        # Encode through selected experts
        z_global = torch.zeros(batch_size, self.latent_dim, device=x.device)

        for i in range(batch_size):
            expert_idx = expert_indices[i].item()
            z_expert = self.experts[expert_idx].encode(x[i:i+1])

            start_idx = expert_idx * self.latent_per_expert
            end_idx = start_idx + self.latent_per_expert
            z_global[i, start_idx:end_idx] = z_expert.squeeze(0)

            # Track usage
            if self.training:
                self.expert_usage[expert_idx] += 1

        return z_global

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode by routing through expert SAEs."""
        batch_size = z.shape[0]
        x_hat = torch.zeros(batch_size, self.input_dim, device=z.device)

        for i in range(batch_size):
            # Find which expert was used (non-zero block)
            for expert_idx in range(self.num_experts):
                start_idx = expert_idx * self.latent_per_expert
                end_idx = start_idx + self.latent_per_expert
                z_expert = z[i, start_idx:end_idx]

                if z_expert.abs().sum() > 0:
                    x_hat[i] = self.experts[expert_idx].decode(z_expert.unsqueeze(0)).squeeze(0)
                    break

        return x_hat

    def compute_loss(self, x: torch.Tensor, x_hat: torch.Tensor,
                     z: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute Switch SAE loss: reconstruction + load balancing."""
        recon_loss = F.mse_loss(x_hat, x)

        # Load balancing loss
        expert_usage_norm = self.expert_usage / (self.expert_usage.sum() + 1e-8)
        balance_loss = expert_usage_norm.var()

        total_loss = recon_loss + 0.01 * balance_loss

        # L1 sparsity for monitoring
        l1_sparsity = torch.mean(torch.abs(z)).item()
        l0_sparsity = (z > 0).float().mean().item()

        loss_dict = {
            'total': total_loss.item(),
            'reconstruction': recon_loss.item(),
            'sparsity': l1_sparsity,  # L1 sparsity for monitoring
            'balance': balance_loss.item(),
            'l0_sparsity': l0_sparsity
        }

        return total_loss, loss_dict

    def get_config(self) -> Dict[str, Any]:
        """Return configuration."""
        return {
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'num_experts': self.num_experts,
            'latent_per_expert': self.latent_per_expert,
            'router_temp': self.router_temp,
            'sparsity_method': 'switch',
            'target_sparsity': None  # Variable
        }


# ============================================================================
# Dataset and Trainer Classes
# ============================================================================

class ActivationDataset(Dataset):
    """
    Dataset for loading GNN layer3 activations from saved .pt files.
    """

    def __init__(self, activation_dir: Path):
        """Initialize dataset by loading all activations from directory."""
        self.activation_files = sorted(activation_dir.glob("graph_*.pt"))

        print(f"Loading activations from {activation_dir}...")
        all_activations = []

        for act_file in tqdm(self.activation_files, desc="Loading"):
            activations = torch.load(act_file, weights_only=True)
            all_activations.append(activations)

        self.activations = torch.cat(all_activations, dim=0)

        print(f"Loaded {len(self.activation_files)} graphs with {self.activations.shape[0]} total nodes")
        print(f"Activation shape: {self.activations.shape}")

    def __len__(self) -> int:
        return self.activations.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.activations[idx]


class SAETrainer:
    """Trainer for Sparse Autoencoder on GNN activations."""

    def __init__(self, model: BaseSAE, device: str = 'cuda',
                 learning_rate: float = 5e-4):
        """Initialize SAE trainer."""
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
        """Train for one epoch."""
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

        return {key: np.mean(values) for key, values in epoch_losses.items()}

    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate on validation/test data."""
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

        return {key: np.mean(values) for key, values in epoch_losses.items()}

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int = 100, patience: int = 15,
              checkpoint_path: str = None, verbose: bool = True) -> float:
        """
        Train the SAE model with early stopping.

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Maximum number of epochs to train
            patience: Early stopping patience (epochs without improvement)
            checkpoint_path: Path to save best model checkpoint
            verbose: Whether to print training progress

        Returns:
            best_val_loss: Best validation loss achieved
        """
        best_val_loss = float('inf')
        patience_counter = 0
        self.best_epoch = 0

        if verbose:
            print("Training...")
            print("-" * 70)

        for epoch in range(num_epochs):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            # Update training history
            self.history['train_loss'].append(train_metrics['total'])
            self.history['train_recon'].append(train_metrics['reconstruction'])
            self.history['train_sparsity'].append(train_metrics['sparsity'])
            self.history['train_l0'].append(train_metrics['l0_sparsity'])

            self.history['val_loss'].append(val_metrics['total'])
            self.history['val_recon'].append(val_metrics['reconstruction'])
            self.history['val_sparsity'].append(val_metrics['sparsity'])
            self.history['val_l0'].append(val_metrics['l0_sparsity'])

            # Verbose logging
            if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
                print(f"Epoch {epoch+1}/{num_epochs}")
                print(f"  Train - Loss: {train_metrics['total']:.6f}, "
                      f"Recon: {train_metrics['reconstruction']:.6f}, "
                      f"L0: {train_metrics['l0_sparsity']:.3f}")
                print(f"  Val   - Loss: {val_metrics['total']:.6f}, "
                      f"Recon: {val_metrics['reconstruction']:.6f}, "
                      f"L0: {val_metrics['l0_sparsity']:.3f}")

            # Early stopping check
            if val_metrics['total'] < best_val_loss:
                best_val_loss = val_metrics['total']
                patience_counter = 0
                self.best_epoch = epoch + 1

                # Save checkpoint if path provided
                if checkpoint_path is not None:
                    self.save_model(checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch+1}")
                    break

        if verbose:
            print("\nTraining complete!")
            print(f"Best validation loss: {best_val_loss:.6f}")
            print(f"Best epoch: {self.best_epoch}")

        return best_val_loss

    def save_model(self, path: str):
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint to

        Raises:
            IOError: If saving fails
        """
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'history': self.history,
                'config': self.model.get_config()
            }, path)
            print(f"✓ Model saved to {path}")
        except IOError as e:
            raise IOError(f"Failed to save model to {path}: {str(e)}")
        except Exception as e:
            raise IOError(f"Unexpected error saving model to {path}: {str(e)}")

    def load_model(self, path: str):
        """Load model checkpoint.

        Args:
            path: Path to checkpoint file

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            ValueError: If checkpoint is corrupted or incompatible
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        try:
            checkpoint = torch.load(path, weights_only=False)
        except Exception as e:
            raise ValueError(f"Failed to load checkpoint file {path}: {str(e)}")

        # Validate checkpoint structure
        required_keys = ['model_state_dict', 'optimizer_state_dict', 'history']
        missing_keys = [k for k in required_keys if k not in checkpoint]
        if missing_keys:
            raise ValueError(f"Checkpoint missing keys: {missing_keys}")

        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.history = checkpoint['history']
            print(f"✓ Model loaded from {path}")
        except RuntimeError as e:
            raise ValueError(f"Failed to load checkpoint state (shape mismatch?): {str(e)}")
        except Exception as e:
            raise ValueError(f"Unexpected error loading checkpoint: {str(e)}")


# ============================================================================
# Utility Functions and Training
# ============================================================================

def save_json(data: Dict, path: str):
    """Save dictionary to JSON file.

    Args:
        data: Dictionary to save
        path: Path to save to

    Raises:
        IOError: If file I/O fails
    """
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    except IOError as e:
        raise IOError(f"Failed to save JSON to {path}: {str(e)}")
    except Exception as e:
        raise IOError(f"Unexpected error saving JSON to {path}: {str(e)}")


def train_single_variant(model: BaseSAE, variant_name: str,
                         train_dataset: ActivationDataset,
                         val_dataset: ActivationDataset,
                         test_dataset: ActivationDataset,
                         device: str,
                         batch_size: int,
                         num_epochs: int,
                         learning_rate: float,
                         seed: int) -> None:
    """Train a single SAE variant and save results.

    Args:
        model: SAE model to train
        variant_name: Name for checkpoint/logging
        train_dataset: Training activation dataset
        val_dataset: Validation activation dataset
        test_dataset: Test activation dataset
        device: Device to train on ('cpu' or 'cuda')
        batch_size: Training batch size
        num_epochs: Maximum number of epochs
        learning_rate: Learning rate for optimizer
        seed: Random seed for reproducibility

    Raises:
        Exception: If training or checkpoint operations fail
    """
    try:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        print("=" * 70)
        print(f"Training: {variant_name} (seed={seed})")
        print("=" * 70)

        # Create data loaders
        try:
            train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                      shuffle=True, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                    shuffle=False, num_workers=4)
            test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                     shuffle=False, num_workers=4)
        except Exception as e:
            raise RuntimeError(f"Failed to create data loaders: {str(e)}")

        # Create trainer
        try:
            trainer = SAETrainer(model, device=device, learning_rate=learning_rate)
        except Exception as e:
            raise RuntimeError(f"Failed to create SAETrainer: {str(e)}")

        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0

        ckpt_path = f"checkpoints/sae_{variant_name}_seed{seed}.pt"

        print("Training...")
        print("-" * 70)

        for epoch in range(num_epochs):
            try:
                train_metrics = trainer.train_epoch(train_loader)
                val_metrics = trainer.evaluate(val_loader)
            except Exception as e:
                print(f"Error: Training epoch {epoch+1} failed: {str(e)}")
                raise RuntimeError(f"Training epoch {epoch+1} failed: {str(e)}")

            # Validate metrics
            try:
                trainer.history['train_loss'].append(train_metrics['total'])
                trainer.history['train_recon'].append(train_metrics['reconstruction'])
                trainer.history['train_sparsity'].append(train_metrics['sparsity'])
                trainer.history['train_l0'].append(train_metrics['l0_sparsity'])

                trainer.history['val_loss'].append(val_metrics['total'])
                trainer.history['val_recon'].append(val_metrics['reconstruction'])
                trainer.history['val_sparsity'].append(val_metrics['sparsity'])
                trainer.history['val_l0'].append(val_metrics['l0_sparsity'])

                # Check for NaN/Inf
                if not all(np.isfinite(v) for v in [
                    train_metrics['total'], val_metrics['total'],
                    train_metrics['l0_sparsity'], val_metrics['l0_sparsity']
                ]):
                    print(f"Error: Non-finite metrics detected at epoch {epoch+1}")
                    raise RuntimeError("Non-finite loss values detected during training")
            except Exception as e:
                raise RuntimeError(f"Failed to record metrics at epoch {epoch+1}: {str(e)}")

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{num_epochs}")
                print(f"  Train - Loss: {train_metrics['total']:.6f}, "
                      f"Recon: {train_metrics['reconstruction']:.6f}, "
                      f"L0: {train_metrics['l0_sparsity']:.3f}")
                print(f"  Val   - Loss: {val_metrics['total']:.6f}, "
                      f"Recon: {val_metrics['reconstruction']:.6f}, "
                      f"L0: {val_metrics['l0_sparsity']:.3f}")

            if val_metrics['total'] < best_val_loss:
                best_val_loss = val_metrics['total']
                patience_counter = 0
                try:
                    trainer.save_model(ckpt_path)
                except IOError as e:
                    print(f"Warning: Failed to save checkpoint: {str(e)}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break

        print("\nTraining complete!")
        print(f"Best validation loss: {best_val_loss:.6f}")

        # Test evaluation
        try:
            trainer.load_model(ckpt_path)
            test_metrics = trainer.evaluate(test_loader)

            print(f"Test Loss: {test_metrics['total']:.6f}")
            print(f"Test Reconstruction: {test_metrics['reconstruction']:.6f}")
            print(f"Test L0 Sparsity: {test_metrics['l0_sparsity']:.3f}")

            # Save metrics
            config = model.get_config()
            config['variant_name'] = variant_name
            config['seed'] = seed

            metrics_path = f"outputs/sae_metrics_{variant_name}_seed{seed}.json"
            final_metrics = {
                'best_val_loss': float(best_val_loss),
                'test_metrics': {k: float(v) for k, v in test_metrics.items()},
                'train_history': trainer.history,
                'config': config
            }
            save_json(final_metrics, metrics_path)
            print(f"Metrics saved to {metrics_path}")
        except Exception as e:
            print(f"Error: Test evaluation or metrics saving failed: {str(e)}")
            raise RuntimeError(f"Test evaluation failed: {str(e)}")
    except Exception as e:
        print(f"Error: Training failed: {str(e)}")
        raise
    finally:
        print("=" * 70)


def main():
    """Main training pipeline for all SAE variants."""
    SEED = 42
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 1024
    NUM_EPOCHS = 200
    LEARNING_RATE = 5e-4
    INPUT_DIM = 64

    print("=" * 70)
    print("Sparse Autoencoder Variant Training")
    print("=" * 70)
    print(f"Device: {DEVICE}")
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

    # ========== TOPK SAE VARIANTS ==========
    print("\n" + "="*70)
    print("TRAINING TOPK SAE VARIANTS")
    print("="*70)

    topk_configs = [
        {'latent_dim': 128, 'k': 4},
        {'latent_dim': 128, 'k': 8},
        {'latent_dim': 128, 'k': 16},
        {'latent_dim': 256, 'k': 4},
        {'latent_dim': 256, 'k': 8},
        {'latent_dim': 256, 'k': 16},
        {'latent_dim': 256, 'k': 32},
        {'latent_dim': 512, 'k': 4},
        {'latent_dim': 512, 'k': 8},
        {'latent_dim': 512, 'k': 16},
        {'latent_dim': 512, 'k': 32},
    ]

    for config in topk_configs:
        model = TopKSAE(input_dim=INPUT_DIM, **config)
        variant_name = f"topk_latent{config['latent_dim']}_k{config['k']}"
        train_single_variant(model, variant_name, train_dataset, val_dataset,
                           test_dataset, DEVICE, BATCH_SIZE, NUM_EPOCHS,
                           LEARNING_RATE, SEED)

    # ========== GATED SAE VARIANTS ==========
    print("\n" + "="*70)
    print("TRAINING GATED SAE VARIANTS")
    print("="*70)

    gated_configs = [
        {'latent_dim': 128, 'sparsity_coef': 1e-4},
        {'latent_dim': 128, 'sparsity_coef': 5e-4},
        {'latent_dim': 128, 'sparsity_coef': 1e-3},
        {'latent_dim': 256, 'sparsity_coef': 1e-4},
        {'latent_dim': 256, 'sparsity_coef': 5e-4},
        {'latent_dim': 256, 'sparsity_coef': 1e-3},
        {'latent_dim': 512, 'sparsity_coef': 1e-4},
        {'latent_dim': 512, 'sparsity_coef': 5e-4},
        {'latent_dim': 512, 'sparsity_coef': 1e-3},
    ]

    for config in gated_configs:
        model = GatedSAE(input_dim=INPUT_DIM, **config)
        coef_str = f"{config['sparsity_coef']:.0e}"
        variant_name = f"gated_latent{config['latent_dim']}_lambda{coef_str}"
        train_single_variant(model, variant_name, train_dataset, val_dataset,
                           test_dataset, DEVICE, BATCH_SIZE, NUM_EPOCHS,
                           LEARNING_RATE, SEED)

    # ========== JUMPRELU SAE VARIANTS ==========
    print("\n" + "="*70)
    print("TRAINING JUMPRELU SAE VARIANTS")
    print("="*70)

    jumprelu_configs = [
        {'latent_dim': 128, 'threshold_init': 0.01, 'bandwidth': 0.01},
        {'latent_dim': 128, 'threshold_init': 0.1, 'bandwidth': 0.01},
        {'latent_dim': 256, 'threshold_init': 0.01, 'bandwidth': 0.01},
        {'latent_dim': 256, 'threshold_init': 0.1, 'bandwidth': 0.01},
        {'latent_dim': 512, 'threshold_init': 0.01, 'bandwidth': 0.01},
        {'latent_dim': 512, 'threshold_init': 0.1, 'bandwidth': 0.01},
    ]

    for config in jumprelu_configs:
        model = JumpReLUSAE(input_dim=INPUT_DIM, **config)
        thresh_str = f"{config['threshold_init']:.0e}"
        bw_str = f"{config['bandwidth']:.0e}"
        variant_name = f"jumprelu_latent{config['latent_dim']}_thresh{thresh_str}_bw{bw_str}"
        train_single_variant(model, variant_name, train_dataset, val_dataset,
                           test_dataset, DEVICE, BATCH_SIZE, NUM_EPOCHS,
                           LEARNING_RATE, SEED)

    # ========== SWITCH SAE VARIANTS ==========
    print("\n" + "="*70)
    print("TRAINING SWITCH SAE VARIANTS")
    print("="*70)

    switch_configs = [
        {'num_experts': 4, 'latent_per_expert': 64, 'k_per_expert': 8},
        {'num_experts': 4, 'latent_per_expert': 128, 'k_per_expert': 16},
        {'num_experts': 8, 'latent_per_expert': 64, 'k_per_expert': 8},
        {'num_experts': 8, 'latent_per_expert': 128, 'k_per_expert': 16},
    ]

    for config in switch_configs:
        model = SwitchSAE(input_dim=INPUT_DIM, **config)
        total_latent = config['num_experts'] * config['latent_per_expert']
        variant_name = f"switch_experts{config['num_experts']}_latent{total_latent}_k{config['k_per_expert']}"
        train_single_variant(model, variant_name, train_dataset, val_dataset,
                           test_dataset, DEVICE, BATCH_SIZE, NUM_EPOCHS,
                           LEARNING_RATE, SEED)

    print("\n" + "="*70)
    print("ALL TRAINING COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
