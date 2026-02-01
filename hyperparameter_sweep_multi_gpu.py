"""
Multi-GPU Hyperparameter Sweep for GCN Training

Runs Optuna hyperparameter optimization in parallel across multiple GPUs.
Each trial is assigned to a different GPU for maximum throughput.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict
import numpy as np
import torch
from torch.utils.data import DataLoader
import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler
from joblib import Parallel, delayed

# Import from gnn_train
from gnn_train import (
    GraphDataset,
    GCNModel,
    GNNTrainer,
    collate_fn,
    load_all_graphs,
    split_data,
)


def get_gpu_id(trial_number: int, num_gpus: int) -> int:
    """Assign trial to a GPU in round-robin fashion."""
    return trial_number % num_gpus


def objective_single_gpu(
    trial: Trial,
    train_paths,
    val_paths,
    test_paths,
    gpu_id: int,
    mask_prob: float,
    batch_size: int,
    num_epochs: int,
    seed: int,
) -> float:
    """
    Objective function for a single trial on a specific GPU.

    Args:
        trial: Optuna trial object
        train_paths: Training graph paths
        val_paths: Validation graph paths
        test_paths: Test graph paths
        gpu_id: GPU ID to use
        mask_prob: Masking probability
        batch_size: Batch size
        num_epochs: Maximum number of epochs
        seed: Random seed

    Returns:
        Best validation loss
    """
    device = f'cuda:{gpu_id}'

    # Suggest hyperparameters
    hidden_dim = trial.suggest_int('hidden_dim', 16, 256, step=8)
    dropout = trial.suggest_float('dropout', 0.0, 0.5, step=0.05)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)

    # Create datasets
    train_dataset = GraphDataset(train_paths, mask_prob=mask_prob, seed=seed)
    val_dataset = GraphDataset(val_paths, mask_prob=mask_prob, seed=seed + 1)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )

    # Create model and trainer
    model = GCNModel(input_dim=2, hidden_dim=hidden_dim, output_dim=1, dropout=dropout)
    trainer = GNNTrainer(model, device=device, learning_rate=learning_rate, seed=seed)

    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 25

    for epoch in range(num_epochs):
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.validate(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break

    return best_val_loss


def run_trial(
    trial_number: int,
    study,
    train_paths,
    val_paths,
    test_paths,
    num_gpus: int,
    mask_prob: float,
    batch_size: int,
    num_epochs: int,
    seed: int,
) -> Dict:
    """
    Run a single trial and return results.

    Args:
        trial_number: Trial number
        study: Optuna study object
        train_paths: Training graph paths
        val_paths: Validation graph paths
        test_paths: Test graph paths
        num_gpus: Number of GPUs available
        mask_prob: Masking probability
        batch_size: Batch size
        num_epochs: Maximum number of epochs
        seed: Random seed

    Returns:
        Dictionary with trial results
    """
    gpu_id = get_gpu_id(trial_number, num_gpus)

    trial = study.ask()

    try:
        value = objective_single_gpu(
            trial,
            train_paths,
            val_paths,
            test_paths,
            gpu_id,
            mask_prob,
            batch_size,
            num_epochs,
            seed,
        )

        study.tell(trial, value)

        return {
            'trial_number': trial_number,
            'gpu_id': gpu_id,
            'value': value,
            'params': trial.params,
            'status': 'COMPLETE'
        }
    except Exception as e:
        study.tell(trial, state=optuna.trial.TrialState.FAIL)
        return {
            'trial_number': trial_number,
            'gpu_id': gpu_id,
            'status': 'FAILED',
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description='Multi-GPU Hyperparameter Sweep for GCN')
    parser.add_argument('--num_trials', type=int, default=20, help='Number of trials')
    parser.add_argument('--num_epochs', type=int, default=100, help='Max epochs per trial')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--mask_prob', type=float, default=0.2, help='Masking probability')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_gpus', type=int, default=4, help='Number of GPUs to use')
    parser.add_argument('--n_jobs', type=int, default=4, help='Number of parallel jobs')
    parser.add_argument('--output_dir', type=str, default='outputs/hyperparameter_sweep_multi_gpu',
                        help='Output directory')
    args = parser.parse_args()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MULTI-GPU GCN HYPERPARAMETER SWEEP WITH OPTUNA")
    print("=" * 80)
    print(f"Number of trials: {args.num_trials}")
    print(f"Max epochs per trial: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of GPUs: {args.num_gpus}")
    print(f"Parallel jobs: {args.n_jobs}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)

    # Check GPU availability
    if not torch.cuda.is_available():
        print("ERROR: No CUDA devices available!")
        return

    available_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {available_gpus}")

    if available_gpus < args.num_gpus:
        print(f"Warning: Requested {args.num_gpus} GPUs but only {available_gpus} available")
        args.num_gpus = available_gpus

    for i in range(args.num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print()

    # Load and split data
    print("Loading data...")
    all_graph_paths = load_all_graphs(single_motif_only=True)
    print(f"Found {len(all_graph_paths)} graphs")

    if len(all_graph_paths) == 0:
        print("Error: No graphs found. Please run graph_motif_generator.py first.")
        return

    train_paths, val_paths, test_paths = split_data(
        all_graph_paths,
        seed=args.seed,
        stratify_by_motif=True,
        equal_counts_per_motif=False
    )
    print(f"Split: {len(train_paths)} train, {len(val_paths)} val, {len(test_paths)} test\n")

    # Create Optuna study
    sampler = TPESampler(seed=args.seed, n_startup_trials=5)
    study = optuna.create_study(
        direction='minimize',
        sampler=sampler,
        study_name='gcn_multi_gpu_optimization'
    )

    print(f"Starting hyperparameter sweep with {args.n_jobs} parallel workers...\n")

    # Run trials in parallel across GPUs
    results = Parallel(n_jobs=args.n_jobs, backend='threading')(
        delayed(run_trial)(
            trial_number=i,
            study=study,
            train_paths=train_paths,
            val_paths=val_paths,
            test_paths=test_paths,
            num_gpus=args.num_gpus,
            mask_prob=args.mask_prob,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            seed=args.seed,
        )
        for i in range(args.num_trials)
    )

    # Print results
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)

    completed_trials = [r for r in results if r['status'] == 'COMPLETE']
    failed_trials = [r for r in results if r['status'] == 'FAILED']

    print(f"\nCompleted trials: {len(completed_trials)}")
    print(f"Failed trials: {len(failed_trials)}")

    if completed_trials:
        best_trial = min(completed_trials, key=lambda x: x['value'])

        print(f"\nBest Trial: {best_trial['trial_number']}")
        print(f"Best Validation Loss: {best_trial['value']:.6f}")
        print(f"GPU used: {best_trial['gpu_id']}")
        print("\nBest Hyperparameters:")
        for key, value in best_trial['params'].items():
            print(f"  {key}: {value}")

        # Save results
        print(f"\nSaving results to {args.output_dir}...")

        with open(output_path / "best_params.json", 'w') as f:
            json.dump(best_trial['params'], f, indent=2)

        with open(output_path / "study_info.json", 'w') as f:
            json.dump({
                'best_value': best_trial['value'],
                'best_trial': best_trial['trial_number'],
                'n_trials': args.num_trials,
                'n_complete_trials': len(completed_trials),
                'n_failed_trials': len(failed_trials)
            }, f, indent=2)

        trials_df = study.trials_dataframe()
        trials_df.to_csv(output_path / "trials.csv", index=False)

        print(f"✓ Saved results to {output_path}")
    else:
        print("\nNo trials completed successfully!")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
