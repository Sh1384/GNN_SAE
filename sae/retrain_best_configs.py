#!/usr/bin/env python3
"""
Retrain Best SAE Configurations with Multiple Seeds

Automatically identifies the best configuration per variant from compare_sae_configs.py
and retrains them with multiple seeds for stability analysis.

This addresses Phase 2 of the multi-seed training workflow from IMPLEMENTATION_SUMMARY.md

Usage:
    python retrain_best_configs.py                          # Use default seeds: 42,123,456,789,1011
    python retrain_best_configs.py --seeds 42 123 456       # Custom seeds
    python retrain_best_configs.py --variant topk            # Single variant
    python retrain_best_configs.py --num-seeds 5             # Generate 5 random seeds

Output:
    - checkpoints/sae_{variant}_{params}_seed{seed}.pt      (multiple checkpoints per config)
    - outputs/sae_metrics_{variant}_{params}_seed{seed}.json (multiple metrics per config)
    - outputs/retrain_summary.json                           (best config summary with stability stats)

Requirements:
    1. Run compare_sae_configs.py first to identify best configs
    2. Phase 1 training (sparse_autoencoder.py) must be completed
    3. GNN activation data must be available

Example Workflow:
    # Phase 1: Train all configs with seed=42
    python sparse_autoencoder.py

    # Phase 2: Identify best config per variant
    python compare_sae_configs.py

    # Phase 3: Retrain best configs with 5 seeds
    python retrain_best_configs.py --seeds 42 123 456 789 1011

    # Phase 4: Analyze feature stability across seeds
    python statistical_analysis_suite.py --seed-analysis
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import pandas as pd
import time

# Import SAE classes and utilities from sparse_autoencoder
from sparse_autoencoder import (
    BaseSAE, TopKSAE, GatedSAE, JumpReLUSAE, SwitchSAE,
    ActivationDataset, SAETrainer
)


def identify_best_configs() -> Dict[str, Dict]:
    """
    Read compare_sae_configs.py output and identify best config per variant.

    Looks for outputs/sae_variant_comparison.csv and identifies the highest-scoring
    configuration for each variant (topk, gated, jumprelu, switch).

    Returns:
        Dict mapping variant name to its best configuration dict
        Format: {
            'topk': {'variant': 'topk', 'config_name': 'x', 'latent_dim': 512, 'k': 8, ...},
            'gated': {...},
            ...
        }

    Raises:
        FileNotFoundError: If comparison CSV not found
    """
    comparison_file = Path('outputs/sae_variant_comparison.csv')

    if not comparison_file.exists():
        raise FileNotFoundError(
            f"\n❌ ERROR: {comparison_file} not found!\n"
            f"   Please run 'python compare_sae_configs.py' first to identify best configs\n"
            f"   Or run the complete Colab pipeline"
        )

    print(f"📋 Reading comparison results from {comparison_file}")

    df = pd.read_csv(comparison_file)
    best_configs = {}

    for variant in ['topk', 'gated', 'jumprelu', 'switch']:
        variant_df = df[df['variant'] == variant]

        if len(variant_df) == 0:
            print(f"⚠️  No configs found for {variant}")
            continue

        # Get best by composite_score if available, else by test_mse
        if 'composite_score' in variant_df.columns:
            best_idx = variant_df['composite_score'].idxmax()
            score_col = 'composite_score'
        elif 'test_mse' in variant_df.columns:
            best_idx = variant_df['test_mse'].idxmin()
            score_col = 'test_mse'
        else:
            print(f"⚠️  Could not find scoring column for {variant}")
            continue

        best_row = df.loc[best_idx]

        # Extract config from the CSV row
        config = {
            'variant': variant,
            'config_name': best_row.get('config_name', 'unknown'),
        }

        # Add variant-specific parameters
        if variant == 'topk':
            config['latent_dim'] = int(best_row.get('latent_dim', 512))
            config['k'] = int(best_row.get('k', 8))

        elif variant == 'gated':
            config['latent_dim'] = int(best_row.get('latent_dim', 512))
            config['sparsity_coef'] = float(best_row.get('sparsity_coef', 1e-3))

        elif variant == 'jumprelu':
            config['latent_dim'] = int(best_row.get('latent_dim', 512))
            config['threshold_init'] = float(best_row.get('threshold_init', 0.01))
            config['bandwidth'] = 0.01

        elif variant == 'switch':
            config['num_experts'] = int(best_row.get('num_experts', 8))
            config['latent_per_expert'] = int(best_row.get('latent_per_expert', 64))
            config['k_per_expert'] = int(best_row.get('k_per_expert', 8))

        best_configs[variant] = config

        score_val = best_row.get(score_col, 'N/A')
        print(f"✓ {variant.upper():10} | Config: {config['config_name']:30} | {score_col}: {score_val}")

    return best_configs


def create_model(
    variant: str,
    config: Dict,
    input_dim: int = 64
) -> BaseSAE:
    """
    Factory function to create SAE model instance based on variant.

    Args:
        variant: One of 'topk', 'gated', 'jumprelu', 'switch'
        config: Configuration dict with hyperparameters
        input_dim: Input dimension (default: 64 for GNN layer2)

    Returns:
        Instantiated SAE model (subclass of BaseSAE)

    Raises:
        ValueError: If variant is unknown
    """
    if variant == 'topk':
        return TopKSAE(
            input_dim=input_dim,
            latent_dim=config['latent_dim'],
            k=config['k']
        )

    elif variant == 'gated':
        return GatedSAE(
            input_dim=input_dim,
            latent_dim=config['latent_dim'],
            sparsity_coef=config['sparsity_coef']
        )

    elif variant == 'jumprelu':
        return JumpReLUSAE(
            input_dim=input_dim,
            latent_dim=config['latent_dim'],
            threshold_init=config['threshold_init'],
            bandwidth=config.get('bandwidth', 0.01)
        )

    elif variant == 'switch':
        return SwitchSAE(
            input_dim=input_dim,
            num_experts=config['num_experts'],
            latent_per_expert=config['latent_per_expert'],
            k_per_expert=config['k_per_expert']
        )

    else:
        raise ValueError(f"Unknown variant: {variant}")


def get_variant_name(variant: str, config: Dict) -> str:
    """
    Generate consistent checkpoint name for a variant config.

    Args:
        variant: Variant name
        config: Configuration dict

    Returns:
        Checkpoint name string (without seed)
    """
    if variant == 'topk':
        return f"topk_latent{config['latent_dim']}_k{config['k']}"

    elif variant == 'gated':
        coef_str = f"{config['sparsity_coef']:.0e}"
        return f"gated_latent{config['latent_dim']}_lambda{coef_str}"

    elif variant == 'jumprelu':
        thresh_str = f"{config['threshold_init']:.0e}"
        bw_str = f"{config.get('bandwidth', 0.01):.0e}"
        return f"jumprelu_latent{config['latent_dim']}_thresh{thresh_str}_bw{bw_str}"

    elif variant == 'switch':
        total_latent = config['num_experts'] * config['latent_per_expert']
        return f"switch_experts{config['num_experts']}_latent{total_latent}_k{config['k_per_expert']}"

    else:
        raise ValueError(f"Unknown variant: {variant}")


def retrain_config(
    variant: str,
    config: Dict,
    train_dataset: ActivationDataset,
    val_dataset: ActivationDataset,
    test_dataset: ActivationDataset,
    seeds: List[int],
    device: str = 'cuda',
    batch_size: int = 1024,
    num_epochs: int = 200,
    learning_rate: float = 5e-4
) -> List[Dict]:
    """
    Retrain a single configuration with multiple seeds.

    Args:
        variant: SAE variant name ('topk', 'gated', 'jumprelu', 'switch')
        config: Configuration dictionary with hyperparameters
        train_dataset: Training dataset (ActivationDataset)
        val_dataset: Validation dataset
        test_dataset: Test dataset
        seeds: List of seeds to use for retraining
        device: Device to use ('cuda' or 'cpu')
        batch_size: Batch size for training
        num_epochs: Maximum number of epochs
        learning_rate: Learning rate

    Returns:
        List of metrics dicts, one per seed:
        [
            {'seed': 42, 'test_mse': 0.00123, 'best_epoch': 150, ...},
            {'seed': 123, 'test_mse': 0.00125, 'best_epoch': 145, ...},
            ...
        ]
    """
    INPUT_DIM = 64
    results = []
    variant_name = get_variant_name(variant, config)

    print(f"\n{'='*80}")
    print(f"RETRAINING: {variant.upper():10} | Config: {config['config_name']:30}")
    print(f"{'='*80}")
    print(f"Checkpoint template: sae_{variant_name}_seed{{seed}}.pt")
    print(f"Number of seeds:     {len(seeds)}")
    print(f"Seeds:               {seeds}")
    print(f"{'='*80}\n")

    for seed_idx, seed in enumerate(seeds, 1):
        print(f"[{seed_idx}/{len(seeds)}] Seed {seed:5d}...", end=' ', flush=True)
        start_time = time.time()

        try:
            # Set random seeds for reproducibility
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

            # Create fresh model instance
            model = create_model(variant, config, INPUT_DIM)
            model = model.to(device)

            # Create trainer
            trainer = SAETrainer(
                model=model,
                device=device,
                learning_rate=learning_rate
            )

            # Create data loaders
            from torch.utils.data import DataLoader
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

            # Train with early stopping
            ckpt_path = f"checkpoints/sae_{variant_name}_seed{seed}.pt"
            trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=num_epochs,
                patience=15,
                checkpoint_path=ckpt_path,
                verbose=True
            )

            # Evaluate on test set
            test_metrics = trainer.evaluate(test_loader)
            test_mse = test_metrics['total']

            # Checkpoint already saved during training (no need to save again)
            Path('checkpoints').mkdir(exist_ok=True)

            # Save metrics
            Path('outputs').mkdir(exist_ok=True)
            metrics = {
                'variant': variant,
                'config_name': config['config_name'],
                'seed': seed,
                'best_epoch': trainer.best_epoch,
                'total_epochs': num_epochs,
                'test_mse': float(test_mse),
                'checkpoint_path': ckpt_path,
                'config': {k: v for k, v in config.items() if k not in ['config_name']},
            }

            metrics_path = f"outputs/sae_metrics_{variant_name}_seed{seed}.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)

            results.append(metrics)

            elapsed = time.time() - start_time
            print(f"✓ MSE: {test_mse:.6f} | Epoch: {trainer.best_epoch:3d} | Time: {elapsed:6.1f}s")

        except Exception as e:
            print(f"❌ Error: {str(e)[:50]}")
            continue

    return results


def save_retrain_summary(all_results: Dict[str, List[Dict]], seeds: List[int]) -> None:
    """
    Save summary statistics of retraining across all variants.

    Creates outputs/retrain_summary.json with per-variant statistics:
    - Mean/std test MSE across seeds
    - Min/max test MSE
    - Which seeds were used
    - Configuration details

    Args:
        all_results: Dict mapping variant -> list of metrics dicts
        seeds: List of seeds used for retraining
    """
    summary = {}

    for variant, results in all_results.items():
        if len(results) == 0:
            continue

        mses = [r['test_mse'] for r in results]
        epochs = [r['best_epoch'] for r in results]

        summary[variant] = {
            'num_seeds_trained': len(results),
            'seeds': seeds,
            'config_name': results[0]['config_name'],
            'config': results[0]['config'],
            'test_mse': {
                'mean': float(np.mean(mses)),
                'std': float(np.std(mses)),
                'min': float(np.min(mses)),
                'max': float(np.max(mses)),
                'cv': float(np.std(mses) / np.mean(mses)),  # Coefficient of variation
            },
            'best_epoch': {
                'mean': float(np.mean(epochs)),
                'std': float(np.std(epochs)),
                'min': int(np.min(epochs)),
                'max': int(np.max(epochs)),
            }
        }

    # Save to file
    Path('outputs').mkdir(exist_ok=True)
    summary_file = Path('outputs/retrain_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Retrain best SAE configurations with multiple seeds for stability analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python retrain_best_configs.py
    → Uses default seeds: 42, 123, 456, 789, 1011

  python retrain_best_configs.py --seeds 42 123 456
    → Uses custom seeds: 42, 123, 456

  python retrain_best_configs.py --variant topk
    → Only retrains TopK SAE best config

  python retrain_best_configs.py --num-seeds 10
    → Generates 10 random seeds (with 42 as first seed for reproducibility)
        """
    )

    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=[42, 123, 456, 789, 1011],
        help='Seeds to use for retraining (default: 42 123 456 789 1011)'
    )
    parser.add_argument(
        '--num-seeds',
        type=int,
        help='Generate N random seeds (seed 42 always included as first seed)'
    )
    parser.add_argument(
        '--variant',
        type=str,
        choices=['topk', 'gated', 'jumprelu', 'switch', 'all'],
        default='all',
        help='Which variant to retrain (default: all best configs)'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        default='cuda',
        help='Device to use for training (default: cuda)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1024,
        help='Batch size for training (default: 1024)'
    )
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=200,
        help='Maximum number of epochs (default: 200)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=5e-4,
        help='Learning rate (default: 5e-4)'
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print("RETRAIN BEST SAE CONFIGS WITH MULTIPLE SEEDS")
    print("="*80)

    # Determine seeds to use
    if args.num_seeds:
        # Always include seed 42 for reproducibility with original Phase 1
        seeds = [42] + list(np.random.randint(100, 10000, args.num_seeds - 1))
        print(f"\n📌 Generated {args.num_seeds} seeds (42 always included for reproducibility)")
        print(f"   Seeds: {seeds}")
    else:
        seeds = args.seeds
        print(f"\n📌 Using provided seeds: {seeds}")

    # Check device
    if args.device == 'cuda':
        if torch.cuda.is_available():
            print(f"📊 Device: {args.device} ({torch.cuda.get_device_name(0)})")
        else:
            print(f"⚠️  GPU not available, falling back to CPU")
            args.device = 'cpu'
    else:
        print(f"📊 Device: {args.device}")

    # Identify best configs
    print(f"\n{'='*80}")
    print("STEP 1: IDENTIFYING BEST CONFIGS PER VARIANT")
    print(f"{'='*80}\n")

    try:
        best_configs = identify_best_configs()
    except FileNotFoundError as e:
        print(str(e))
        return

    if not best_configs:
        print("\n❌ Could not identify any best configs")
        return

    # Filter by variant if specified
    if args.variant != 'all':
        if args.variant not in best_configs:
            print(f"\n❌ No best config found for variant: {args.variant}")
            return
        best_configs = {args.variant: best_configs[args.variant]}

    print(f"\n✓ Identified {len(best_configs)} best config(s) to retrain")

    # Load datasets
    print(f"\n{'='*80}")
    print("STEP 2: LOADING DATASETS")
    print(f"{'='*80}\n")

    train_dir = Path("outputs/activations/layer2/train")
    val_dir = Path("outputs/activations/layer2/val")
    test_dir = Path("outputs/activations/layer2/test")

    if not train_dir.exists():
        print(f"\n❌ Error: Activation directory not found")
        print(f"   Expected: {train_dir}")
        print(f"   Please ensure GNN activation data is available")
        return

    train_dataset = ActivationDataset(train_dir)
    val_dataset = ActivationDataset(val_dir)
    test_dataset = ActivationDataset(test_dir)

    print(f"✓ Train: {len(train_dataset)} activations")
    print(f"✓ Val:   {len(val_dataset)} activations")
    print(f"✓ Test:  {len(test_dataset)} activations")

    # Retrain each best config
    print(f"\n{'='*80}")
    print("STEP 3: RETRAINING BEST CONFIGS WITH MULTIPLE SEEDS")
    print(f"{'='*80}")

    all_results = {}

    for variant, config in best_configs.items():
        results = retrain_config(
            variant=variant,
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            seeds=seeds,
            device=args.device,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate
        )

        all_results[variant] = results

    # Generate summary
    print(f"\n{'='*80}")
    print("STEP 4: GENERATING SUMMARY")
    print(f"{'='*80}\n")

    summary = save_retrain_summary(all_results, seeds)

    for variant, stats in summary.items():
        print(f"\n✓ {variant.upper()}:")
        print(f"    Config:      {stats['config_name']}")
        print(f"    Seeds:       {len(stats['seeds'])} ({stats['seeds']})")
        print(f"    Test MSE:    {stats['test_mse']['mean']:.6f} ± {stats['test_mse']['std']:.6f}")
        print(f"    MSE Range:   {stats['test_mse']['min']:.6f} - {stats['test_mse']['max']:.6f}")
        print(f"    CV:          {stats['test_mse']['cv']:.3f} (coefficient of variation)")
        print(f"    Best Epoch:  {stats['best_epoch']['mean']:.0f} ± {stats['best_epoch']['std']:.0f}")

    # Final summary
    print(f"\n{'='*80}")
    print("✅ RETRAINING COMPLETE!")
    print(f"{'='*80}")

    print(f"\n📊 Results Summary:")
    print(f"   Total variants retrained: {len(all_results)}")
    total_checkpoints = sum(len(results) for results in all_results.values())
    print(f"   Total checkpoints saved: {total_checkpoints}")
    print(f"\n📁 Output Files:")
    print(f"   Checkpoints: checkpoints/sae_*_seed{{seed}}.pt ({total_checkpoints} files)")
    print(f"   Metrics:     outputs/sae_metrics_*_seed{{seed}}.json ({total_checkpoints} files)")
    print(f"   Summary:     outputs/retrain_summary.json")

    print(f"\n→ Next Step: Analyze feature stability across seeds")
    print(f"   python statistical_analysis_suite.py --seed-analysis")

    print(f"\n" + "="*80)


if __name__ == "__main__":
    main()
