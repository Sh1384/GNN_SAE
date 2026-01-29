#!/usr/bin/env python3
"""
Multi-GPU SAE Training and Analysis Pipeline

Complete implementation of the SAE variants pipeline from sae_colab_pipeline.ipynb,
optimized for multi-GPU execution on local cluster.

Phases:
  1: Train all 30 SAE configurations in parallel (4 GPUs)
  2: Configuration comparison
  2.5: Cross-variant comparison
  2b: Multi-seed retraining of best configs
  3a: SAE latent space ablations
  3b: Native GNN ablations
  3c: Ablation strategy comparison
  3d: Mixed-motif generalization
  4: Statistical validation
  5: Visualization

Hardware: 4x NVIDIA TITAN V (12 GB each)
Input: Layer 2 GNN activations (80-dim)

Usage:
    python run_sae_pipeline_multi_gpu.py --all
    python run_sae_pipeline_multi_gpu.py --phase 1
    python run_sae_pipeline_multi_gpu.py --phase 2
    python run_sae_pipeline_multi_gpu.py --gpus 0,1,2,3
"""

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch

# Add sae folder to path for imports
SAE_DIR = Path(__file__).parent / "sae"
sys.path.insert(0, str(SAE_DIR))

# Now we can import from sae folder
from sparse_autoencoder import TopKSAE, GatedSAE, JumpReLUSAE, SwitchSAE, ActivationDataset, SAETrainer, save_json


# ============================================================================
# Configuration
# ============================================================================

NUM_GPUS = 4
GPU_DEVICES = [0, 1, 2, 3]
SEED = 42
BATCH_SIZE = 1024
NUM_EPOCHS = 200
LEARNING_RATE = 5e-4
INPUT_DIM = 80

# All 30 SAE configurations
SAE_CONFIGS = {
    'topk': [
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
    ],
    'gated': [
        {'latent_dim': 128, 'sparsity_coef': 1e-4},
        {'latent_dim': 128, 'sparsity_coef': 5e-4},
        {'latent_dim': 128, 'sparsity_coef': 1e-3},
        {'latent_dim': 256, 'sparsity_coef': 1e-4},
        {'latent_dim': 256, 'sparsity_coef': 5e-4},
        {'latent_dim': 256, 'sparsity_coef': 1e-3},
        {'latent_dim': 512, 'sparsity_coef': 1e-4},
        {'latent_dim': 512, 'sparsity_coef': 5e-4},
        {'latent_dim': 512, 'sparsity_coef': 1e-3},
    ],
    'jumprelu': [
        {'latent_dim': 128, 'threshold_init': 0.01, 'bandwidth': 0.01},
        {'latent_dim': 128, 'threshold_init': 0.1, 'bandwidth': 0.01},
        {'latent_dim': 256, 'threshold_init': 0.01, 'bandwidth': 0.01},
        {'latent_dim': 256, 'threshold_init': 0.1, 'bandwidth': 0.01},
        {'latent_dim': 512, 'threshold_init': 0.01, 'bandwidth': 0.01},
        {'latent_dim': 512, 'threshold_init': 0.1, 'bandwidth': 0.01},
    ],
    'switch': [
        {'num_experts': 4, 'latent_per_expert': 64, 'k_per_expert': 8},
        {'num_experts': 4, 'latent_per_expert': 128, 'k_per_expert': 16},
        {'num_experts': 8, 'latent_per_expert': 64, 'k_per_expert': 8},
        {'num_experts': 8, 'latent_per_expert': 128, 'k_per_expert': 16},
    ]
}


# ============================================================================
# Helper Functions
# ============================================================================

def setup_logging() -> Path:
    """Setup logging directory and return log file path."""
    log_dir = Path("outputs/pipeline_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pipeline_{timestamp}.log"

    return log_file


def log_message(message: str, log_file: Path):
    """Log message to both console and file."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"

    print(log_entry)

    with open(log_file, 'a') as f:
        f.write(log_entry + "\n")


def check_prerequisites(log_file: Path) -> bool:
    """Check if all prerequisites are met."""
    log_message("Checking prerequisites...", log_file)

    # Check activations
    train_dir = Path("outputs/activations/layer2_new/train")
    val_dir = Path("outputs/activations/layer2_new/val")
    test_dir = Path("outputs/activations/layer2_new/test")

    if not train_dir.exists():
        log_message(f"ERROR: Training activations not found at {train_dir}", log_file)
        log_message("Run: python gnn_train.py first", log_file)
        return False

    train_files = list(train_dir.glob("graph_*.pt"))
    val_files = list(val_dir.glob("graph_*.pt"))
    test_files = list(test_dir.glob("graph_*.pt"))

    log_message(f"  ✓ Train: {len(train_files)} files", log_file)
    log_message(f"  ✓ Val: {len(val_files)} files", log_file)
    log_message(f"  ✓ Test: {len(test_files)} files", log_file)

    # Check GPU
    if not torch.cuda.is_available():
        log_message("WARNING: CUDA not available", log_file)
    else:
        n_gpus = torch.cuda.device_count()
        log_message(f"  ✓ Found {n_gpus} CUDA devices", log_file)
        for i in range(min(n_gpus, NUM_GPUS)):
            gpu_name = torch.cuda.get_device_name(i)
            log_message(f"    GPU {i}: {gpu_name}", log_file)

    log_message("✓ All prerequisites met", log_file)
    return True


def get_variant_name(variant_type: str, config: Dict) -> str:
    """Generate variant name from config."""
    if variant_type == 'topk':
        return f"topk_latent{config['latent_dim']}_k{config['k']}"
    elif variant_type == 'gated':
        coef_str = f"{config['sparsity_coef']:.0e}"
        return f"gated_latent{config['latent_dim']}_lambda{coef_str}"
    elif variant_type == 'jumprelu':
        thresh_str = f"{config['threshold_init']:.0e}"
        bw_str = f"{config['bandwidth']:.0e}"
        return f"jumprelu_latent{config['latent_dim']}_thresh{thresh_str}_bw{bw_str}"
    elif variant_type == 'switch':
        total_latent = config['num_experts'] * config['latent_per_expert']
        return f"switch_experts{config['num_experts']}_latent{total_latent}_k{config['k_per_expert']}"


def get_existing_checkpoints() -> List[str]:
    """Get list of already trained configs."""
    ckpt_dir = Path("checkpoints")
    if not ckpt_dir.exists():
        return []

    existing = []
    for ckpt_file in ckpt_dir.glob("sae_*.pt"):
        # Extract variant name from filename
        name = ckpt_file.stem.replace("sae_", "").replace("_seed42", "")
        existing.append(name)

    return existing


# ============================================================================
# Phase 1: Parallel Training
# ============================================================================

def train_single_config(
    gpu_id: int,
    variant_type: str,
    config: Dict,
    log_file: Path
) -> Tuple[str, bool]:
    """Train a single SAE configuration on specific GPU."""
    import torch
    import numpy as np
    from sparse_autoencoder import TopKSAE, GatedSAE, JumpReLUSAE, SwitchSAE, ActivationDataset, SAETrainer, save_json
    from torch.utils.data import DataLoader

    device = f'cuda:{gpu_id}'

    # Set seeds
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    variant_name = get_variant_name(variant_type, config)

    try:
        log_message(f"[GPU {gpu_id}] Starting: {variant_name}", log_file)

        # Load datasets
        train_dir = Path("outputs/activations/layer2_new/train")
        val_dir = Path("outputs/activations/layer2_new/val")
        test_dir = Path("outputs/activations/layer2_new/test")

        train_dataset = ActivationDataset(train_dir)
        val_dataset = ActivationDataset(val_dir)
        test_dataset = ActivationDataset(test_dir)

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        # Create model
        if variant_type == 'topk':
            model = TopKSAE(input_dim=INPUT_DIM, **config)
        elif variant_type == 'gated':
            model = GatedSAE(input_dim=INPUT_DIM, **config)
        elif variant_type == 'jumprelu':
            model = JumpReLUSAE(input_dim=INPUT_DIM, **config)
        elif variant_type == 'switch':
            model = SwitchSAE(input_dim=INPUT_DIM, **config)

        # Train
        trainer = SAETrainer(model, device=device, learning_rate=LEARNING_RATE)

        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0

        ckpt_path = Path(f"checkpoints/sae_{variant_name}_seed{SEED}.pt")
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(NUM_EPOCHS):
            train_metrics = trainer.train_epoch(train_loader)
            val_metrics = trainer.evaluate(val_loader)

            # Update history
            trainer.history['train_loss'].append(train_metrics['total'])
            trainer.history['train_recon'].append(train_metrics['reconstruction'])
            trainer.history['train_sparsity'].append(train_metrics['sparsity'])
            trainer.history['train_l0'].append(train_metrics['l0_sparsity'])
            trainer.history['val_loss'].append(val_metrics['total'])
            trainer.history['val_recon'].append(val_metrics['reconstruction'])
            trainer.history['val_sparsity'].append(val_metrics['sparsity'])
            trainer.history['val_l0'].append(val_metrics['l0_sparsity'])

            if (epoch + 1) % 10 == 0:
                log_message(
                    f"[GPU {gpu_id}] {variant_name} Epoch {epoch+1}/{NUM_EPOCHS} "
                    f"Train: {train_metrics['total']:.6f} Val: {val_metrics['total']:.6f}",
                    log_file
                )

            if val_metrics['total'] < best_val_loss:
                best_val_loss = val_metrics['total']
                patience_counter = 0
                trainer.save_model(str(ckpt_path))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    log_message(f"[GPU {gpu_id}] Early stopping at epoch {epoch+1}", log_file)
                    break

        # Evaluate on test
        trainer.load_model(str(ckpt_path))
        test_metrics = trainer.evaluate(test_loader)

        # Save metrics
        variant_config = model.get_config()
        variant_config['variant_name'] = variant_name
        variant_config['seed'] = SEED

        metrics_path = Path(f"outputs/sae_metrics_{variant_name}_seed{SEED}.json")
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

        final_metrics = {
            'best_val_loss': float(best_val_loss),
            'test_metrics': {k: float(v) for k, v in test_metrics.items()},
            'train_history': trainer.history,
            'config': variant_config
        }
        save_json(final_metrics, str(metrics_path))

        log_message(
            f"[GPU {gpu_id}] ✓ Completed: {variant_name} Test Loss: {test_metrics['total']:.6f}",
            log_file
        )

        return (variant_name, True)

    except Exception as e:
        log_message(f"[GPU {gpu_id}] ✗ Error training {variant_name}: {str(e)}", log_file)
        return (variant_name, False)


def run_phase1_parallel(gpu_devices: List[int], log_file: Path, force_retrain: bool = False) -> Dict:
    """Train all 30 SAE configurations in parallel across GPUs."""
    log_message(f"\n{'='*70}", log_file)
    log_message("PHASE 1: PARALLEL SAE TRAINING", log_file)
    log_message(f"{'='*70}", log_file)

    # Get all configs to train
    all_configs = []
    for variant_type, configs in SAE_CONFIGS.items():
        for config in configs:
            variant_name = get_variant_name(variant_type, config)
            all_configs.append((variant_type, config, variant_name))

    # Filter out already trained if not forcing retrain
    if not force_retrain:
        existing = set(get_existing_checkpoints())
        to_train = [(vt, c, vn) for vt, c, vn in all_configs if vn not in existing]

        if len(to_train) < len(all_configs):
            log_message(f"Skipping {len(all_configs) - len(to_train)} already trained configs", log_file)
    else:
        to_train = all_configs

    if not to_train:
        log_message("All configurations already trained!", log_file)
        return {'total': len(all_configs), 'successful': len(all_configs), 'skipped': len(all_configs)}

    log_message(f"Training {len(to_train)} configurations", log_file)
    log_message(f"Using GPUs: {gpu_devices}\n", log_file)

    # Parallel training
    results = []
    config_idx = 0
    running_processes = []

    ctx = mp.get_context('spawn')

    while config_idx < len(to_train) or running_processes:
        # Launch new processes
        while len(running_processes) < len(gpu_devices) and config_idx < len(to_train):
            variant_type, config, variant_name = to_train[config_idx]
            gpu_id = gpu_devices[len(running_processes)]

            process = ctx.Process(
                target=train_single_config,
                args=(gpu_id, variant_type, config, log_file)
            )
            process.start()
            running_processes.append((process, variant_name, gpu_id))

            config_idx += 1
            time.sleep(2)

        # Check for completed
        still_running = []
        for process, variant_name, gpu_id in running_processes:
            if not process.is_alive():
                process.join()

                # Check success
                ckpt_path = Path(f"checkpoints/sae_{variant_name}_seed{SEED}.pt")
                success = ckpt_path.exists()

                results.append({
                    'variant': variant_name,
                    'gpu_id': gpu_id,
                    'success': success
                })

                log_message(f"Process completed: {variant_name}, GPU {gpu_id}, Success: {success}", log_file)
            else:
                still_running.append((process, variant_name, gpu_id))

        running_processes = still_running

        if running_processes:
            time.sleep(10)

    successful = sum(1 for r in results if r['success'])
    log_message(f"\nPhase 1 Summary: {successful}/{len(to_train)} completed", log_file)

    return {
        'total': len(to_train),
        'successful': successful,
        'failed': len(to_train) - successful,
        'results': results
    }


# ============================================================================
# Subsequent Phases (run scripts from sae folder)
# ============================================================================

def run_script(script_name: str, args: List[str], log_file: Path, timeout: Optional[int] = None) -> bool:
    """Run a Python script from the sae folder."""
    script_path = SAE_DIR / script_name

    if not script_path.exists():
        log_message(f"ERROR: {script_name} not found in sae/", log_file)
        return False

    log_message(f"Running: {script_name} {' '.join(args)}", log_file)

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)] + args,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(Path.cwd())  # Run from project root, not sae/
        )

        if result.returncode == 0:
            log_message(f"✓ {script_name} completed successfully", log_file)
            if result.stdout:
                log_message(result.stdout, log_file)
            return True
        else:
            log_message(f"✗ {script_name} failed", log_file)
            if result.stderr:
                log_message(f"Error: {result.stderr}", log_file)
            return False

    except subprocess.TimeoutExpired:
        log_message(f"✗ {script_name} timed out", log_file)
        return False
    except Exception as e:
        log_message(f"✗ Error running {script_name}: {str(e)}", log_file)
        return False


def run_phase2(log_file: Path) -> bool:
    """Phase 2: Configuration comparison."""
    log_message(f"\n{'='*70}", log_file)
    log_message("PHASE 2: CONFIGURATION COMPARISON", log_file)
    log_message(f"{'='*70}\n", log_file)

    return run_script("compare_sae_configs.py", [], log_file, timeout=3600)


def run_phase2_5(log_file: Path) -> bool:
    """Phase 2.5: Cross-variant comparison."""
    log_message(f"\n{'='*70}", log_file)
    log_message("PHASE 2.5: CROSS-VARIANT COMPARISON", log_file)
    log_message(f"{'='*70}\n", log_file)

    return run_script("compare_sae_variants.py", [], log_file, timeout=600)


def run_phase2b(log_file: Path) -> bool:
    """Phase 2b: Multi-seed retraining."""
    log_message(f"\n{'='*70}", log_file)
    log_message("PHASE 2b: MULTI-SEED RETRAINING", log_file)
    log_message(f"{'='*70}\n", log_file)

    return run_script("retrain_best_configs.py", [], log_file, timeout=14400)  # 4 hours


def get_best_variant(log_file: Path) -> str:
    """Determine best variant from Phase 2 results."""
    config_csv = Path("outputs/sae_config_comparison.csv")
    variant = "topk"  # Default
    
    if config_csv.exists():
        try:
            import pandas as pd
            df = pd.read_csv(config_csv)
            if len(df) > 0:
                # Get the top-ranked config (assuming sorted by composite_score or similar)
                best_row = df.iloc[0]
                variant = best_row.get('variant', 'topk')
                log_message(f"Using best variant from Phase 2: {variant}", log_file)
        except Exception as e:
            log_message(f"Warning: Could not read config CSV, using default variant: {e}", log_file)
    else:
        log_message("Warning: Phase 2 results not found, using default variant 'topk'", log_file)
    
    return variant


def run_phase3a(log_file: Path) -> bool:
    """Phase 3a: SAE latent space ablations."""
    log_message(f"\n{'='*70}", log_file)
    log_message("PHASE 3a: SAE LATENT SPACE ABLATIONS", log_file)
    log_message(f"{'='*70}\n", log_file)

    # Determine best variant
    variant = get_best_variant(log_file)
    
    # First, run analyze_feature_significance.py to generate correlation data
    log_message(f"Running feature significance analysis for variant: {variant}", log_file)
    feature_sig_args = [
        "--variant", variant,
        "--source-csv", "outputs/sae_config_comparison.csv"
    ]
    
    if not run_script("analyze_feature_significance.py", feature_sig_args, log_file, timeout=7200):  # 2 hours
        log_message(f"Warning: Feature significance analysis failed, but continuing...", log_file)
    
    # Now run interpretability experiments
    args = ["--variant", variant]
    success = run_script("run_interpretability_experiments.py", args, log_file, timeout=10800)  # 3 hours
    
    # Move config file to expected location if it was created
    config_file = Path("phase_3a_config.json")
    target_config = Path("ablations/phase_3a_config.json")
    if config_file.exists() and not target_config.exists():
        target_config.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.move(str(config_file), str(target_config))
        log_message(f"Moved phase_3a_config.json to {target_config}", log_file)
    
    return success


def run_phase3b(log_file: Path) -> bool:
    """Phase 3b: Native GNN ablations."""
    log_message(f"\n{'='*70}", log_file)
    log_message("PHASE 3b: NATIVE GNN ABLATIONS", log_file)
    log_message(f"{'='*70}\n", log_file)

    # Read config from Phase 3a metadata
    config_file = Path("ablations/phase_3a_config.json")
    if not config_file.exists():
        log_message("ERROR: Phase 3a config not found. Run Phase 3a first.", log_file)
        return False

    with open(config_file) as f:
        config = json.load(f)

    variant = config['variant']
    latent_dim = config.get('latent_dim', 256)
    variant_kwargs = config.get('variant_kwargs', {})
    
    # Get k from variant_kwargs for topk variant
    k = variant_kwargs.get('k', 16) if variant == 'topk' else None

    # Run for all 4 motifs
    motifs = ['in_feedback_loop', 'in_cascade', 'in_feedforward_loop', 'in_single_input_module']

    for motif in motifs:
        log_message(f"Running native ablation for motif: {motif}", log_file)
        args = [
            '--variant', variant,
            '--latent_dim', str(latent_dim),
            '--motif', motif,
            '--use-rpb'
        ]
        
        # Add k for topk variant
        if variant == 'topk' and k is not None:
            args.extend(['--k', str(k)])

        if not run_script("native_gnn_ablation.py", args, log_file, timeout=3600):
            log_message(f"Failed on motif: {motif}", log_file)
            return False

    return True


def run_phase3c(log_file: Path) -> bool:
    """Phase 3c: Ablation strategy comparison."""
    log_message(f"\n{'='*70}", log_file)
    log_message("PHASE 3c: ABLATION STRATEGY COMPARISON", log_file)
    log_message(f"{'='*70}\n", log_file)

    # Get variant from Phase 3a config if available, otherwise determine from Phase 2
    config_file = Path("ablations/phase_3a_config.json")
    if config_file.exists():
        try:
            with open(config_file) as f:
                config = json.load(f)
            variant = config.get('variant', 'topk')
        except Exception:
            variant = get_best_variant(log_file)
    else:
        variant = get_best_variant(log_file)
    
    args = ['--motif-mode', '--variant', variant]
    return run_script("compare_ablation_strategies.py", args, log_file, timeout=1800)


def run_phase3d(log_file: Path) -> bool:
    """Phase 3d: Mixed-motif generalization."""
    log_message(f"\n{'='*70}", log_file)
    log_message("PHASE 3d: MIXED-MOTIF GENERALIZATION", log_file)
    log_message(f"{'='*70}\n", log_file)

    # First generate mixed-motif activations
    if not run_script("generate_mixed_motif_activations.py", [], log_file, timeout=1800):
        return False

    # Then run ablations (would need to specify features - this is a placeholder)
    log_message("Mixed-motif generalization requires manual feature selection", log_file)
    log_message("Please run run_ablation.py and native_gnn_ablation.py with --use_mixed_motifs", log_file)

    return True


def run_phase4(log_file: Path) -> bool:
    """Phase 4: Statistical validation."""
    log_message(f"\n{'='*70}", log_file)
    log_message("PHASE 4: STATISTICAL VALIDATION", log_file)
    log_message(f"{'='*70}\n", log_file)

    args = ['--seed-analysis']
    return run_script("statistical_analysis_suite.py", args, log_file, timeout=3600)


def run_phase5(log_file: Path) -> bool:
    """Phase 5: Visualization."""
    log_message(f"\n{'='*70}", log_file)
    log_message("PHASE 5: VISUALIZATION", log_file)
    log_message(f"{'='*70}\n", log_file)

    success = True

    # Feature activations
    if not run_script("visualize_feature_activations.py", [], log_file, timeout=1800):
        success = False

    # Reconstruction fidelity
    if not run_script("analyze_sae_reconstruction_fidelity.py", [], log_file, timeout=1800):
        success = False

    return success


# ============================================================================
# Main Pipeline
# ============================================================================

def run_full_pipeline(
    gpu_devices: List[int],
    start_phase: int = 1,
    end_phase: int = 5,
    force_retrain: bool = False
):
    """Run the complete SAE pipeline."""
    log_file = setup_logging()

    log_message("="*70, log_file)
    log_message("MULTI-GPU SAE PIPELINE", log_file)
    log_message("="*70, log_file)
    log_message(f"GPUs: {gpu_devices}", log_file)
    log_message(f"Phases: {start_phase} to {end_phase}", log_file)
    log_message(f"Log: {log_file}\n", log_file)

    # Check prerequisites
    if not check_prerequisites(log_file):
        log_message("\n✗ Prerequisites not met. Exiting.", log_file)
        return

    # Run phases
    if start_phase <= 1 <= end_phase:
        run_phase1_parallel(gpu_devices, log_file, force_retrain)

    if start_phase <= 2 <= end_phase:
        run_phase2(log_file)

    if start_phase <= 2.5 <= end_phase:
        run_phase2_5(log_file)

    if start_phase <= 2.9 <= end_phase:  # Phase 2b
        run_phase2b(log_file)

    if start_phase <= 3 <= end_phase:
        run_phase3a(log_file)
        run_phase3b(log_file)
        run_phase3c(log_file)
        run_phase3d(log_file)

    if start_phase <= 4 <= end_phase:
        run_phase4(log_file)

    if start_phase <= 5 <= end_phase:
        run_phase5(log_file)

    log_message(f"\n{'='*70}", log_file)
    log_message("PIPELINE COMPLETE", log_file)
    log_message(f"{'='*70}", log_file)
    log_message(f"Log saved to: {log_file}", log_file)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-GPU SAE Training and Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python run_sae_pipeline_multi_gpu.py --all

  # Run specific phase
  python run_sae_pipeline_multi_gpu.py --phase 1
  python run_sae_pipeline_multi_gpu.py --phase 2

  # Custom GPU selection
  python run_sae_pipeline_multi_gpu.py --gpus 0,1,2 --phase 1

  # Force retrain all
  python run_sae_pipeline_multi_gpu.py --phase 1 --force-retrain
        """
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all phases (1-5)'
    )

    parser.add_argument(
        '--phase',
        type=str,
        help='Run specific phase (1, 2, 2.5, 2b, 3, 4, 5) or range (e.g., "1-3")'
    )

    parser.add_argument(
        '--gpus',
        type=str,
        default='0,1,2,3',
        help='Comma-separated GPU IDs (default: 0,1,2,3)'
    )

    parser.add_argument(
        '--force-retrain',
        action='store_true',
        help='Force retrain all configs in Phase 1'
    )

    args = parser.parse_args()

    # Parse GPUs
    try:
        gpu_devices = [int(x.strip()) for x in args.gpus.split(',')]
    except ValueError:
        print(f"Error: Invalid GPU specification: {args.gpus}")
        sys.exit(1)

    # Determine phases to run
    phase_map = {'1': 1, '2': 2, '2.5': 2.5, '2b': 2.9, '3': 3, '4': 4, '5': 5}

    def parse_phase(phase_str):
        """Parse a phase string, handling both numeric and named phases."""
        if phase_str in phase_map:
            return phase_map[phase_str]
        try:
            return float(phase_str)
        except ValueError:
            print(f"Error: Invalid phase specification: {phase_str}")
            sys.exit(1)

    if args.all:
        start_phase, end_phase = 1, 5
    elif args.phase:
        if '-' in args.phase:
            start, end = args.phase.split('-')
            start_phase = parse_phase(start)
            end_phase = parse_phase(end)
        else:
            start_phase = end_phase = parse_phase(args.phase)
    else:
        print("Error: Must specify --all or --phase")
        parser.print_help()
        sys.exit(1)

    # Run pipeline
    run_full_pipeline(
        gpu_devices,
        start_phase,
        end_phase,
        args.force_retrain
    )


if __name__ == "__main__":
    main()
