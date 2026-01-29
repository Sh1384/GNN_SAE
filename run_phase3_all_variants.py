#!/usr/bin/env python3
"""
Run Phase 3 (Ablation Experiments) for ALL SAE Variants

This script extends the pipeline to run Phase 3a, 3b, 3c for all SAE variants
(TopK, Gated, JumpReLU, Switch) using their respective top configurations from Phase 2.

Usage:
    python run_phase3_all_variants.py --phase 3a  # Run only Phase 3a for all variants
    python run_phase3_all_variants.py --phase 3b  # Run only Phase 3b for all variants
    python run_phase3_all_variants.py --phase 3c  # Run only Phase 3c for all variants
    python run_phase3_all_variants.py --all       # Run all phases for all variants
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

# Add sae folder to path
SAE_DIR = Path(__file__).parent / "sae"
sys.path.insert(0, str(SAE_DIR))

# Output directories
OUTPUT_DIR = Path("outputs")
ABLATION_DIR = Path("ablations")
LOG_DIR = OUTPUT_DIR / "pipeline_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Top configuration for each variant (from Phase 2 results)
TOP_CONFIGS = {
    'jumprelu': {
        'latent_dim': 512,
        'variant_kwargs': {'threshold_init': 0.01, 'bandwidth': 0.01},
        'description': 'High capacity, low threshold (best overall composite_score=0.737)'
    },
    'topk': {
        'latent_dim': 256,
        'variant_kwargs': {'k': 16},
        'description': 'Medium capacity, moderate sparsity (composite_score=0.639)'
    },
    'switch': {
        'latent_dim': 1024,
        'variant_kwargs': {'num_experts': 8, 'latent_per_expert': 128, 'k_per_expert': 16},
        'description': '8 experts, 128D per expert, k=16 (composite_score=0.616)'
    },
    'gated': {
        'latent_dim': 512,
        'variant_kwargs': {'sparsity_coef': 0.0001},
        'description': 'High capacity, low sparsity penalty (composite_score=0.451)'
    }
}


def log_message(msg: str, log_file: Path):
    """Log message to both console and file."""
    print(msg)
    with open(log_file, 'a') as f:
        f.write(msg + '\n')


def run_script(script_name: str, args: List[str], log_file: Path, timeout: int = 3600) -> bool:
    """
    Run a Python script in the sae/ folder with given arguments.

    Args:
        script_name: Name of script (e.g., "run_ablation.py")
        args: List of command line arguments
        log_file: Path to log file
        timeout: Timeout in seconds

    Returns:
        True if successful, False otherwise
    """
    # All scripts are in sae/ folder
    script_path = SAE_DIR / script_name

    if not script_path.exists():
        log_message(f"ERROR: Script not found: {script_path}", log_file)
        return False

    cmd = [sys.executable, str(script_path)] + args
    log_message(f"Running: {' '.join(cmd)}", log_file)

    try:
        result = subprocess.run(
            cmd,
            check=True,
            timeout=timeout,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        log_message(result.stdout, log_file)
        return True
    except subprocess.CalledProcessError as e:
        log_message(f"ERROR: Script failed with exit code {e.returncode}", log_file)
        log_message(e.stdout, log_file)
        return False
    except subprocess.TimeoutExpired:
        log_message(f"ERROR: Script timed out after {timeout} seconds", log_file)
        return False
    except Exception as e:
        log_message(f"ERROR: Unexpected error: {str(e)}", log_file)
        return False


def run_phase3a_all_variants(log_file: Path) -> bool:
    """Phase 3a: SAE latent space ablations for ALL variants."""
    log_message(f"\n{'='*70}", log_file)
    log_message("PHASE 3a: SAE LATENT SPACE ABLATIONS (ALL VARIANTS)", log_file)
    log_message(f"{'='*70}\n", log_file)

    all_success = True

    for variant, config in TOP_CONFIGS.items():
        log_message(f"\n{'-'*70}", log_file)
        log_message(f"Processing variant: {variant.upper()}", log_file)
        log_message(f"Config: {config['description']}", log_file)
        log_message(f"{'-'*70}\n", log_file)

        latent_dim = config['latent_dim']
        variant_kwargs = config['variant_kwargs']

        # First, run feature significance analysis
        log_message(f"Running feature significance analysis for {variant}...", log_file)
        feature_sig_args = [
            "--variant", variant,
            "--source-csv", "outputs/sae_config_comparison.csv"
        ]

        if not run_script("analyze_feature_significance.py", feature_sig_args, log_file, timeout=7200):
            log_message(f"Warning: Feature significance analysis failed for {variant}, but continuing...", log_file)

        # Now run interpretability experiments
        # Note: The script auto-loads the best config for each variant from Phase 2 results
        args = ["--variant", variant]

        success = run_script("run_interpretability_experiments.py", args, log_file, timeout=10800)

        if success:
            # Move config file to variant-specific location
            config_file = Path("phase_3a_config.json")
            target_config = ABLATION_DIR / f"phase_3a_config_{variant}.json"
            if config_file.exists():
                target_config.parent.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.move(str(config_file), str(target_config))
                log_message(f"Saved config to {target_config}", log_file)
        else:
            log_message(f"Failed on variant: {variant}", log_file)
            all_success = False

    return all_success


def run_phase3b_all_variants(log_file: Path) -> bool:
    """Phase 3b: Native GNN ablations for ALL variants."""
    log_message(f"\n{'='*70}", log_file)
    log_message("PHASE 3b: NATIVE GNN ABLATIONS (ALL VARIANTS)", log_file)
    log_message(f"{'='*70}\n", log_file)

    all_success = True
    motifs = ['in_feedback_loop', 'in_cascade', 'in_feedforward_loop', 'in_single_input_module']

    for variant, config in TOP_CONFIGS.items():
        log_message(f"\n{'-'*70}", log_file)
        log_message(f"Processing variant: {variant.upper()}", log_file)
        log_message(f"Config: {config['description']}", log_file)
        log_message(f"{'-'*70}\n", log_file)

        # Check if Phase 3a config exists for this variant
        config_file = ABLATION_DIR / f"phase_3a_config_{variant}.json"
        if not config_file.exists():
            log_message(f"ERROR: Phase 3a config not found for {variant}. Run Phase 3a first.", log_file)
            all_success = False
            continue

        with open(config_file) as f:
            phase3a_config = json.load(f)

        latent_dim = phase3a_config.get('latent_dim', config['latent_dim'])
        variant_kwargs = phase3a_config.get('variant_kwargs', config['variant_kwargs'])

        # Run for all 4 motifs
        for motif in motifs:
            log_message(f"Running native ablation for motif: {motif}", log_file)
            args = [
                '--variant', variant,
                '--latent_dim', str(latent_dim),
                '--motif', motif,
                '--use-rpb'
            ]

            # Add variant-specific kwargs
            # Note: For now we only pass --k for topk; the native_gnn_ablation.py script
            # has hardcoded defaults for other variants that match the Phase 1 checkpoints
            if variant == 'topk':
                k = variant_kwargs.get('k', 16)
                args.extend(['--k', str(k)])

            if not run_script("native_gnn_ablation.py", args, log_file, timeout=3600):
                log_message(f"Failed on variant: {variant}, motif: {motif}", log_file)
                all_success = False

    return all_success


def run_phase3c_all_variants(log_file: Path) -> bool:
    """Phase 3c: Ablation strategy comparison for ALL variants."""
    log_message(f"\n{'='*70}", log_file)
    log_message("PHASE 3c: ABLATION STRATEGY COMPARISON (ALL VARIANTS)", log_file)
    log_message(f"{'='*70}\n", log_file)

    all_success = True

    for variant in TOP_CONFIGS.keys():
        log_message(f"\n{'-'*70}", log_file)
        log_message(f"Comparing ablation strategies for: {variant.upper()}", log_file)
        log_message(f"{'-'*70}\n", log_file)

        args = ['--motif-mode', '--variant', variant]

        if not run_script("compare_ablation_strategies.py", args, log_file, timeout=1800):
            log_message(f"Failed on variant: {variant}", log_file)
            all_success = False

    return all_success


def main():
    parser = argparse.ArgumentParser(description="Run Phase 3 ablation experiments for all SAE variants")
    parser.add_argument('--phase', type=str, choices=['3a', '3b', '3c', 'all'], default='all',
                        help='Which phase to run (3a, 3b, 3c, or all)')
    args = parser.parse_args()

    # Create log file
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR / f"phase3_all_variants_{timestamp}.log"

    log_message("="*70, log_file)
    log_message("PHASE 3 ABLATION EXPERIMENTS - ALL SAE VARIANTS", log_file)
    log_message("="*70, log_file)
    log_message(f"\nTimestamp: {timestamp}", log_file)
    log_message(f"Running phase(s): {args.phase}", log_file)
    log_message(f"\nTop configurations:", log_file)
    for variant, config in TOP_CONFIGS.items():
        log_message(f"  {variant}: {config['description']}", log_file)
    log_message("", log_file)

    success = True

    if args.phase in ['3a', 'all']:
        success = run_phase3a_all_variants(log_file) and success

    if args.phase in ['3b', 'all']:
        success = run_phase3b_all_variants(log_file) and success

    if args.phase in ['3c', 'all']:
        success = run_phase3c_all_variants(log_file) and success

    log_message("\n" + "="*70, log_file)
    if success:
        log_message("SUCCESS: All phases completed for all variants!", log_file)
    else:
        log_message("WARNING: Some phases failed. Check log for details.", log_file)
    log_message("="*70, log_file)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
