#!/usr/bin/env python3
"""
Run GNNExplainer Comparison for ALL SAE Variants

This script runs the SAE vs GNNExplainer comparison for all SAE variants
using their top configurations from Phase 2.

Usage:
    python run_gnnexplainer_comparison_all_variants.py --all
    python run_gnnexplainer_comparison_all_variants.py --variant topk
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Add sae folder to path
SAE_DIR = Path(__file__).parent / "sae"
sys.path.insert(0, str(SAE_DIR))

LOG_DIR = Path("outputs/pipeline_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def log_message(msg: str, log_file: Path):
    """Log message to both console and file."""
    print(msg)
    with open(log_file, 'a') as f:
        f.write(msg + '\n')


def run_comparison(variant: str, log_file: Path, timeout: int = 7200) -> bool:
    """
    Run GNNExplainer comparison for a single variant.

    Args:
        variant: SAE variant name
        log_file: Path to log file
        timeout: Timeout in seconds

    Returns:
        True if successful, False otherwise
    """
    script_path = SAE_DIR / "compare_sae_vs_gnnexplainer.py"

    cmd = [sys.executable, str(script_path), '--variant', variant]
    log_message(f"\nRunning: {' '.join(cmd)}", log_file)

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
        log_message(f"ERROR: Comparison failed with exit code {e.returncode}", log_file)
        log_message(e.stdout, log_file)
        return False
    except subprocess.TimeoutExpired:
        log_message(f"ERROR: Comparison timed out after {timeout} seconds", log_file)
        return False
    except Exception as e:
        log_message(f"ERROR: Unexpected error: {str(e)}", log_file)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run GNNExplainer comparison for all SAE variants",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all variants
  python run_gnnexplainer_comparison_all_variants.py --all

  # Run specific variant
  python run_gnnexplainer_comparison_all_variants.py --variant jumprelu

  # Run multiple specific variants
  python run_gnnexplainer_comparison_all_variants.py --variant topk jumprelu
        """
    )
    parser.add_argument('--variant', type=str, nargs='+', choices=['topk', 'gated', 'jumprelu', 'switch'],
                        help='SAE variant(s) to run')
    parser.add_argument('--all', action='store_true',
                        help='Run comparison for all variants')
    args = parser.parse_args()

    # Determine which variants to run
    if args.all:
        variants = ['jumprelu', 'topk', 'switch', 'gated']  # Ordered by Phase 2 performance
    elif args.variant:
        variants = args.variant
    else:
        print("ERROR: Must specify --variant or --all")
        return 1

    # Create log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR / f"gnnexplainer_comparison_{timestamp}.log"

    log_message("="*70, log_file)
    log_message("GNNExplainer COMPARISON - ALL SAE VARIANTS", log_file)
    log_message("="*70, log_file)
    log_message(f"\nTimestamp: {timestamp}", log_file)
    log_message(f"Variants to run: {', '.join(variants)}", log_file)
    log_message("", log_file)

    success_count = 0
    failed_variants = []

    for variant in variants:
        log_message(f"\n{'='*70}", log_file)
        log_message(f"VARIANT: {variant.upper()}", log_file)
        log_message(f"{'='*70}", log_file)

        if run_comparison(variant, log_file, timeout=7200):
            success_count += 1
            log_message(f"\n✓ {variant} comparison completed successfully", log_file)
        else:
            failed_variants.append(variant)
            log_message(f"\n✗ {variant} comparison failed", log_file)

    # Print summary
    log_message("\n" + "="*70, log_file)
    log_message("SUMMARY", log_file)
    log_message("="*70, log_file)
    log_message(f"\nTotal variants: {len(variants)}", log_file)
    log_message(f"Successful: {success_count}", log_file)
    log_message(f"Failed: {len(failed_variants)}", log_file)

    if failed_variants:
        log_message(f"\nFailed variants: {', '.join(failed_variants)}", log_file)

    log_message("\n" + "="*70, log_file)
    if success_count == len(variants):
        log_message("SUCCESS: All comparisons completed!", log_file)
        log_message("="*70, log_file)
        log_message(f"\nResults saved to: outputs/gnnexplainer_comparison/comparison_all_variants.csv", log_file)
        return 0
    else:
        log_message("WARNING: Some comparisons failed. Check log for details.", log_file)
        log_message("="*70, log_file)
        return 1


if __name__ == "__main__":
    sys.exit(main())
