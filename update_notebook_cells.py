#!/usr/bin/env python3
"""
Updates sae_colab_pipeline.ipynb to incorporate new Phase 3 architecture:
1. Phase 3a: Select best config by max_rpb_abs, extract variant, save metadata
2. Phase 3b: Read metadata, use correct variant/parameters
3. Phase 3c: Remove variant loop, load single variant results
"""

import json
from pathlib import Path

notebook_path = Path('sae_colab_pipeline.ipynb')

# Phase 3a new cell code
phase_3a_code = '''#@title 5. PHASE 3a: Motif-Guided SAE Latent Ablations (~2 hours)

import subprocess
import sys
from pathlib import Path
import pandas as pd
import json

print("\\n" + "="*80)
print("PHASE 3a: MOTIF-GUIDED SAE LATENT SPACE ABLATIONS")
print("="*80)

script = Path('run_interpretability_experiments.py')

if not script.exists():
    print(f"\\n❌ ERROR: {script} not found!")
else:
    print(f"\\n✓ Found motif-guided ablation script: {script}")

    # Load Phase 2 results to get best config
    csv_file = Path('outputs/sae_config_comparison.csv')
    if not csv_file.exists():
        print(f"\\n❌ ERROR: {csv_file} not found!")
        print(f"   CRITICAL: Phase 2 (compare_sae_configs.py) must complete first")
        sys.exit(1)

    df = pd.read_csv(csv_file)

    # Select best config by MAX_RPB_ABS (not composite_score)
    best_idx = df['max_rpb_abs'].idxmax()
    best = df.loc[best_idx]

    best_variant = best['variant']
    best_latent = int(best['latent_dim'])

    print(f"\\n✓ Selected best config by MAX POINT-BISERIAL CORRELATION:")
    print(f"   Variant: {best_variant.upper()}")
    print(f"   Latent Dim: {best_latent}")

    # Extract variant-specific parameters
    if best_variant == 'topk':
        best_k = int(best['k'])
        param_str = f"latent_dim={best_latent}, k={best_k}"
        params = {'variant': best_variant, 'latent_dim': best_latent, 'k': best_k}
    elif best_variant == 'gated':
        best_sparsity_coef = float(best['sparsity_coef'])
        param_str = f"latent_dim={best_latent}, sparsity_coef={best_sparsity_coef:.0e}"
        params = {'variant': best_variant, 'latent_dim': best_latent, 'sparsity_coef': best_sparsity_coef}
    elif best_variant == 'jumprelu':
        best_threshold = float(best['threshold_init'])
        param_str = f"latent_dim={best_latent}, threshold_init={best_threshold:.0e}"
        params = {'variant': best_variant, 'latent_dim': best_latent, 'threshold_init': best_threshold}
    elif best_variant == 'switch':
        best_num_experts = int(best['num_experts'])
        best_latent_per_expert = int(best['latent_per_expert'])
        best_k_per_expert = int(best['k_per_expert'])
        param_str = f"num_experts={best_num_experts}, latent_per_expert={best_latent_per_expert}, k_per_expert={best_k_per_expert}"
        params = {
            'variant': best_variant,
            'num_experts': best_num_experts,
            'latent_per_expert': best_latent_per_expert,
            'k_per_expert': best_k_per_expert
        }

    print(f"   {param_str}")
    print(f"   Max |rpb|: {best['max_rpb_abs']:.3f}")

    # Save metadata for Phase 3b and 3c to use
    metadata = {
        'phase': '3a',
        'selected_by': 'max_rpb_abs',
        'best_config': params,
        'max_rpb_abs': float(best['max_rpb_abs']),
        'composite_score': float(best['composite_score'])
    }

    ablations_dir = Path('ablations')
    ablations_dir.mkdir(exist_ok=True)

    metadata_file = ablations_dir / 'phase_3a_config.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\\n✓ Saved config metadata to: {metadata_file}")

    print(f"\\n🔍 Running motif-guided SAE latent space ablations...")
    print(f"   Process: Ablate top features for each motif group")
    print(f"   Statistics: z-scores, percentiles, p-values vs random controls")
    print(f"   Output: ablations/results/ (grouped motif results)")
    print(f"   Also: ablations/interpretability_*/ (aggregated stats)\\n")

    # Run with variant parameter
    cmd = [
        sys.executable, str(script),
        '--variant', best_variant,
        '--latent_dim', str(best_latent),
        '--min_rpb', '0.05',  # Configurable threshold
        '--n_random_trials', '20'  # Configurable trials
    ]

    # Add variant-specific parameters to command if needed
    if best_variant == 'topk':
        cmd.extend(['--k', str(best_k)])

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        print(f"\\n\\n{'='*80}")
        print(f"✅ PHASE 3a COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")

        # Verify outputs
        ablation_results = Path('ablations/results')
        if ablation_results.exists():
            result_files = list(ablation_results.glob('*.csv'))
            print(f"\\n📊 Ablation Results:")
            print(f"   Total result files: {len(result_files)}")

            # Show motif results
            motif_names = ['feedback_loop', 'cascade', 'feedforward_loop', 'single_input_module']
            print(f"\\n   Motif-Specific Results (by variant {best_variant.upper()}):")
            for motif in motif_names:
                motif_files = [f for f in result_files if motif in f.name]
                status = "✓" if len(motif_files) > 0 else "⚠️ "
                print(f"   {status} {motif}: {len(motif_files)} file(s)")
                for f in motif_files:
                    print(f"      - {f.name}")

        print(f"\\n✓ Outputs are ready for Phase 3b and 3c")
        print(f"\\n✓ Config metadata saved to: {metadata_file}")
        print(f"\\n→ Continue to PHASE 3b")
    else:
        print(f"\\n❌ Phase 3a failed with return code {result.returncode}")
        print(f"   CRITICAL: Phase 3b and 3c will fail without Phase 3a outputs!")
'''

# Phase 3b new cell code
phase_3b_code = '''#@title 6. PHASE 3b: Native GNN Ablations (~1 hour for all 4 motifs)

import subprocess
import sys
from pathlib import Path
import pandas as pd
import json

print("\\n" + "="*80)
print("PHASE 3b: NATIVE GNN ACTIVATION SPACE ABLATIONS (ALL 4 MOTIFS)")
print("="*80)

script = Path('native_gnn_ablation.py')

if not script.exists():
    print(f"\\n❌ ERROR: {script} not found!")
else:
    print(f"\\n✓ Found ablation script: {script}")

    # Get best config from Phase 3a metadata
    metadata_file = Path('ablations/phase_3a_config.json')
    if not metadata_file.exists():
        print(f"\\n❌ ERROR: Phase 3a metadata not found!")
        print(f"   Please run Phase 3a first to generate: {metadata_file}")
        sys.exit(1)

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    best_config = metadata['best_config']
    best_variant = best_config['variant']
    best_latent = best_config['latent_dim']

    print(f"\\n✓ Using best config from Phase 3a:")
    print(f"   Variant: {best_variant}")
    print(f"   Latent Dim: {best_latent}")

    # Extract variant-specific parameters
    if best_variant == 'topk':
        best_k = int(best_config['k'])
        print(f"   K: {best_k}")
    elif best_variant == 'gated':
        best_k = int(best_config.get('k', 8))  # Use default or from config
        print(f"   Sparsity Coef: {best_config['sparsity_coef']:.0e}")
    elif best_variant == 'jumprelu':
        best_k = 8  # JumpReLU doesn't use k
        print(f"   Threshold: {best_config['threshold_init']:.0e}")
    elif best_variant == 'switch':
        best_k = int(best_config['k_per_expert'])
        print(f"   Num Experts: {best_config['num_experts']}")
        print(f"   Latent per Expert: {best_config['latent_per_expert']}")
        print(f"   K per Expert: {best_config['k_per_expert']}")

    print(f"\\n🔍 Running native GNN activation space ablations...")
    print(f"   Strategy: SAE r_pb-guided node patching per motif")
    print(f"   Method: Direct activation patching in 64D space")
    print(f"   Processing: All 4 motifs separately")
    print(f"     1. in_feedback_loop")
    print(f"     2. in_cascade")
    print(f"     3. in_feedforward_loop")
    print(f"     4. in_single_input_module")
    print(f"   Output: outputs/native_gnn_ablations/ (REQUIRED for Phase 3c)\\n")

    # Define all 4 motifs
    motifs = [
        'in_feedback_loop',
        'in_cascade',
        'in_feedforward_loop',
        'in_single_input_module'
    ]

    results_count = 0
    failed_motifs = []

    # Run native ablation for EACH motif SEPARATELY
    for i, motif in enumerate(motifs, 1):
        print(f"\\n{'─'*80}")
        print(f"[{i}/{len(motifs)}] Processing Motif: {motif}")
        print(f"{'─'*80}")
        print(f"Running: native_gnn_ablation.py --variant {best_variant} --latent_dim {best_latent} --use-rpb --motif {motif}\\n")

        result = subprocess.run(
            [sys.executable, str(script),
             '--variant', best_variant,
             '--latent_dim', str(best_latent),
             '--use-rpb',
             '--motif', motif],
            capture_output=False
        )

        if result.returncode == 0:
            results_count += 1
            print(f"\\n✓ [{i}/{len(motifs)}] {motif} completed successfully")
        else:
            failed_motifs.append(motif)
            print(f"\\n⚠️  [{i}/{len(motifs)}] {motif} failed (return code {result.returncode})")

    print(f"\\n\\n{'='*80}")
    if results_count == len(motifs):
        print(f"✅ PHASE 3b COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        print(f"\\n✓ All 4 motifs processed successfully")

        # Verify outputs
        ablation_dir = Path('outputs/native_gnn_ablations')
        if ablation_dir.exists():
            results = list(ablation_dir.glob('native_ablation_*.csv'))
            print(f"\\n📊 Native Ablation Results:")
            print(f"   Total result files: {len(results)}")

            print(f"\\n   Motif-Specific Results:")
            for motif in motifs:
                motif_files = [f for f in results if motif in f.name]
                status = "✓" if len(motif_files) > 0 else "⚠️ "
                print(f"   {status} {motif}: {len(motif_files)} file(s)")
                for f in motif_files:
                    print(f"      - {f.name}")

            print(f"\\n✓ Outputs are ready for Phase 3c (motif-grouped comparison)")

        print(f"\\n✓ Results saved to: outputs/native_gnn_ablations/")
        print(f"\\n→ Continue to PHASE 3c")
    else:
        print(f"⚠️  PHASE 3b PARTIALLY COMPLETED")
        print(f"{'='*80}")
        print(f"\\n✓ {results_count}/{len(motifs)} motifs processed successfully")
        if failed_motifs:
            print(f"❌ Failed motifs: {', '.join(failed_motifs)}")
            print(f"\\n⚠️  CRITICAL: Phase 3c will fail without all 4 motif results")
            print(f"   Recommend re-running failed motifs before Phase 3c:")
            for motif in failed_motifs:
                print(f"     python native_gnn_ablation.py --variant {best_variant} --latent_dim {best_latent} --use-rpb --motif {motif}")
'''

# Phase 3c new cell code
phase_3c_code = '''#@title 7. PHASE 3c: Ablation Strategy Comparison (Motif-Grouped, Single Variant) (~15 min)

import subprocess
import sys
from pathlib import Path
import json

print("\\n" + "="*80)
print("PHASE 3c: ABLATION STRATEGY COMPARISON (MOTIF-GROUPED, SINGLE BEST VARIANT)")
print("="*80)

script = Path('compare_ablation_strategies.py')

if not script.exists():
    print(f"\\n❌ ERROR: {script} not found!")
else:
    print(f"\\n✓ Found comparison script: {script}")

    # Get best variant from Phase 3a metadata
    metadata_file = Path('ablations/phase_3a_config.json')
    if not metadata_file.exists():
        print(f"\\n❌ ERROR: Phase 3a metadata not found!")
        print(f"   Please run Phase 3a first to generate: {metadata_file}")
        sys.exit(1)

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    best_config = metadata['best_config']
    best_variant = best_config['variant']
    best_latent = best_config['latent_dim']

    print(f"\\n✓ Using best variant from Phase 3a: {best_variant.upper()}")
    print(f"   Latent Dim: {best_latent}")

    # Check prerequisites
    print(f"\\n📋 Checking Phase 3a and 3b outputs...")

    # Phase 3a check - look for grouped motif results for this variant
    phase_3a_results = []
    if Path('ablations/results').exists():
        phase_3a_results = list(Path('ablations/results').glob(f'*_{best_variant}_l{best_latent}_k*_results.csv'))
        if not phase_3a_results:
            # Fall back to old format (without variant)
            phase_3a_results = list(Path('ablations/results').glob(f'*_l{best_latent}_k*_results.csv'))

    # Phase 3b check - look for motif-specific native ablation files for this variant
    phase_3b_results = []
    if Path('outputs/native_gnn_ablations').exists():
        phase_3b_results = list(Path('outputs/native_gnn_ablations').glob(f'native_ablation_{best_variant}_rpb_*.csv'))

    print(f"   Phase 3a (SAE latent motif groups for {best_variant}): {len(phase_3a_results)} result file(s)")
    if phase_3a_results:
        for f in phase_3a_results[:3]:
            print(f"      ✓ {f.name}")
        if len(phase_3a_results) > 3:
            print(f"      ... and {len(phase_3a_results) - 3} more")

    print(f"   Phase 3b (Native GNN motif groups for {best_variant}): {len(phase_3b_results)} result file(s)")
    if phase_3b_results:
        for f in phase_3b_results[:3]:
            print(f"      ✓ {f.name}")
        if len(phase_3b_results) > 3:
            print(f"      ... and {len(phase_3b_results) - 3} more")

    if len(phase_3a_results) == 0:
        print(f"\\n❌ MISSING: Phase 3a outputs for {best_variant}")

    if len(phase_3b_results) == 0:
        print(f"❌ MISSING: Phase 3b outputs for {best_variant}")

    if len(phase_3a_results) > 0 and len(phase_3b_results) > 0:
        print(f"\\n✓ Both Phase 3a and 3b outputs found")

        print(f"\\n🔗 Comparing SAE latent vs native GNN ablations (MOTIF-GROUPED)...")
        print(f"   Analysis: Motif-group level agreement")
        print(f"   Variant: {best_variant.upper()}")
        print(f"   - Phase 3a: Groups features by motif, ablates groups")
        print(f"   - Phase 3b: Ranks features per motif via r_pb, patches top nodes")
        print(f"   - Phase 3c: Validates agreement between strategies AT MOTIF-GROUP LEVEL")
        print(f"   ")
        print(f"   All 4 motifs: in_feedback_loop, in_cascade, in_feedforward_loop, in_single_input_module")
        print(f"   ")
        print(f"   Output: Motif-group agreement scores + visualizations\\n")

        result = subprocess.run(
            [sys.executable, str(script),
             '--variant', best_variant,   # Use SINGLE best variant (not all-variants)
             '--latent_dim', str(best_latent),
             '--motif-mode'],             # Use MOTIF-GROUP mode
            capture_output=False
        )

        if result.returncode == 0:
            print(f"\\n\\n{'='*80}")
            print(f"✅ PHASE 3c COMPLETED SUCCESSFULLY!")
            print(f"{'='*80}")

            # Check outputs
            comparison_dir = Path('outputs/ablation_strategy_comparison')
            if comparison_dir.exists():
                csvs = list(comparison_dir.glob('*.csv'))
                plots = list(comparison_dir.glob('*.png'))
                print(f"\\n📊 Strategy Comparison Results (Motif-Grouped, {best_variant.upper()}):")
                print(f"   CSV files: {len(csvs)}")
                for csv in sorted(csvs)[:5]:
                    print(f"      ✓ {csv.name}")
                if len(csvs) > 5:
                    print(f"      ... and {len(csvs) - 5} more")

                print(f"   Plots: {len(plots)}")

                # Load and show results
                try:
                    import pandas as pd
                    comp_csv = comparison_dir / 'motif_agreement_summary.csv'
                    if comp_csv.exists():
                        df = pd.read_csv(comp_csv)
                        print(f"\\n🎯 MECHANISTIC VALIDITY FOR {best_variant.upper()}:")
                        print(f"{'─'*80}")
                        print("\\nAgreement by Motif:")
                        for motif in sorted(df['motif'].unique()):
                            motif_data = df[df['motif'] == motif]
                            if len(motif_data) > 0:
                                spearman_r = motif_data['spearman_r'].values[0]
                                n_graphs = int(motif_data['n_graphs'].values[0])
                                print(f"  {motif:25} | ρ = {spearman_r:.3f} ({n_graphs} graphs)")

                        # Overall summary
                        mean_r = df['spearman_r'].mean()
                        print(f"\\n  ✓ Average agreement: ρ = {mean_r:.3f}")
                    else:
                        print(f"\\n   (Could not load detailed results: {comp_csv.name} not found)")
                except Exception as e:
                    print(f"\\n   (Error loading results: {e})")

            print(f"\\n✓ Results saved to: outputs/ablation_strategy_comparison/")
            print(f"\\n→ Continue to PHASE 3d")
        else:
            print(f"\\n❌ Phase 3c failed with return code {result.returncode}")
    else:
        print(f"\\n❌ CANNOT RUN PHASE 3c - MISSING PREREQUISITES:")
        if len(phase_3a_results) == 0:
            print(f"   ✗ Phase 3a outputs for {best_variant}")
            print(f"     Expected: ablations/results/motif_*{best_variant}_l{best_latent}_k*_results.csv")
        if len(phase_3b_results) == 0:
            print(f"   ✗ Phase 3b outputs for {best_variant}")
            print(f"     Expected: outputs/native_gnn_ablations/native_ablation_{best_variant}_rpb_*.csv")
        print(f"\\n   → Complete Phase 3a and 3b first")
'''

# Phase 3d new cell code
phase_3d_code = '''#@title 8. PHASE 3d: Mixed-Motif Generalization Test (~45 min)

import subprocess
import sys
from pathlib import Path
import pandas as pd
import json

print("\\n" + "="*80)
print("PHASE 3d: MIXED-MOTIF GENERALIZATION TEST")
print("="*80)

print("\\n📋 Step 0: Load Phase 3a config metadata")

# Get best variant from Phase 3a metadata
metadata_file = Path('ablations/phase_3a_config.json')
if not metadata_file.exists():
    print(f"❌ ERROR: Phase 3a metadata not found!")
    print(f"   Please run Phase 3a first to generate: {metadata_file}")
    sys.exit(1)

with open(metadata_file, 'r') as f:
    metadata = json.load(f)

best_config = metadata['best_config']
best_variant = best_config['variant']
best_latent = best_config['latent_dim']

print(f"\\n✓ Using best config from Phase 3a:")
print(f"   Variant: {best_variant}")
print(f"   Latent Dim: {best_latent}")

# Extract variant-specific parameters
if best_variant == 'topk':
    best_k = int(best_config['k'])
    print(f"   K: {best_k}")
elif best_variant == 'gated':
    best_k = int(best_config.get('k', 8))
    print(f"   Sparsity Coef: {best_config['sparsity_coef']:.0e}")
elif best_variant == 'jumprelu':
    best_k = 8
    print(f"   Threshold: {best_config['threshold_init']:.0e}")
elif best_variant == 'switch':
    best_k = int(best_config['k_per_expert'])
    print(f"   Num Experts: {best_config['num_experts']}")
    print(f"   Latent per Expert: {best_config['latent_per_expert']}")
    print(f"   K per Expert: {best_config['k_per_expert']}")

print(f"\\n📋 Step 1: Preprocessing - Generate Mixed-Motif Activations")

script = Path('generate_mixed_motif_activations.py')
if not script.exists():
    print(f"⚠️  {script} not found, skipping preprocessing")
else:
    print(f"Running: python {script}")
    result = subprocess.run([sys.executable, str(script)], capture_output=False)
    if result.returncode == 0:
        print(f"✓ Mixed-motif preprocessing completed")
    else:
        print(f"⚠️  Preprocessing failed, continuing anyway...")

print(f"\\n📋 Step 2: SAE Latent Ablation on Mixed-Motif Graphs")

# Load Phase 2 data to get feature list
csv_file = Path('outputs/latent_correlations.csv')
if not csv_file.exists():
    print(f"❌ ERROR: {csv_file} not found!")
    print(f"   Phase 2 (compare_sae_configs.py) must complete first")
    sys.exit(1)

df_corr = pd.read_csv(csv_file)

# Filter to best variant correlations
df_variant_corr = df_corr[df_corr['variant'] == best_variant]

if df_variant_corr.empty:
    print(f"❌ ERROR: No correlations found for variant {best_variant}")
    sys.exit(1)

# Get top features (by |rpb|) from each motif
motif_features = {}
for motif in ['in_cascade', 'in_feedback_loop', 'in_feedforward_loop', 'in_single_input_module']:
    top_features = df_variant_corr[df_variant_corr['motif'] == motif].nlargest(5, 'rpb_abs')['feature'].tolist()
    motif_features[motif] = top_features

# Combine all features
all_features = []
for features in motif_features.values():
    all_features.extend(features)
all_features = sorted(list(set(all_features)))  # Remove duplicates

print(f"\\n✓ Selected top features across all motifs: {len(all_features)} features")
feature_str = ','.join([str(f) for f in all_features])

# Run SAE ablation on mixed-motif graphs
ablation_script = Path('run_ablation.py')
if not ablation_script.exists():
    print(f"❌ ERROR: {ablation_script} not found!")
else:
    print(f"\\nRunning: run_ablation.py on mixed-motif data with top features")
    print(f"  Command: python run_ablation.py --variant {best_variant} --latent_dim {best_latent} --use_mixed_motifs --feature {feature_str}")

    result = subprocess.run([
        sys.executable, str(ablation_script),
        '--variant', best_variant,
        '--latent_dim', str(best_latent),
        '--use_mixed_motifs',
        '--feature', feature_str,
        '--experiment_name', f'mixed_motifs_{best_variant}'
    ], capture_output=False)

    if result.returncode == 0:
        print(f"\\n✓ SAE mixed-motif ablation completed")
    else:
        print(f"\\n⚠️  SAE mixed-motif ablation failed")

print(f"\\n📋 Step 3: Native GNN Ablation on Mixed-Motif Graphs")

native_script = Path('native_gnn_ablation.py')
if not native_script.exists():
    print(f"❌ ERROR: {native_script} not found!")
else:
    print(f"Running: native_gnn_ablation.py on mixed-motif data with top features")
    print(f"  Command: python native_gnn_ablation.py --variant {best_variant} --latent_dim {best_latent} --use_mixed_motifs --feature {feature_str}")

    result = subprocess.run([
        sys.executable, str(native_script),
        '--variant', best_variant,
        '--latent_dim', str(best_latent),
        '--use_mixed_motifs',
        '--feature', feature_str
    ], capture_output=False)

    if result.returncode == 0:
        print(f"\\n✓ Native mixed-motif ablation completed")
    else:
        print(f"\\n⚠️  Native mixed-motif ablation failed")

print(f"\\n\\n{'='*80}")
print(f"✅ PHASE 3d COMPLETED!")
print(f"{'='*80}")
print(f"\\nMixed-motif generalization test results:")
print(f"  SAE ablation: ablations/results/ablation_*_mixed_motifs.csv")
print(f"  Native ablation: outputs/native_gnn_ablations/native_ablation_*_mixed_motifs.csv")
print(f"\\n→ Continue to PHASE 4 (Statistical Analysis)")
'''

print("This script updates the Jupyter notebook with new Phase 3a/3b/3c/3d cell code.")
print("Due to notebook JSON complexity, the code snippets below should be manually copied")
print("into the notebook cells:")
print("\n✓ Phase 3a cell: Copy phase_3a_code from this file")
print("✓ Phase 3b cell: Copy phase_3b_code from this file")
print("✓ Phase 3c cell: Copy phase_3c_code from this file")
print("✓ Phase 3d cell: Copy phase_3d_code from this file")
print("\nAlternatively, use NotebookEdit tool or Jupyter's cell editing features.")
