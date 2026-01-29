# Workflow: Adding Centrality Controls to SAE Analysis

## Overview

Your current analysis controls for **degree** only. This workflow adds **centrality measures** (betweenness, closeness, PageRank, clustering coefficient) as additional control variables.

This is important because motifs may correlate with centrality patterns (e.g., feedback loops involve central nodes), and we want to ensure SAE features capture motif semantics beyond just node importance.

## What You Have Now

Current metadata columns:
- `degree`, `in_degree`, `out_degree`, `degree_ratio` (degree features)
- Motif labels: `feedforward_loop`, `feedback_loop`, `single_input_module`, `cascade`

## What Will Be Added

New centrality columns:
- `betweenness_centrality` - How often node is on shortest paths
- `in_closeness_centrality` - How accessible node is from others
- `out_closeness_centrality` - How quickly node can reach others
- `pagerank` - Node importance (Google PageRank algorithm)
- `clustering_coefficient` - How connected node's neighbors are

## Commands to Run (In Order)

### Step 1: Add Centrality Features to Metadata
```bash
cd /data/users/goodarzilab/shervin/182-GNN_SAE
python virtual_graphs/compute_node_features_with_centrality.py
```

**What this does:**
- Reads all graph pickle files
- Computes 5 centrality measures for each node
- Adds 9 columns total (4 degree + 5 centrality) to each metadata CSV
- Takes ~5-10 minutes for ~4000 graphs

**Expected output:**
```
Computing Node-Level Features (Degree + Centrality)
Processing graphs...
  Computing betweenness centrality... ✓
  Computing closeness centrality... ✓
  Computing PageRank... ✓
  Computing clustering coefficient... ✓
100%|████████████████████████████| 4000/4000 [08:32<00:00, 7.8it/s]

✓ Successfully processed: 4000 graphs
Metadata files updated with 9 new columns
```

**Verify it worked:**
```bash
head -2 virtual_graphs/data/all_graphs/graph_motif_metadata/graph_0_metadata.csv
```

You should see new columns: `betweenness_centrality`, `pagerank`, etc.

### Step 2: Run SAE Comparison with Centrality Controls
```bash
python compare_sae_configs_with_centrality.py
```

**What this does:**
- Loads SAE models and test data
- Computes **partial correlations** controlling for degree + centrality
- Shows sanity check of topology-motif correlations
- Compares SAE configurations
- Outputs results to CSV

**Expected output:**
```
SANITY CHECK: Topology-Motif Correlations
Top 12 strongest correlations:
  feedback_loop       ~ pagerank                  : rpb= 0.682, p=1.2e-156
  feedback_loop       ~ betweenness_centrality    : rpb= 0.534, p=3.4e-89
  feedforward_loop    ~ out_closeness_centrality  : rpb= 0.423, p=2.1e-67
  ...

Breakdown by feature type:
  Degree features: max |rpb|=0.456
  Centrality features: max |rpb|=0.682

⚠ Strong confounding detected (max |rpb|=0.682)
  → Partial correlations are CRITICAL for valid interpretation
```

This quantifies how much centrality confounds motif detection!

### Step 3: Compare Results

The script will show:

**Before (degree controls only):**
- Controls for: degree, in_degree, out_degree
- May miss centrality confounds

**After (degree + centrality controls):**
- Controls for: degree, in_degree, out_degree, betweenness, closeness, PageRank, clustering
- More conservative estimates
- True motif-specific signal vs. centrality patterns

## Output Files

1. **Updated metadata files:**
   - `virtual_graphs/data/all_graphs/graph_motif_metadata/graph_*_metadata.csv`
   - Now have 9 topological feature columns

2. **Comparison results:**
   - `outputs/sae_config_comparison.csv`
   - Contains both bivariate and partial correlations
   - Column `rpb_partial` = correlation after controlling for topology

## Interpreting Results

### Sanity Check Output

```
feedback_loop ~ pagerank: rpb=0.682
```
**Means:** Feedback loop membership is strongly predicted by PageRank (r=0.68)
**Implication:** SAE features correlated with feedback loops may be detecting high-PageRank nodes rather than the loop structure itself

### Partial Correlations

**Before controls:**
```
SAE feature z45 ~ feedback_loop: rpb=0.720
```

**After controls:**
```
SAE feature z45 ~ feedback_loop: rpb_partial=0.352
```

**Interpretation:** After removing topology effects, the correlation drops from 0.72 to 0.35. This means:
- ~49% of the correlation was due to topology
- ~51% is true motif-specific signal

If `rpb_partial` stays high, the feature truly captures motif semantics!

## Key Questions Answered

### Q1: Which confounds are strongest?

Check the sanity check output:
- If centrality features have higher correlations than degree → centrality is the main confound
- If degree features are highest → degree controls were sufficient

### Q2: Do SAE features capture more than centrality?

Compare `rpb` vs `rpb_partial`:
- If `rpb_partial` is still substantial (>0.3) → Yes, captures motif-specific signal
- If `rpb_partial` drops to ~0 → No, only detecting centrality patterns

### Q3: Which SAE config is best (after controls)?

The script ranks configs by `rpb_partial` (not raw `rpb`):
```
Recommended configuration:
  latent_dim=128, k=16
  Max |rpb_partial|=0.523 (feature z112 ~ feedback_loop)
```

This is the config that captures the most motif-specific signal **after** removing topology effects.

## Comparison with Original Analysis

### Original (degree only):
```bash
python compare_sae_configs.py  # Uses degree controls only
```

### New (degree + centrality):
```bash
python compare_sae_configs_with_centrality.py  # Uses all topology controls
```

**Expected differences:**
- Centrality controls will show **lower** partial correlations (more conservative)
- But these are more **trustworthy** estimates
- Ranking of best features/configs may change

## Troubleshooting

### Error: "No topological features found"
**Fix:** Run Step 1 first (compute_node_features_with_centrality.py)

### Error: Module not found
**Fix:** Make sure you're in the right conda environment

### Slow computation
- Betweenness centrality is O(n³) - normal for large graphs
- PageRank may fail to converge - will use uniform fallback
- First graph shows detailed output, rest are in progress bar

## Time Estimates

- Step 1 (compute features): ~5-10 minutes for 4000 graphs
- Step 2 (SAE comparison): ~15-30 minutes (same as before)
- Total: ~20-40 minutes

## Next Steps After This

1. ✅ Compare degree-only vs degree+centrality results
2. ✅ Use `rpb_partial` from centrality-controlled analysis for paper
3. ✅ Update identify_top_sae_features.py to use centrality controls (if needed)
4. ✅ Update compare_sae_vs_gnnexplainer.py to show partial correlations (if needed)

## Summary

**Command sequence (assuming GNN and SAE are already trained):**

```bash
# Step 1: Add centrality features to metadata (~8 minutes)
python virtual_graphs/compute_node_features_with_centrality.py

# Step 2: Re-run SAE comparison with centrality controls (~25 minutes)
python compare_sae_configs_with_centrality.py

# Done! Check outputs/sae_config_comparison.csv
```

The key insight: **Motifs correlate strongly with centrality** (e.g., feedback loops have high PageRank). Controlling for centrality gives you the **true motif-specific signal** in your SAE features.
