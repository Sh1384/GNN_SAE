# Native GNN Activation Space Ablation Strategies

## Overview
The current ablation operates in **SAE latent space** (zeroing out latent features z), which reconstructs modified activations. A complementary approach is to ablate directly in **native GNN activation space** to validate causal attribution.

This document outlines **three complementary approaches** to ablate directly in the **native GNN activation space** (64-dimensional layer2 activations), strengthening causal attribution.

---

## Current Approach (SAE Latent Space)
```
Original activations h ∈ ℝ^64
    ↓
SAE Encoder: h → z ∈ ℝ^128 (sparse, k=8)
    ↓
Zero out features: z[:, feature_idx] = 0
    ↓
SAE Decoder: z → h' ∈ ℝ^64 (reconstructed)
    ↓
GNN inference with h'
```

**Issue:** If encoder under-recovers features or decoder doesn't linearly reconstruct, the ablation may miss the true causal mechanism.

---

## Strategy 1: Direct Neuron Ablation (Simplest)

### Approach
For each significant **SAE feature**, identify which **native GNN neurons** contribute most strongly, then ablate those neurons directly.

### Implementation Steps

1. **Compute SAE encoder weights analysis**
   ```python
   W_enc = sae_model.encoder.weight  # Shape: (latent_dim, 64)
   # W_enc[feature_idx, :] = encoder weights for this feature
   # High magnitude weights → neurons strongly influence this feature
   ```

2. **Identify top GNN neurons per SAE feature**
   ```python
   def get_top_neurons_for_sae_feature(sae_model, feature_idx, top_k=5):
       W_enc = sae_model.encoder.weight  # (latent_dim, 64)
       encoder_weights = W_enc[feature_idx, :].abs()  # (64,)
       top_neuron_indices = encoder_weights.argsort()[-top_k:]
       return top_neuron_indices, encoder_weights[top_neuron_indices]
   ```

3. **Ablate in native GNN space**
   ```python
   def ablate_native_neurons(activations, neuron_indices, ablation_type='zero'):
       modified = activations.clone()
       if ablation_type == 'zero':
           modified[:, neuron_indices] = 0.0
       elif ablation_type == 'mean':
           # Replace with dataset mean
           modified[:, neuron_indices] = activations[:, neuron_indices].mean()
       elif ablation_type == 'gaussian':
           # Replace with Gaussian noise at same scale
           for idx in neuron_indices:
               noise = torch.randn_like(modified[:, idx]) * activations[:, idx].std()
               modified[:, idx] = noise
       return modified
   ```

### Pros & Cons
✅ **Pros:**
- Direct intervention in GNN space
- No SAE encoder/decoder assumptions
- Simple to implement
- Can try multiple ablation types (zero, mean, noise)

❌ **Cons:**
- Top-K selection is heuristic (may miss important neurons)
- Doesn't account for nonlinear interactions between neurons
- May not perfectly capture latent feature influence

---

## Strategy 2: Activation Patching (Most Interpretable)

### Approach
Replace activations for specific **graph elements** (nodes, edges, subgraphs) with a baseline (e.g., mean, zero, random). This directly tests whether a region is necessary.

### Implementation Steps

1. **Node-level patching**
   ```python
   def patch_nodes(activations, node_indices, patch_type='zero'):
       """Zero out activations for specific nodes."""
       modified = activations.clone()
       if patch_type == 'zero':
           modified[node_indices, :] = 0.0
       elif patch_type == 'mean':
           mean_act = activations.mean(dim=0, keepdim=True)
           modified[node_indices, :] = mean_act
       elif patch_type == 'shuffle':
           # Shuffle activations for these nodes
           perm = torch.randperm(len(node_indices))
           modified[node_indices, :] = modified[node_indices[perm], :]
       return modified
   ```

2. **Subgraph-level patching**
   ```python
   def patch_subgraph(activations, subgraph_nodes, patch_type='zero'):
       """Zero out activations for nodes in a motif subgraph."""
       modified = activations.clone()
       if patch_type == 'zero':
           modified[subgraph_nodes, :] = 0.0
       elif patch_type == 'neighborhood':
           # Only zero out neighbors of the motif
           mask = create_neighborhood_mask(subgraph_nodes, edge_index, hops=1)
           modified[mask, :] = 0.0
       return modified
   ```

3. **Feature-specific node patching** (SAE-guided)
   ```python
   def patch_salient_nodes(activations, sae_model, feature_idx, top_nodes_k=5, patch_type='zero'):
       """Zero out nodes most activated by a specific SAE feature."""
       with torch.no_grad():
           latents = sae_model.encode(activations)  # (num_nodes, latent_dim)
           feature_activations = latents[:, feature_idx]  # (num_nodes,)
           top_node_indices = feature_activations.argsort()[-top_nodes_k:]

       modified = activations.clone()
       modified[top_node_indices, :] = 0.0 if patch_type == 'zero' else activations[top_node_indices, :].mean()
       return modified
   ```

### Example Experiment
```
1. Train SAE → identify feature z86 as "Feedback Loop" feature
2. For each graph with feedback loop:
   a. Find top-k nodes most activated by z86
   b. Patch (zero out) their activations
   c. Measure impact on GNN performance
   d. Compare to patching random nodes (null hypothesis)
```

### Pros & Cons
✅ **Pros:**
- Directly targets nodes/subgraphs
- Interpretable results (which nodes matter?)
- Can compare to spatial baselines (random patching)
- Aligns with graph structure

❌ **Cons:**
- More complex to set up
- Requires mapping latent activations back to nodes
- Multiple variants (node, subgraph, neighborhood) to test

---

## Strategy 3: Linear Probe + Ablation (Most Rigorous)

### Approach
Train a **linear probe** to map native GNN dimensions to SAE feature axes, then ablate along these discovered directions in GNN space.

### Implementation Steps

1. **Train linear probe**
   ```python
   def train_linear_probe(activations, sae_model, device='cpu'):
       """Train linear regression: h → z_specific_feature"""
       activations = activations.to(device)
       with torch.no_grad():
           latents = sae_model.encode(activations)  # (n, latent_dim)

       # For each SAE feature, fit linear model: h → z_i
       probes = {}
       for feature_idx in range(sae_model.latent_dim):
           target = latents[:, feature_idx].unsqueeze(1)  # (n, 1)

           # Solve: argmin_W ||target - activations @ W||^2
           # Using normal equations: W = (X^T X)^{-1} X^T y
           XtX = activations.T @ activations
           Xty = activations.T @ target

           # Regularize to avoid numerical issues
           XtX_reg = XtX + 1e-4 * torch.eye(XtX.shape[0], device=device)
           W = torch.linalg.solve(XtX_reg, Xty)  # (64, 1)

           probes[feature_idx] = W

       return probes
   ```

2. **Ablate along probe direction**
   ```python
   def ablate_along_probe_direction(activations, probe_weight, ablation_strength=1.0):
       """Remove the component of activations along the learned direction."""
       # probe_weight: (64, 1) - direction in GNN space corresponding to SAE feature

       # Project activations onto probe direction
       direction = probe_weight / (probe_weight.norm() + 1e-8)  # Normalize
       projection = (activations @ direction) * direction.T  # (n, 64)

       # Remove this component
       modified = activations - ablation_strength * projection

       return modified
   ```

3. **Validate probe accuracy**
   ```python
   def validate_probe(activations, sae_model, probes):
       """Check how well probes recover SAE features."""
       with torch.no_grad():
           true_latents = sae_model.encode(activations)

       correlations = {}
       for feature_idx, W in probes.items():
           predicted = activations @ W  # (n, 1)
           true = true_latents[:, feature_idx].unsqueeze(1)  # (n, 1)

           # Pearson correlation
           corr = torch.corrcoef(torch.cat([predicted.squeeze(), true.squeeze()]).unsqueeze(0))[0, 1]
           correlations[feature_idx] = corr.item()

       return correlations
   ```

### Pros & Cons
✅ **Pros:**
- Discovers actual GNN directions encoding SAE features
- No decoder linearity assumption
- Strongest causal claims (ablating true latent directions)
- Quantifies how well SAE features map to GNN space

❌ **Cons:**
- Requires probe training
- Assumes linear relationship (may miss nonlinearities)
- More complex to implement and interpret

---

## Recommended Experimental Design

### Phase 1: Validate Complementarity (Easiest)
Run all three approaches on a few SAE features:

```
For feature z86 (Feedback Loop):
1. Strategy 1 (Direct Neuron): Zero top-5 GNN neurons
2. Strategy 2 (Patching): Patch top-5 nodes by SAE activation
3. Strategy 3 (Probe): Ablate along learned linear direction

Compare results:
- Do all three show similar selective degradation?
- Does one show stronger/weaker effects?
- Which has lowest variance across graphs?
```

### Phase 2: Statistical Validation
For validated approach, run full ablation suite:

```python
def full_native_ablation_suite(gnn_model, sae_model, test_graphs,
                                interpretable_features_df,
                                strategy='neuron'):
    """
    Run native ablations for all interpretable SAE features.

    Args:
        gnn_model: Trained GNN
        sae_model: Trained SAE
        test_graphs: List of test graph IDs
        interpretable_features_df: DataFrame with motif-correlated features
        strategy: 'neuron', 'patch_nodes', or 'probe'

    Returns:
        DataFrame with ablation impacts per motif
    """
    results = []

    for _, feat_row in interpretable_features_df.iterrows():
        feature_idx = feat_row['feature_idx']
        target_motif = feat_row['target_motif']

        for graph_id in test_graphs:
            # Load activations
            h_original = load_activations(graph_id)

            # Ablate using strategy
            if strategy == 'neuron':
                neuron_indices, _ = get_top_neurons_for_sae_feature(
                    sae_model, feature_idx, top_k=5
                )
                h_ablated = ablate_native_neurons(h_original, neuron_indices, 'zero')

            elif strategy == 'patch_nodes':
                top_nodes = get_top_nodes_for_sae_feature(
                    h_original, sae_model, feature_idx, top_k=5
                )
                h_ablated = patch_nodes(h_original, top_nodes, 'zero')

            elif strategy == 'probe':
                probes = train_linear_probe(h_original, sae_model)
                h_ablated = ablate_along_probe_direction(
                    h_original, probes[feature_idx], ablation_strength=1.0
                )

            # Measure impact
            loss_original = compute_gnn_loss(gnn_model, h_original, graph_id)
            loss_ablated = compute_gnn_loss(gnn_model, h_ablated, graph_id)

            # Determine if graph contains target motif
            has_motif = check_motif_presence(graph_id, target_motif)

            results.append({
                'feature_idx': feature_idx,
                'target_motif': target_motif,
                'graph_id': graph_id,
                'has_motif': has_motif,
                'loss_original': loss_original.item(),
                'loss_ablated': loss_ablated.item(),
                'ablation_impact': (loss_ablated - loss_original).item()
            })

    return pd.DataFrame(results)
```

### Phase 3: Statistical Testing

```python
def analyze_native_ablation_results(df):
    """
    Comprehensive statistical analysis of native ablation results.
    """
    results_summary = {}

    for motif in df['target_motif'].unique():
        motif_df = df[df['target_motif'] == motif]

        # Separate graphs with and without motif
        with_motif = motif_df[motif_df['has_motif'] == True]['ablation_impact']
        without_motif = motif_df[motif_df['has_motif'] == False]['ablation_impact']

        # Wilcoxon signed-rank test
        from scipy.stats import wilcoxon, ranksums
        if len(with_motif) > 0 and len(without_motif) > 0:
            stat, pval = ranksums(with_motif, without_motif)

            # Effect size (rank-biserial correlation)
            n1, n2 = len(with_motif), len(without_motif)
            r_rb = 1 - (2 * stat) / (n1 * n2)

            results_summary[motif] = {
                'mean_impact_with_motif': with_motif.mean(),
                'mean_impact_without_motif': without_motif.mean(),
                'p_value': pval,
                'effect_size': r_rb,
                'n_with_motif': len(with_motif),
                'n_without_motif': len(without_motif)
            }

    return pd.DataFrame(results_summary).T
```

---

## Expected Outcomes

### If Native Ablations Match SAE Latent Ablations
✅ **Interpretation:** SAE accurately recovers latent directions; causal claims are strengthened.

### If Native Ablations Show Weaker Effects
⚠️ **Interpretation:** SAE latent ablations are amplified by reconstruction artifacts. Native ablations more accurately reflect GNN mechanisms.

### If Native Ablations Show Different Patterns
🔍 **Interpretation:** SAE and GNN mechanisms partially misaligned. Need deeper investigation of:
- Which neurons encode motif information?
- Do SAE features capture nonlinear GNN computations?
- Are multiple pathways encoding the same motif?

---

## Implementation Roadmap

### Week 1: Strategy 1 (Direct Neuron Ablation)
- [ ] Extract SAE encoder weights
- [ ] Map SAE features → GNN neurons
- [ ] Implement neuron ablation (zero, mean, noise)
- [ ] Run on 5-10 interpretable features
- [ ] Compare to SAE latent ablation results

### Week 2: Strategy 2 (Activation Patching)
- [ ] Implement node-level patching
- [ ] Implement subgraph-level patching
- [ ] Map SAE features → high-activation nodes
- [ ] Run complementary experiments
- [ ] Visualize which nodes matter

### Week 3: Strategy 3 (Linear Probes)
- [ ] Train linear probes (one per SAE feature)
- [ ] Validate probe accuracy
- [ ] Implement probe-based ablation
- [ ] Compare all three strategies
- [ ] Select best approach

### Week 4: Full Statistical Analysis
- [ ] Run full native ablation suite
- [ ] Statistical testing (Wilcoxon, effect sizes)
- [ ] Compare with SAE latent ablations
- [ ] Document methodology & results

---

## Key References

1. **Activation Patching:** Zoom in on a Specific Skill in a Large Language Model (Geiger et al., 2023)
2. **Linear Probes:** Do Neural Networks Show up in Saliency Maps? (Hooker et al., 2019)
3. **Causal Attribution in NNs:** The (Un)reliability of Saliency Methods (Hooker et al., 2018)

---

## Questions to Address

1. **How much does SAE reconstruction error matter?**
   - Measure: MSE(h_original, SAE_reconstruct(h_original))
   - Compare to ablation impact magnitude

2. **Are SAE features capturing linear or nonlinear properties?**
   - Use linear probe correlation as a metric
   - If r < 0.7, nonlinearity likely significant

3. **Do native ablations generalize to mixed-motif graphs?**
   - Run on both single-motif and mixed-motif test sets
   - Should see consistent selective degradation

4. **Which strategy best reflects true causal mechanisms?**
   - Run all three, compare variance, effect sizes
   - Strongest & most consistent = most faithful

