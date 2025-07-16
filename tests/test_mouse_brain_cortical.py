"""
Mouse Brain Cortical Layers Analysis

This test implements one of the key "golden standard" tests mentioned in the chat files:
analyzing mouse brain cortical layers (layer 1-6) to validate spatial gradient preservation.

This addresses the specific requirement for spatial transcriptomics validation where
the ground truth is known anatomical layer organization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import pdist, squareform
import warnings
import time
import os
import sys
from datetime import datetime

# Add project paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from src.normalized_dynamics_optimized import NormalizedDynamicsOptimized

# Optional imports with fallbacks
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("UMAP not available, skipping UMAP comparisons")

try:
    import scanpy as sc
    import anndata as ad
    HAS_SCANPY = True
except ImportError:
    HAS_SCANPY = False
    print("Scanpy not available, using synthetic cortical layer data")

warnings.filterwarnings('ignore')


def create_synthetic_cortical_data(n_cells=8000, n_genes=2000, n_layers=6):
    """
    Create synthetic mouse brain cortical layer data with realistic spatial organization.
    
    This creates data that mimics the Allen Brain Atlas mouse cortical structure
    with layers 1-6 having distinct gene expression profiles and spatial arrangement.
    """
    print(f"Generating synthetic mouse brain cortical data...")
    print(f"  {n_cells} cells, {n_genes} genes, {n_layers} cortical layers")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define cortical layer characteristics based on mouse brain anatomy
    layer_info = {
        'Layer1': {
            'depth_range': (0.0, 0.1),    # Molecular layer (superficial)
            'cell_density': 0.05,          # Low cell density
            'marker_genes': ['Reln', 'Cxcl14', 'Ndnf'],  # Known layer 1 markers
            'thickness': 0.1
        },
        'Layer2/3': {
            'depth_range': (0.1, 0.4),    # Granular/pyramidal layers
            'cell_density': 0.35,          # High cell density
            'marker_genes': ['Cux2', 'Satb2', 'Rorb'],   # Layer 2/3 markers
            'thickness': 0.3
        },
        'Layer4': {
            'depth_range': (0.4, 0.55),   # Internal granular layer
            'cell_density': 0.25,          # Medium-high density
            'marker_genes': ['Rorb', 'Scnn1a', 'Rspo1'], # Layer 4 markers
            'thickness': 0.15
        },
        'Layer5': {
            'depth_range': (0.55, 0.75),  # Internal pyramidal layer
            'cell_density': 0.25,          # Medium density
            'marker_genes': ['Bcl11b', 'Etv1', 'Fezf2'], # Layer 5 markers
            'thickness': 0.2
        },
        'Layer6': {
            'depth_range': (0.75, 1.0),   # Multiform layer (deep)
            'cell_density': 0.10,          # Lower density
            'marker_genes': ['Foxp2', 'Ctgf', 'Syt6'],   # Layer 6 markers
            'thickness': 0.25
        }
    }
    
    # Calculate cells per layer based on density
    layer_names = list(layer_info.keys())
    layer_densities = [layer_info[layer]['cell_density'] for layer in layer_names]
    total_density = sum(layer_densities)
    
    cells_per_layer = {}
    cell_assignments = []
    spatial_coords = []
    depth_values = []
    
    for i, layer in enumerate(layer_names):
        n_layer_cells = int(n_cells * layer_densities[i] / total_density)
        cells_per_layer[layer] = n_layer_cells
        
        # Generate spatial coordinates for this layer
        depth_min, depth_max = layer_info[layer]['depth_range']
        
        for _ in range(n_layer_cells):
            # Cortical depth (0 = surface, 1 = white matter)
            depth = np.random.uniform(depth_min, depth_max)
            depth_values.append(depth)
            
            # Lateral coordinates (simulate cortical surface area)
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            
            spatial_coords.append([x, y, depth])
            cell_assignments.append(layer)
    
    # Adjust if we have too few/many cells due to rounding
    current_total = len(cell_assignments)
    if current_total < n_cells:
        # Add cells to Layer2/3 (most populous)
        layer = 'Layer2/3'
        depth_min, depth_max = layer_info[layer]['depth_range']
        for _ in range(n_cells - current_total):
            depth = np.random.uniform(depth_min, depth_max)
            depth_values.append(depth)
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-1, 1)
            spatial_coords.append([x, y, depth])
            cell_assignments.append(layer)
    elif current_total > n_cells:
        # Remove excess cells
        indices_to_keep = np.random.choice(current_total, n_cells, replace=False)
        cell_assignments = [cell_assignments[i] for i in indices_to_keep]
        spatial_coords = [spatial_coords[i] for i in indices_to_keep]
        depth_values = [depth_values[i] for i in indices_to_keep]
    
    spatial_coords = np.array(spatial_coords)
    depth_values = np.array(depth_values)
    
    print(f"  Layer distribution:")
    layer_counts = {}
    for layer in layer_names:
        count = cell_assignments.count(layer)
        layer_counts[layer] = count
        print(f"    {layer}: {count} cells ({count/len(cell_assignments)*100:.1f}%)")
    
    # Generate gene expression data with layer-specific patterns
    X = np.random.lognormal(0, 1, (n_cells, n_genes))
    
    # Add layer-specific gene expression patterns
    for i, layer in enumerate(cell_assignments):
        cell_idx = i
        layer_depth = depth_values[i]
        
        # Layer-specific expression modifications
        if layer == 'Layer1':
            # Molecular layer - sparse, specific markers
            X[cell_idx, :100] *= 0.5  # Generally lower expression
            X[cell_idx, 10:13] *= 5   # Specific markers highly expressed
            
        elif layer == 'Layer2/3':
            # Superficial pyramidal - high activity
            X[cell_idx, :] *= 1.2      # Generally higher expression
            X[cell_idx, 50:53] *= 4    # Layer-specific markers
            
        elif layer == 'Layer4':
            # Granular layer - distinct profile
            X[cell_idx, 100:103] *= 6  # Strong layer 4 markers
            X[cell_idx, :50] *= 0.8    # Moderate other expression
            
        elif layer == 'Layer5':
            # Deep pyramidal - projection neurons
            X[cell_idx, 150:153] *= 5  # Layer 5 specific
            X[cell_idx, 200:250] *= 1.5  # Projection-related genes
            
        elif layer == 'Layer6':
            # Multiform layer - diverse cell types
            X[cell_idx, 300:303] *= 4  # Layer 6 markers
            X[cell_idx, :] *= np.random.uniform(0.7, 1.3)  # Variable expression
    
    # Add depth-dependent gradient
    for i in range(n_cells):
        depth = depth_values[i]
        # Create smooth depth gradient for certain genes
        gradient_genes = range(500, 600)  # 100 genes with depth gradient
        for gene_idx in gradient_genes:
            gradient_factor = 1 + 2 * depth  # Increases with depth
            X[i, gene_idx] *= gradient_factor
    
    # Add noise and ensure positive values
    X += np.random.lognormal(0, 0.1, X.shape)
    X = np.maximum(X, 0.01)  # Ensure no zeros
    
    # Create additional metadata
    metadata = {
        'layer_info': layer_info,
        'layer_counts': layer_counts,
        'spatial_coords': spatial_coords,
        'depth_values': depth_values,
        'cell_assignments': cell_assignments
    }
    
    print(f"✅ Synthetic cortical data generated successfully")
    print(f"  Spatial coordinates: {spatial_coords.shape}")
    print(f"  Depth range: {depth_values.min():.3f} - {depth_values.max():.3f}")
    
    return X, cell_assignments, metadata


def compute_spatial_gradient_preservation(embedding, spatial_coords, depth_values, verbose=True):
    """
    Compute spatial gradient preservation metrics for cortical layer analysis.
    
    This is the key novel metric mentioned in the chat files for evaluating
    how well algorithms preserve smooth transitions between cortical layers.
    """
    if verbose:
        print("Computing spatial gradient preservation metrics...")
    
    n_cells = len(embedding)
    
    # 1. Depth Correlation Analysis
    # How well does the embedding preserve cortical depth information?
    depth_corr_metrics = {}
    
    for dim in range(embedding.shape[1]):
        # Correlation between embedding dimension and cortical depth
        corr_pearson, p_pearson = pearsonr(embedding[:, dim], depth_values)
        corr_spearman, p_spearman = spearmanr(embedding[:, dim], depth_values)
        
        depth_corr_metrics[f'dim_{dim}_pearson'] = abs(corr_pearson)
        depth_corr_metrics[f'dim_{dim}_spearman'] = abs(corr_spearman)
    
    # Best correlation across dimensions (some dimension should capture depth)
    max_pearson = max([depth_corr_metrics[k] for k in depth_corr_metrics.keys() if 'pearson' in k])
    max_spearman = max([depth_corr_metrics[k] for k in depth_corr_metrics.keys() if 'spearman' in k])
    
    # 2. Spatial Continuity Analysis
    # Do nearby points in space remain nearby in embedding?
    spatial_distances = pdist(spatial_coords)
    embedding_distances = pdist(embedding)
    
    spatial_continuity, _ = spearmanr(spatial_distances, embedding_distances)
    spatial_continuity = max(0, spatial_continuity)  # Ensure non-negative
    
    # 3. Layer Gradient Smoothness
    # Are transitions between layers smooth rather than abrupt?
    gradient_smoothness_scores = []
    
    # Sample points along depth gradient
    depth_samples = np.linspace(0.05, 0.95, 20)
    
    for i in range(len(depth_samples) - 1):
        # Find cells in adjacent depth windows
        depth_window_1 = (depth_values >= depth_samples[i]) & (depth_values < depth_samples[i] + 0.1)
        depth_window_2 = (depth_values >= depth_samples[i+1]) & (depth_values < depth_samples[i+1] + 0.1)
        
        if np.sum(depth_window_1) > 5 and np.sum(depth_window_2) > 5:
            # Compute distance between window centroids in embedding
            centroid_1 = np.mean(embedding[depth_window_1], axis=0)
            centroid_2 = np.mean(embedding[depth_window_2], axis=0)
            embedding_dist = np.linalg.norm(centroid_2 - centroid_1)
            
            # Compute expected distance based on depth difference
            depth_diff = abs(depth_samples[i+1] - depth_samples[i])
            
            # Smoothness score: smaller embedding distance for small depth differences
            smoothness = 1.0 / (1.0 + embedding_dist / (depth_diff + 0.01))
            gradient_smoothness_scores.append(smoothness)
    
    avg_gradient_smoothness = np.mean(gradient_smoothness_scores) if gradient_smoothness_scores else 0.0
    
    # 4. Layer Boundary Preservation
    # Are layer boundaries preserved without artificial clustering?
    layer_boundary_score = 0.0
    
    # Define layer boundaries based on depth
    layer_boundaries = [0.1, 0.4, 0.55, 0.75]  # Boundaries between layers
    
    boundary_preservation_scores = []
    for boundary in layer_boundaries:
        # Find cells near this boundary
        near_boundary = np.abs(depth_values - boundary) < 0.05
        if np.sum(near_boundary) > 10:
            # Compute local variance in embedding space
            local_embedding = embedding[near_boundary]
            local_variance = np.mean(np.var(local_embedding, axis=0))
            
            # Good preservation = low variance (smooth transition)
            boundary_score = 1.0 / (1.0 + local_variance)
            boundary_preservation_scores.append(boundary_score)
    
    layer_boundary_score = np.mean(boundary_preservation_scores) if boundary_preservation_scores else 0.0
    
    # 5. Overall Spatial Gradient Preservation Score
    spatial_gradient_preservation = (
        0.3 * max_spearman +           # Depth correlation
        0.3 * spatial_continuity +     # Spatial continuity
        0.25 * avg_gradient_smoothness + # Gradient smoothness
        0.15 * layer_boundary_score    # Boundary preservation
    )
    
    metrics = {
        'spatial_gradient_preservation': spatial_gradient_preservation,
        'depth_correlation_pearson': max_pearson,
        'depth_correlation_spearman': max_spearman,
        'spatial_continuity': spatial_continuity,
        'gradient_smoothness': avg_gradient_smoothness,
        'layer_boundary_preservation': layer_boundary_score,
        'n_gradient_samples': len(gradient_smoothness_scores),
        'n_boundary_samples': len(boundary_preservation_scores)
    }
    
    if verbose:
        print(f"  Spatial gradient preservation: {spatial_gradient_preservation:.3f}")
        print(f"  Depth correlation (Spearman): {max_spearman:.3f}")
        print(f"  Spatial continuity: {spatial_continuity:.3f}")
        print(f"  Gradient smoothness: {avg_gradient_smoothness:.3f}")
        print(f"  Layer boundary preservation: {layer_boundary_score:.3f}")
    
    return metrics


def compute_cortical_layer_metrics(embedding, layer_assignments, spatial_coords, depth_values):
    """
    Compute comprehensive metrics for cortical layer analysis.
    """
    print("Computing cortical layer validation metrics...")
    
    # 1. Spatial gradient preservation (novel metric from chat suggestions)
    spatial_metrics = compute_spatial_gradient_preservation(
        embedding, spatial_coords, depth_values, verbose=True
    )
    
    # 2. Layer separation analysis
    unique_layers = list(set(layer_assignments))
    layer_to_idx = {layer: i for i, layer in enumerate(unique_layers)}
    layer_indices = [layer_to_idx[layer] for layer in layer_assignments]
    
    # Adjusted Rand Index for layer clustering
    from sklearn.cluster import KMeans
    
    # Use k-means to find clusters in embedding
    n_layers = len(unique_layers)
    kmeans = KMeans(n_clusters=n_layers, random_state=42, n_init=10)
    predicted_clusters = kmeans.fit_predict(embedding)
    
    ari_score = adjusted_rand_score(layer_indices, predicted_clusters)
    nmi_score = normalized_mutual_info_score(layer_indices, predicted_clusters)
    
    # 3. Layer ordering preservation
    # Check if layers are ordered correctly by depth
    layer_depth_means = {}
    for layer in unique_layers:
        layer_mask = np.array(layer_assignments) == layer
        layer_depth_means[layer] = np.mean(np.array(depth_values)[layer_mask])
    
    # Expected layer order by depth (surface to deep)
    expected_order = sorted(layer_depth_means.items(), key=lambda x: x[1])
    
    # Compute layer ordering in embedding space
    layer_embedding_means = {}
    for layer in unique_layers:
        layer_mask = np.array(layer_assignments) == layer
        layer_embedding_means[layer] = np.mean(embedding[layer_mask], axis=0)
    
    # Project layer means to 1D for ordering analysis
    from sklearn.decomposition import PCA
    layer_coords = np.array(list(layer_embedding_means.values()))
    pca = PCA(n_components=1)
    layer_coords_1d = pca.fit_transform(layer_coords).flatten()
    
    # Create mapping from layer to 1D coordinate
    layer_order_embedding = {}
    for i, layer in enumerate(layer_embedding_means.keys()):
        layer_order_embedding[layer] = layer_coords_1d[i]
    
    # Compare expected vs actual ordering
    expected_depths = [layer_depth_means[layer] for layer, _ in expected_order]
    actual_coords = [layer_order_embedding[layer] for layer, _ in expected_order]
    
    ordering_correlation, _ = spearmanr(expected_depths, actual_coords)
    ordering_correlation = abs(ordering_correlation)  # Take absolute value
    
    # 4. Combine all metrics
    cortical_metrics = {
        'spatial_gradient_preservation': spatial_metrics['spatial_gradient_preservation'],
        'layer_separation_ari': ari_score,
        'layer_separation_nmi': nmi_score,
        'layer_ordering_correlation': ordering_correlation,
        'depth_correlation': spatial_metrics['depth_correlation_spearman'],
        'spatial_continuity': spatial_metrics['spatial_continuity'],
        'gradient_smoothness': spatial_metrics['gradient_smoothness'],
        'layer_boundary_preservation': spatial_metrics['layer_boundary_preservation']
    }
    
    # Overall cortical preservation score
    cortical_preservation_score = (
        0.4 * spatial_metrics['spatial_gradient_preservation'] +
        0.2 * ari_score +
        0.2 * ordering_correlation +
        0.2 * spatial_metrics['spatial_continuity']
    )
    
    cortical_metrics['cortical_preservation_score'] = cortical_preservation_score
    
    print(f"  Layer separation (ARI): {ari_score:.3f}")
    print(f"  Layer ordering correlation: {ordering_correlation:.3f}")
    print(f"  Overall cortical preservation: {cortical_preservation_score:.3f}")
    
    return cortical_metrics


def run_cortical_analysis(algorithm, X, layer_assignments, spatial_coords, depth_values, algorithm_name):
    """Run analysis for a single algorithm."""
    print(f"\n🧠 Running {algorithm_name} analysis...")
    
    start_time = time.time()
    
    try:
        embedding = algorithm.fit_transform(X)
        runtime = time.time() - start_time
        
        # Compute cortical-specific metrics
        metrics = compute_cortical_layer_metrics(
            embedding, layer_assignments, spatial_coords, depth_values
        )
        
        return {
            'embedding': embedding,
            'metrics': metrics,
            'runtime': runtime,
            'success': True,
            'error': None
        }
        
    except Exception as e:
        runtime = time.time() - start_time
        print(f"  ❌ {algorithm_name} failed: {e}")
        return {
            'embedding': None,
            'metrics': {},
            'runtime': runtime,
            'success': False,
            'error': str(e)
        }


def create_cortical_visualization(results, layer_assignments, depth_values, save_path=None):
    """Create comprehensive visualization of cortical layer analysis."""
    print("Creating cortical layer visualization...")
    
    # Create figure with subplots
    n_algorithms = len([r for r in results.values() if r['success']])
    fig, axes = plt.subplots(2, max(2, n_algorithms), figsize=(5*n_algorithms, 10))
    if n_algorithms == 1:
        axes = axes.reshape(2, 1)
    
    # Color map for layers
    unique_layers = sorted(list(set(layer_assignments)))
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_layers)))
    layer_colors = dict(zip(unique_layers, colors))
    
    # Plot embeddings (top row)
    plot_idx = 0
    for alg_name, result in results.items():
        if not result['success']:
            continue
            
        embedding = result['embedding']
        metrics = result['metrics']
        
        ax = axes[0, plot_idx] if n_algorithms > 1 else axes[0]
        
        # Plot points colored by layer
        for i, layer in enumerate(unique_layers):
            layer_mask = np.array(layer_assignments) == layer
            if np.any(layer_mask):
                ax.scatter(embedding[layer_mask, 0], embedding[layer_mask, 1], 
                          c=[layer_colors[layer]], label=layer, alpha=0.6, s=20)
        
        ax.set_title(f'{alg_name}\nCortical Preservation: {metrics.get("cortical_preservation_score", 0):.3f}')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        if plot_idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plot_idx += 1
    
    # Plot depth gradient visualization (bottom row)
    plot_idx = 0
    for alg_name, result in results.items():
        if not result['success']:
            continue
            
        embedding = result['embedding']
        metrics = result['metrics']
        
        ax = axes[1, plot_idx] if n_algorithms > 1 else axes[1]
        
        # Plot points colored by cortical depth
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], 
                           c=depth_values, cmap='plasma', alpha=0.7, s=20)
        
        ax.set_title(f'{alg_name}\nDepth Correlation: {metrics.get("depth_correlation", 0):.3f}')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        
        # Add colorbar for depth
        if plot_idx == n_algorithms - 1:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Cortical Depth (0=surface, 1=deep)')
        
        plot_idx += 1
    
    # Remove empty subplots
    for i in range(plot_idx, axes.shape[1]):
        if n_algorithms > 1:
            axes[0, i].remove()
            axes[1, i].remove()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Visualization saved: {save_path}")
    
    return fig


def run_and_visualize_mouse_brain_cortical():
    """
    Main function to run the mouse brain cortical layers analysis.
    
    This implements the "Mouse Brain Cortical Layers" golden standard test
    mentioned in the chat files with spatial gradient preservation metrics.
    """
    print("=" * 80)
    print("MOUSE BRAIN CORTICAL LAYERS ANALYSIS")
    print("Golden Standard Test: Spatial Gradient Preservation")
    print("=" * 80)
    
    # Generate synthetic cortical data
    X, layer_assignments, metadata = create_synthetic_cortical_data(
        n_cells=6000, n_genes=2000, n_layers=5
    )
    spatial_coords = metadata['spatial_coords']
    depth_values = metadata['depth_values']
    
    # Standardize data
    print(f"\nStandardizing expression data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"Data shape: {X_scaled.shape}")
    
    # Initialize algorithms
    algorithms = {
        'NormalizedDynamics': NormalizedDynamicsOptimized(
            dim=2, max_iter=30, adaptive_params=True
        ),
        't-SNE': TSNE(
            n_components=2, random_state=42, max_iter=1000, 
            perplexity=min(30, len(X_scaled)//4)
        )
    }
    
    if HAS_UMAP:
        algorithms['UMAP'] = umap.UMAP(
            n_components=2, random_state=42, 
            n_neighbors=min(15, len(X_scaled)//4), min_dist=0.1
        )
    
    # Run analysis for each algorithm
    results = {}
    for alg_name, algorithm in algorithms.items():
        results[alg_name] = run_cortical_analysis(
            algorithm, X_scaled, layer_assignments, spatial_coords, depth_values, alg_name
        )
    
    # Create visualization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(project_root, 'static', 'results', 
                            f'mouse_brain_cortical_comparison_{timestamp}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig = create_cortical_visualization(results, layer_assignments, depth_values, save_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print("MOUSE BRAIN CORTICAL ANALYSIS SUMMARY")
    print("=" * 60)
    
    print(f"\n{'Algorithm':<20} | {'Runtime (s)':<12} | {'Spatial Gradient':<15} | {'Cortical Score':<15}")
    print("-" * 70)
    
    for alg_name, result in results.items():
        if result['success']:
            runtime = result['runtime']
            spatial_grad = result['metrics'].get('spatial_gradient_preservation', 0)
            cortical_score = result['metrics'].get('cortical_preservation_score', 0)
            
            print(f"{alg_name:<20} | {runtime:8.2f}    | {spatial_grad:10.3f}    | {cortical_score:10.3f}")
        else:
            print(f"{alg_name:<20} | {'FAILED':<12} | {'N/A':<15} | {'N/A':<15}")
    
    # Detailed analysis
    print(f"\n📊 SPATIAL GRADIENT PRESERVATION ANALYSIS:")
    
    best_spatial = max([r['metrics'].get('spatial_gradient_preservation', 0) 
                       for r in results.values() if r['success']])
    
    for alg_name, result in results.items():
        if result['success']:
            metrics = result['metrics']
            spatial_score = metrics.get('spatial_gradient_preservation', 0)
            
            print(f"\n🧠 {alg_name}:")
            print(f"   • Spatial gradient preservation: {spatial_score:.3f}")
            print(f"   • Depth correlation: {metrics.get('depth_correlation', 0):.3f}")
            print(f"   • Layer separation (ARI): {metrics.get('layer_separation_ari', 0):.3f}")
            print(f"   • Layer ordering: {metrics.get('layer_ordering_correlation', 0):.3f}")
            
            if spatial_score == best_spatial:
                print(f"   ⭐ BEST spatial gradient preservation!")
    
    print(f"\n🎯 KEY FINDINGS:")
    print(f"   • This test validates spatial transcriptomics analysis capability")
    print(f"   • Spatial gradient preservation measures smooth layer transitions")
    print(f"   • Results show algorithm performance on anatomical structure preservation")
    
    # Return results for further analysis
    rel_save_path = os.path.relpath(save_path, project_root)
    return rel_save_path, results, metadata


if __name__ == "__main__":
    # Run the analysis
    try:
        image_path, results, metadata = run_and_visualize_mouse_brain_cortical()
        print(f"\n✅ Mouse brain cortical analysis completed successfully!")
        print(f"📊 Results visualization: {image_path}")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc() 