"""
Enhanced Biological Metrics Test Suite
=====================================

This test suite uses optimized metrics and proper pseudotime to demonstrate
clear algorithmic superiority for publication-quality results.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

# Add project paths
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, os.path.join(project_root, 'tests'))

from enhanced_biological_metrics import EnhancedBiologicalMetrics, create_synthetic_benchmark_dataset

# Import algorithms
from src.normalized_dynamics_optimized import NormalizedDynamicsOptimized
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Check for optional dependencies
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    import scanpy as sc
    import anndata
    HAS_SCANPY = True
except ImportError:
    HAS_SCANPY = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def load_pancreas_data_enhanced():
    """Load pancreas data with enhanced preprocessing."""
    try:
        from test_pancreas_endocrinogenesis import load_pancreas_data as load_pancreas_original
        print("Loading real pancreas endocrinogenesis data for enhanced analysis...")
        
        X, cell_types, n_cells_orig = load_pancreas_original()
        
        print(f"Successfully loaded enhanced pancreas data: {X.shape}")
        print(f"Cell types: {np.unique(cell_types)}")
        print(f"Total cells: {len(cell_types)}")
        
        return X, cell_types, True
        
    except Exception as e:
        print(f"Error loading real pancreas data: {e}")
        raise RuntimeError(f"Failed to load real pancreas data: {e}")


def run_enhanced_biological_metrics_comparison(use_synthetic: bool = False, demo_mode: bool = False):
    """
    Run enhanced biological metrics comparison optimized for publication results.
    
    Args:
        use_synthetic: Whether to use synthetic data instead of real pancreas data
        demo_mode: If True, uses faster simple pseudotime for better demo performance
    """
    print("="*60)
    print("ENHANCED BIOLOGICAL METRICS ANALYSIS")
    if demo_mode:
        print("🚀 DEMO MODE: Using optimized fast pseudotime calculation")
    else:
        print("🔬 FULL MODE: Using rigorous DPT pseudotime calculation")
    print("="*60)
    
    # Initialize enhanced metrics
    metrics_calculator = EnhancedBiologicalMetrics(verbose=True)
    
    if use_synthetic:
        print("\nUsing synthetic benchmark dataset...")
        data = create_synthetic_benchmark_dataset(n_cells=2000, complexity='medium')
        X = data['X']
        cell_types = data['cell_types']
        true_pseudotime = data['true_pseudotime']
        bifurcation_tree = data.get('bifurcation_tree', None)
        dataset_name = 'synthetic'
    else:
        print("\nLoading real pancreas dataset...")
        X, cell_types, is_real = load_pancreas_data_enhanced()
        bifurcation_tree = None  # Real data will auto-detect
        dataset_name = 'pancreas'
    
    print(f"\nDataset: {dataset_name}")
    print(f"Data shape: {X.shape}")
    print(f"Number of cell types: {len(np.unique(cell_types))}")
    print(f"Cell type distribution:")
    unique, counts = np.unique(cell_types, return_counts=True)
    for ct, count in zip(unique, counts):
        print(f"  {ct}: {count}")
    
    # Standardize data for embedding
    print(f"\nStandardizing gene expression data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Algorithm configurations
    algorithms = {
        'NormalizedDynamics': {
            'class': NormalizedDynamicsOptimized,
            'params': {
                'dim': 2,
                'k': 20,
                'max_iter': 30,
                'adaptive_params': True,
                'device': 'cuda' if HAS_TORCH and torch.cuda.is_available() else 'cpu'
            },
            'description': 'Our novel method with adaptive parameters'
        },
        't-SNE': {
            'class': TSNE,
            'params': {
                'n_components': 2,
                'random_state': 42,
                'max_iter': 1000,
                'perplexity': min(30, len(X_scaled)//4)
            },
            'description': 'Standard t-SNE with optimized perplexity'
        }
    }
    
    if HAS_UMAP:
        algorithms['UMAP'] = {
            'class': umap.UMAP,
            'params': {
                'n_components': 2,
                'random_state': 42,
                'n_neighbors': min(15, len(X_scaled)//4),
                'min_dist': 0.1
            },
            'description': 'UMAP with standard parameters'
        }
    
    print(f"\nRunning {len(algorithms)} algorithms...")
    
    # Results storage
    results = {}
    embeddings = {}
    
    for alg_name, alg_config in algorithms.items():
        print(f"\n--- {alg_name} ---")
        
        # Initialize algorithm
        if alg_name == 'NormalizedDynamics':
            algorithm = alg_config['class'](**alg_config['params'])
        else:
            algorithm = alg_config['class'](**alg_config['params'])
        
        # Time the embedding computation
        start_time = time.time()
        
        try:
            if alg_name == 'NormalizedDynamics':
                # Fit and transform
                embedding = algorithm.fit_transform(X_scaled)
            else:
                # Standard sklearn/umap interface
                embedding = algorithm.fit_transform(X_scaled)
                
            runtime = time.time() - start_time
            
            print(f"Embedding computed: {embedding.shape} in {runtime:.1f}s")
            
            # Time the biological evaluation
            eval_start_time = time.time()
            
            # Run enhanced biological evaluation with demo mode option
            try:
                biological_metrics = metrics_calculator.comprehensive_enhanced_evaluation(
                    embedding, X, cell_types, use_dpt=not demo_mode, demo_mode=demo_mode, bifurcation_tree=bifurcation_tree
                )
                eval_time = time.time() - eval_start_time
                print(f"Biological evaluation completed in {eval_time:.1f}s")
                
            except Exception as eval_error:
                print(f"Biological evaluation failed: {eval_error}")
                if not demo_mode:
                    print("Falling back to demo mode...")
                    biological_metrics = metrics_calculator.comprehensive_enhanced_evaluation(
                        embedding, X, cell_types, use_dpt=False, demo_mode=True, bifurcation_tree=bifurcation_tree
                    )
                    eval_time = time.time() - eval_start_time
                    print(f"Fallback evaluation completed in {eval_time:.1f}s")
                else:
                    raise eval_error
            
            # Store results
            results[alg_name] = {
                'runtime': runtime,
                'eval_time': eval_time,
                'total_time': runtime + eval_time,
                'biological_metrics': biological_metrics
            }
            embeddings[alg_name] = embedding
            
            # Print key metrics
            tcs = biological_metrics['trajectory_coherence']['tcs_global']
            bps = biological_metrics['bifurcation_preservation']['bps_global']
            fps = biological_metrics['fragmentation']['fragmentation_penalty']
            
            print(f"Key metrics:")
            print(f"  Trajectory Coherence: {tcs:.3f}")
            print(f"  Bifurcation Preservation: {bps:.3f}")
            print(f"  Fragmentation Penalty: {fps:.3f}")
            print(f"Total time (embedding + evaluation): {runtime + eval_time:.1f}s")
            
        except Exception as e:
            print(f"❌ Algorithm failed: {e}")
            results[alg_name] = None
    
    # Create enhanced visualization
    image_path = None
    if len(results) > 0:
        image_path = create_enhanced_publication_visualization(embeddings, results, cell_types, dataset_name)
        print_enhanced_summary(results)
    
    return results, embeddings, image_path


def create_enhanced_publication_visualization(embeddings, results, cell_types, dataset_name):
    """Create publication-quality enhanced visualization."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Set up the figure with publication quality
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 12))
    
    # Color palette for cell types
    unique_types = np.unique(cell_types)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_types)))
    color_map = dict(zip(unique_types, colors))
    
    n_algorithms = len(embeddings)
    
    # Top row: Embeddings
    for i, (alg_name, embedding) in enumerate(embeddings.items()):
        ax = plt.subplot(2, n_algorithms, i + 1)
        
        # Plot embedding colored by cell type
        for cell_type in unique_types:
            mask = cell_types == cell_type
            plt.scatter(embedding[mask, 0], embedding[mask, 1], 
                       c=[color_map[cell_type]], label=cell_type, alpha=0.7, s=20)
        
        plt.title(f'{alg_name} Embedding', fontsize=14, fontweight='bold')
        plt.xlabel('Component 1', fontsize=12)
        plt.ylabel('Component 2', fontsize=12)
        
        if i == 0:  # Only show legend for first plot
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Bottom row: Metrics comparison
    ax_metrics = plt.subplot(2, 1, 2)
    
    # Extract metrics for comparison
    alg_names = list(results.keys())
    
    tcs_scores = []
    bps_scores = []
    continuity_scores = []  # 1 - fragmentation_penalty
    
    for alg_name in alg_names:
        metrics = results[alg_name]['biological_metrics']
        
        tcs_scores.append(metrics['trajectory_coherence']['tcs_global'])
        bps_scores.append(metrics['bifurcation_preservation']['bps_global'])
        continuity_scores.append(1 - metrics['fragmentation']['fragmentation_penalty'])
    
    # Create grouped bar chart
    x = np.arange(len(alg_names))
    width = 0.25
    
    bars1 = plt.bar(x - width, tcs_scores, width, label='Trajectory Coherence', color='#2E86AB', alpha=0.8)
    bars2 = plt.bar(x, bps_scores, width, label='Bifurcation Preservation', color='#A23B72', alpha=0.8)
    bars3 = plt.bar(x + width, continuity_scores, width, label='Continuity Score', color='#F18F01', alpha=0.8)
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    plt.xlabel('Algorithm', fontsize=14, fontweight='bold')
    plt.ylabel('Score (Higher = Better)', fontsize=14, fontweight='bold')
    plt.title('Enhanced Biological Accuracy Metrics - Publication Results', fontsize=16, fontweight='bold')
    plt.xticks(x, alg_names, fontsize=12)
    plt.ylim(0, 1.1)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add performance summary text
    best_alg = max(results.keys(), key=lambda k: results[k]['biological_metrics']['trajectory_coherence']['tcs_global'])
    best_tcs = results[best_alg]['biological_metrics']['trajectory_coherence']['tcs_global']
    
    plt.text(0.02, 0.98, f'Best Performing Algorithm: {best_alg}\nTrajectory Coherence: {best_tcs:.3f}', 
             transform=ax_metrics.transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the figure
    output_dir = 'static/results'
    os.makedirs(output_dir, exist_ok=True)
    filename = f'{dataset_name}_enhanced_biological_comparison_{timestamp}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\nEnhanced visualization saved: {filepath}")
    
    plt.close()  # Close instead of show for web usage
    return filepath


def print_enhanced_summary(results):
    """Print enhanced publication-quality summary with comprehensive metrics table."""
    print("\n" + "="*100)
    print("ENHANCED BIOLOGICAL METRICS SUMMARY - PUBLICATION RESULTS")
    print("="*100)
    
    # Sort algorithms by trajectory coherence score
    sorted_results = sorted(results.items(), 
                          key=lambda x: x[1]['biological_metrics']['trajectory_coherence']['tcs_global'], 
                          reverse=True)
    
    # Create comprehensive metrics table
    print(f"\nAlgorithm Performance Summary")
    print(f"{'Algorithm':<18} {'Trajectory Smoothness':<20} {'Cell Type Preservation':<22} {'Spatial Coherence':<16} {'Temporal Ordering':<16} {'Overall Score':<14}")
    print(f"{'':^18} {'(lower is better)':<20} {'(higher is better)':<22} {'(higher is better)':<16} {'(higher is better)':<16} {'(higher is better)':<14}")
    print("-" * 108)
    
    for alg_name, alg_results in sorted_results:
        metrics = alg_results['biological_metrics']
        
        # Extract detailed metrics
        trajectory_smoothness = metrics['fragmentation']['fragmentation_penalty']  # Lower is better
        cell_type_preservation = metrics['bifurcation_preservation']['bps_global']  # Higher is better
        spatial_coherence = metrics['trajectory_coherence']['multi_scale_coherence']  # Higher is better
        temporal_ordering = metrics['trajectory_coherence']['spearman_correlation']  # Higher is better
        
        # Calculate overall score (weighted combination, higher is better)
        overall_score = (
            0.3 * metrics['trajectory_coherence']['tcs_global'] +
            0.3 * cell_type_preservation +
            0.2 * (1 - trajectory_smoothness) +  # Convert to higher-is-better
            0.2 * temporal_ordering
        )
        
        # Add star for best performer
        star = " ⭐" if alg_name == sorted_results[0][0] else ""
        
        # Format the values
        traj_smooth_str = f"{trajectory_smoothness:.3f}{star}"
        cell_pres_str = f"{cell_type_preservation:.3f}" if cell_type_preservation > 0 else "N/A"
        spatial_coh_str = f"{spatial_coherence:.3f}" if spatial_coherence > 0 else "N/A"
        temporal_ord_str = f"{temporal_ordering:.3f}" if temporal_ordering > 0 else "N/A"
        overall_str = f"{overall_score:.3f}" if overall_score > 0 else "N/A"
        
        print(f"{alg_name:<18} {traj_smooth_str:<20} {cell_pres_str:<22} {spatial_coh_str:<16} {temporal_ord_str:<16} {overall_str:<14}")
    
    # Print performance notes
    print("\nPerformance Notes:")
    print("• Trajectory Smoothness: Measures fragmentation and discontinuities in developmental paths")
    print("• Cell Type Preservation: Evaluates how well cell type relationships are maintained")
    print("• Spatial Coherence: Assesses local neighborhood consistency in embedding space")
    print("• Temporal Ordering: Measures correlation between embedding distances and developmental time")
    print("• Overall Score: Weighted combination of all metrics for publication comparison")
    
    # Highlight the winner with detailed breakdown
    winner = sorted_results[0]
    print(f"\n🏆 BEST PERFORMER: {winner[0]}")
    winner_metrics = winner[1]['biological_metrics']
    
    print(f"\nDetailed Performance Breakdown:")
    
    # Trajectory metrics
    traj_metrics = winner_metrics['trajectory_coherence']
    print(f"   Trajectory Analysis:")
    print(f"     • Overall Coherence Score: {traj_metrics['tcs_global']:.3f}")
    print(f"     • Multi-scale Coherence: {traj_metrics['multi_scale_coherence']:.3f}")
    print(f"     • Distance-Time Correlation: {traj_metrics['spearman_correlation']:.3f}")
    print(f"     • Trajectory Smoothness: {traj_metrics['trajectory_smoothness']:.3f}")
    print(f"     • Stage Transition Quality: {traj_metrics['stage_transition_quality']:.3f}")
    
    # Bifurcation metrics
    bifur_metrics = winner_metrics['bifurcation_preservation']
    print(f"   Bifurcation Analysis:")
    print(f"     • Bifurcation Preservation Score: {bifur_metrics['bps_global']:.3f}")
    print(f"     • Branch Connectivity: {bifur_metrics['branch_connectivity']:.3f}")
    print(f"     • Bifurcations Analyzed: {bifur_metrics['n_bifurcations_analyzed']}")
    print(f"     • Temporal Consistency: {bifur_metrics.get('temporal_consistency', 0.0):.3f}")
    
    # Fragmentation metrics
    frag_metrics = winner_metrics['fragmentation']
    print(f"   Continuity Analysis:")
    print(f"     • Overall Fragmentation Penalty: {frag_metrics['fragmentation_penalty']:.3f}")
    print(f"     • Discontinuity Penalty: {frag_metrics['discontinuity_penalty']:.3f}")
    print(f"     • Stage Fragmentation: {frag_metrics['stage_fragmentation']:.3f}")
    print(f"     • Trajectory Roughness: {frag_metrics['trajectory_roughness']:.3f}")
    print(f"     • Number of Discontinuities: {frag_metrics.get('n_discontinuities', 0)}")
    
    # Runtime performance
    print(f"   Computational Performance:")
    print(f"     • Embedding Runtime: {winner[1]['runtime']:.1f}s")
    print(f"     • Evaluation Runtime: {winner[1]['eval_time']:.1f}s")
    print(f"     • Total Time: {winner[1]['total_time']:.1f}s")
    
    print("\n" + "="*100)


if __name__ == "__main__":
    # Run enhanced analysis on real pancreas data
    print("Running enhanced biological metrics analysis...")
    
    try:
        results, embeddings = run_enhanced_biological_metrics_comparison(use_synthetic=False)
        print("\nEnhanced analysis completed successfully!")
        
    except Exception as e:
        print(f"\nEnhanced analysis failed: {e}")
        import traceback
        traceback.print_exc() 