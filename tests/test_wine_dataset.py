import matplotlib
matplotlib.use('Agg')  # Set non-GUI backend
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import os
import sys
from datetime import datetime

# Check for UMAP availability
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# Add the project root to the Python path to allow importing from `src`
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.normalized_dynamics_optimized import NormalizedDynamicsOptimized, compute_metrics_optimized

def run_and_visualize_wine(results_dir="static/results"):
    """
    Analysis of the Wine dataset using NormalizedDynamics, t-SNE, and UMAP.
    
    The Wine dataset contains the results of a chemical analysis of wines grown 
    in the same region in Italy but derived from three different cultivars.
    
    Dataset details:
    - 178 samples
    - 13 features (alcohol, malic acid, ash, etc.)
    - 3 classes (wine cultivars: 0, 1, 2)
    
    Returns:
        - Path to the saved plot.
        - A dictionary containing the execution times for each model.
    """
    # --- 1. Setup and Data Loading ---
    print("Starting Wine dataset analysis...")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Set seed for reproducibility and device for computation
    random_seed = 42
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load Wine dataset
    print("Loading Wine dataset...")
    wine_data = load_wine()
    X = wine_data.data
    y = wine_data.target
    feature_names = wine_data.feature_names
    target_names = wine_data.target_names
    
    print(f"Dataset loaded successfully:")
    print(f"  - Shape: {X.shape}")
    print(f"  - Classes: {len(target_names)} ({', '.join(target_names)})")
    print(f"  - Class distribution: {np.bincount(y)}")
    
    # Standardize the features (important for wine dataset due to different scales)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"Features standardized (mean=0, std=1)")

    # --- 2. Model Initialization ---
    print("Initializing models...")
    models = {
        'NormalizedDynamics': NormalizedDynamicsOptimized(
            dim=2,
            k=15,  # Good for small dataset
            max_iter=50,
            device=device
        ),
        't-SNE': TSNE(
            n_components=2,
            perplexity=min(30, (len(X)-1)//3),  # Adapt to dataset size
            learning_rate='auto',
            init='pca',
            random_state=random_seed
        )
    }
    
    if UMAP_AVAILABLE:
        models['UMAP'] = UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            random_state=random_seed
        )

    # --- 3. Apply Methods and Benchmark ---
    embeddings = {}
    timings = {}
    metrics = {}
    for name, model in models.items():
        print(f"Applying {name}...")
        start_time = time.time()
        embedding = model.fit_transform(X_scaled)
        embeddings[name] = embedding
        end_time = time.time()
        timings[name] = round(end_time - start_time, 2)
        
        # Compute distortion and local structure metrics
        computed_metrics = compute_metrics_optimized(X_scaled, embedding)
        metrics[name] = computed_metrics
        
        print(f"{name} completed in {timings[name]:.2f}s - Distortion: {computed_metrics['distortion']:.4f}")
    
    # Find the method with lowest distortion
    best_distortion_method = min(metrics.keys(), key=lambda x: metrics[x]['distortion'])

    # --- 4. Visualization ---
    print("Generating visualization...")
    
    # Set modern dark theme
    plt.style.use('dark_background')
    plt.rcParams.update({
        "figure.facecolor": "#1a1a1a",
        "axes.facecolor": "#1a1a1a",
        "axes.edgecolor": "#00d4aa",
        "text.color": "#00d4aa",
        "xtick.color": "#00d4aa",
        "ytick.color": "#00d4aa",
        "axes.labelcolor": "#00d4aa",
    })

    # Create the comparison plot
    n_methods = len(embeddings)
    fig, axes = plt.subplots(1, n_methods, figsize=(6*n_methods, 6))
    if n_methods == 1:
        axes = [axes]
    
    # Colors for wine classes - improved visibility
    colors = ['#ff4757', '#ffd700', '#3742fa']  # Red, Bright Yellow, Blue
    wine_labels = ['Class 0 (Cultivar 1)', 'Class 1 (Cultivar 2)', 'Class 2 (Cultivar 3)']
    
    for i, (name, embedding) in enumerate(embeddings.items()):
        ax = axes[i]
        
        # Create scatter plot for each class
        for class_idx in range(3):
            mask = y == class_idx
            ax.scatter(embedding[mask, 0], embedding[mask, 1], 
                      c=colors[class_idx], label=wine_labels[class_idx], 
                      s=40, alpha=0.8, edgecolors='white', linewidth=0.5)
        
        # Highlight best distortion method
        is_best = name == best_distortion_method
        distortion = metrics[name]['distortion']
        title_color = '#00ff00' if is_best else '#00d4aa'  # Green for best, teal for others
        
        title = f"{name}\nTime: {timings[name]:.2f}s | Distortion: {distortion:.4f}"
        if is_best:
            title += "\nLOWEST DISTORTION"
        
        ax.set_title(title, fontsize=12, weight='bold', pad=20, color=title_color)
        ax.set_xlabel("Component 1", fontsize=12)
        ax.set_ylabel("Component 2", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

    # Add a single legend for the entire figure
    fig.legend(handles=axes[0].get_legend_handles_labels()[0], 
               labels=wine_labels,
               title="Wine Classes",
               bbox_to_anchor=(1.02, 0.9), 
               loc='upper left',
               fontsize=11)
    
    fig.suptitle('Wine Dataset: Dimensionality Reduction Comparison\n' + 
                 f'178 samples, 13 features → 2D embedding', 
                 fontsize=16, weight='bold', y=0.98)
    
    # Use tight_layout with rect to reserve space for title and legend
    plt.tight_layout(rect=[0, 0, 0.85, 0.90])  # [left, bottom, right, top] - reserves space for title
    
    # Save the figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    static_results_dir = os.path.join('static', 'results')
    if not os.path.exists(static_results_dir):
        os.makedirs(static_results_dir)
    save_path = os.path.join(static_results_dir, f"wine_comparison_{timestamp}.png")
    
    plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to {save_path}")
    
    return save_path, timings

if __name__ == "__main__":
    # Run the analysis when script is executed directly
    run_and_visualize_wine() 