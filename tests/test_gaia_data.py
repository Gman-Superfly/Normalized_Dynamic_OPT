import matplotlib
matplotlib.use('Agg') # Set non-GUI backend
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from umap import UMAP
from sklearn.manifold import TSNE
import os
import sys
from datetime import datetime

# Add the project root to the Python path to allow importing from `src`
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.normalized_dynamics_optimized import NormalizedDynamicsOptimized, compute_metrics_optimized

def run_and_visualize_gaia(data_path="data/gaia_data_500.csv", results_dir="static/results"):
    """
    Refactored analysis logic that can be imported and called from another script.
    
    Returns:
        - Path to the saved plot.
        - A dictionary containing the execution times for each model.
    """
    # --- 1. Setup and Data Loading ---
    print("Starting Gaia data analysis...")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Set seed for reproducibility and device for computation
    random_seed = 42
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load data
    print(f"Loading data from {data_path}...")
    try:
        df = pl.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}.")
        return None, {"error": f"Data file not found at {data_path}"}
    
    # Prepare data: X is the full 5D stellar parameter space, color_val is for visualization
    X = df.select(['x', 'y', 'z', 'bp_rp', 'mag']).to_numpy()
    color_val = df['bp_rp'].to_numpy()  # Keep bp_rp for coloring the plot
    print(f"Data loaded successfully. Shape: {X.shape} (5D: position + photometry)")

    # --- 2. Model Initialization ---
    print("Initializing models...")
    models = {
        'NormalizedDynamics': NormalizedDynamicsOptimized(
            dim=2,
            k=20,
            device=device  # Offload computation to GPU if available
        ),
        'UMAP': UMAP(
            n_neighbors=15,  # Standard default
            min_dist=0.1,    # Standard default
            n_components=2,
            random_state=random_seed,
            transform_queue_size=0.0 # Suppress a warning with n_jobs=1
        ),
        't-SNE': TSNE(
            n_components=2,
            perplexity=30,   # Standard default
            learning_rate='auto',
            init='pca',
            random_state=random_seed
        )
    }

    # --- 3. Apply Methods and Benchmark ---
    embeddings = {}
    timings = {}
    metrics = {}
    for name, model in models.items():
        print(f"Applying {name}...")
        start_time = time.time()
        if hasattr(model, 'fit_transform'):
            embeddings[name] = model.fit_transform(X)
        else:
            embeddings[name] = model.fit(X)
        end_time = time.time()
        timings[name] = round(end_time - start_time, 2)
        
        # Compute distortion and local structure metrics
        computed_metrics = compute_metrics_optimized(X, embeddings[name])
        metrics[name] = computed_metrics
        
        print(f"{name} completed in {timings[name]:.2f}s - Distortion: {computed_metrics['distortion']:.4f}")
    
    # Find the method with lowest distortion
    best_distortion_method = min(metrics.keys(), key=lambda x: metrics[x]['distortion'])

    # --- 4. Visualization ---
    print("Generating visualization...")
    
    # Set CRT theme for the plot
    plt.style.use('dark_background')
    plt.rcParams.update({
        "figure.facecolor": "#1a1a1a",
        "axes.facecolor": "#1a1a1a",
        "axes.edgecolor": "#ff9900",
        "text.color": "#ff9900",
        "xtick.color": "#ff9900",
        "ytick.color": "#ff9900",
        "axes.labelcolor": "#ff9900",
        "figure.edgecolor": "#1a1a1a",
    })

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle('Dimensionality Reduction on Gaia 5D Stellar Parameters (Position + Photometry)', fontsize=16)

    for i, name in enumerate(models.keys()):
        ax = axes[i]
        result = embeddings[name]
        
        scatter = ax.scatter(result[:, 0], result[:, 1], c=color_val, cmap='coolwarm', s=5, alpha=0.7)
        
        # Highlight best distortion method
        is_best = name == best_distortion_method
        distortion = metrics[name]['distortion']
        title_color = '#00ff00' if is_best else '#ff9900'  # Green for best, orange for others
        
        title = f"{name}\nTime: {timings[name]:.2f}s | Distortion: {distortion:.4f}"
        if is_best:
            title += "\nLOWEST DISTORTION"
        
        ax.set_title(title, fontsize=11, color=title_color)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.set_aspect('equal', adjustable='box')

    # Adjust layout and add colorbar
    plt.subplots_adjust(left=0.05, right=0.88, top=0.9, bottom=0.1)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label('Stellar Color (BP-RP)')
    cbar.ax.yaxis.set_tick_params(color='#ff9900')
    
    # Save the figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Make the results directory part of the static folder for web access
    static_results_dir = os.path.join('static', 'results')
    if not os.path.exists(static_results_dir):
        os.makedirs(static_results_dir)
    save_path = os.path.join(static_results_dir, f"GAIA_comparison_{timestamp}.png")
    
    plt.savefig(save_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig) # Prevent plot from showing in the console
    print(f"Plot saved to {save_path}")
    
    return save_path, timings

if __name__ == "__main__":
    # This allows the script to be run directly for testing
    run_and_visualize_gaia() 