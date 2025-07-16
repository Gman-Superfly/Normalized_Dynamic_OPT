import unittest
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import random
from datetime import datetime
from sklearn.datasets import make_swiss_roll, make_moons, make_circles, make_blobs
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# Import our optimized algorithm
import sys
sys.path.append('..')
from normalized_dynamics_optimized import NormalizedDynamicsOptimized

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available. Install with: pip install umap-learn")

def set_global_seed(seed=None):
    """
    Set global random seed for reproducible results within a test run.
    
    Args:
        seed (int, optional): Random seed. If None, generates a random seed.
        
    Returns:
        int: The seed that was set
    """
    if seed is None:
        # Generate a random seed based on current time
        seed = int(datetime.now().timestamp() * 1000) % 2**32
    
    # Set all random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Set CUDA seed if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Make PyTorch deterministic (optional, can slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    return seed

class TestComprehensiveVisualizations(unittest.TestCase):
    """Comprehensive tests with visualization outputs for NormalizedDynamics."""
    
    def setUp(self):
        """Set up the test environment."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = NormalizedDynamicsOptimized(dim=2, device=self.device)
        
        # Use static/results instead of local results directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.results_dir = os.path.join(project_root, 'static', 'results')
        self.individual_dir = os.path.join(self.results_dir, 'individual')
        self.comprehensive_dir = os.path.join(self.results_dir, 'comprehensive')
        
        # Ensure directories exist
        os.makedirs(self.individual_dir, exist_ok=True)
        os.makedirs(self.comprehensive_dir, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        sns.set_palette("husl")

    def generate_datasets(self, random_seed):
        """Generate test datasets with specified random seed for reproducibility."""
        datasets = {}
        
        # Multi-Scale Circles
        X_circles, y_circles = make_circles(n_samples=1000, noise=0.05, factor=0.5, random_state=random_seed)
        datasets['Multi-Scale Circles'] = (X_circles, y_circles)
        
        # Clustered Data (4 blobs)
        X_blobs, y_blobs = make_blobs(n_samples=1000, centers=4, n_features=2, 
                                     cluster_std=1.0, random_state=random_seed)
        datasets['Clustered Data'] = (X_blobs, y_blobs)
        
        # Two Moons
        X_moons, y_moons = make_moons(n_samples=1000, noise=0.1, random_state=random_seed)
        datasets['Two Moons'] = (X_moons, y_moons)
        
        # Swiss Roll (3D -> 2D)
        X_swiss, color_swiss = make_swiss_roll(n_samples=1000, noise=0.1, random_state=random_seed)
        # For visualization, we'll use the X,Z view (removing Y)
        X_swiss_2d = X_swiss[:, [0, 2]]
        datasets['Swiss Roll'] = (X_swiss_2d, color_swiss)
        
        return datasets

    def compute_metrics(self, original, embedded):
        """Compute distortion and local structure preservation metrics."""
        scaler_orig = StandardScaler()
        scaler_emb = StandardScaler()
        
        original_norm = scaler_orig.fit_transform(original)
        embedded_norm = scaler_emb.fit_transform(embedded)
        
        dist_orig = cdist(original_norm, original_norm)
        dist_emb = cdist(embedded_norm, embedded_norm)
        
        # Distortion
        distortion = np.mean(np.abs(dist_orig - dist_emb)) / (np.mean(dist_orig) + 1e-8)
        
        # Local structure preservation
        k = min(10, len(original) - 1)
        if k == 0:
            return {'distortion': distortion, 'local_structure': 0.0}
            
        nn_orig = np.argsort(dist_orig, axis=1)[:, 1:k+1]
        nn_emb = np.argsort(dist_emb, axis=1)[:, 1:k+1]
        
        local_structure = np.mean([len(np.intersect1d(nn_orig[i], nn_emb[i])) / k
                                   for i in range(len(original))])
        
        return {'distortion': distortion, 'local_structure': local_structure}

    def apply_methods(self, X, y, random_seed):
        """Apply different dimensionality reduction methods with consistent seeding."""
        results = {}
        
        # Original data (if already 2D)
        results['Original'] = X.copy()
        
        # NormalizedDynamics (Optimized) - uses global torch seed
        start_time = time.time()
        nd_result = self.model.fit_transform(X)
        nd_time = time.time() - start_time
        results['NormalizedDynamics'] = nd_result
        
        # t-SNE - use the run seed with robust parameters for a fair comparison
        start_time = time.time()
        # Using PCA initialization and auto learning rate, which are best practices.
        # Perplexity is adapted to dataset size for stability.
        tsne = TSNE(n_components=2, 
                    random_state=random_seed, 
                    perplexity=min(30, (len(X)-1)//3 if len(X) > 3 else 1), 
                    init='pca', 
                    learning_rate='auto')
        tsne_result = tsne.fit_transform(X)
        tsne_time = time.time() - start_time
        results['t-SNE'] = tsne_result
        
        # UMAP (if available) - use the run seed with standard defaults for a fair comparison
        if UMAP_AVAILABLE:
            start_time = time.time()
            # Default n_neighbors=15 and min_dist=0.1 are widely used and considered
            # a fair baseline for standard comparisons.
            umap_reducer = umap.UMAP(n_components=2, 
                                     random_state=random_seed,
                                     n_neighbors=15,
                                     min_dist=0.1)
            umap_result = umap_reducer.fit_transform(X)
            umap_time = time.time() - start_time
            results['UMAP'] = umap_result
        
        return results

    def plot_comparison(self, datasets, save_path, random_seed):
        """Create a comprehensive comparison plot like the reference image."""
        n_datasets = len(datasets)
        methods = ['Original', 'NormalizedDynamics', 't-SNE'] + (['UMAP'] if UMAP_AVAILABLE else [])
        n_methods = len(methods)
        
        fig, axes = plt.subplots(n_datasets, n_methods, figsize=(4*n_methods, 4*n_datasets))
        if n_datasets == 1:
            axes = axes.reshape(1, -1)
        
        for i, (dataset_name, (X, y)) in enumerate(datasets.items()):
            results = self.apply_methods(X, y, random_seed)
            
            for j, method in enumerate(methods):
                if method not in results:
                    continue
                    
                ax = axes[i, j]
                embedded = results[method]
                
                # Create scatter plot
                scatter = ax.scatter(embedded[:, 0], embedded[:, 1], c=y, 
                                   cmap='viridis', s=20, alpha=0.7)
                
                # Compute and display metrics
                if method != 'Original':
                    metrics = self.compute_metrics(X, embedded)
                    dist_text = f"Dist: {metrics['distortion']:.3f}"
                    local_text = f"Local: {metrics['local_structure']:.3f}"
                    ax.text(0.02, 0.98, f"{dist_text}, {local_text}", 
                           transform=ax.transAxes, fontsize=8, 
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Set title
                if i == 0:
                    ax.set_title(method, fontsize=12, fontweight='bold')
                if j == 0:
                    ax.set_ylabel(dataset_name, fontsize=12, fontweight='bold')
                
                ax.set_aspect('equal', adjustable='box')
                ax.grid(True, alpha=0.3)
        
        # Add seed information to the plot
        fig.suptitle(f'Comprehensive Algorithm Comparison (Seed: {random_seed})', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.94)  # Make room for title
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def test_individual_datasets(self):
        """Test individual datasets and save separate plots."""
        print("\n" + "="*60)
        print("Running Individual Dataset Tests")
        print("="*60)
        
        # Set global seed for this test run
        run_seed = set_global_seed()
        print(f"Test run seed: {run_seed}")
        
        # Create timestamp for all individual tests
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        datasets = self.generate_datasets(run_seed)
        
        for dataset_name, (X, y) in datasets.items():
            print(f"\nTesting {dataset_name}...")
            
            # Test the optimized algorithm (uses global torch seed)
            start_time = time.time()
            X_embedded = self.model.fit_transform(X)
            elapsed = time.time() - start_time
            
            # Verify shape
            self.assertEqual(X_embedded.shape, (len(X), 2))
            
            # Compute metrics
            metrics = self.compute_metrics(X, X_embedded)
            
            print(f"-> {dataset_name} completed in {elapsed:.2f}s")
            print(f"  Distortion: {metrics['distortion']:.3f}")
            print(f"  Local Structure: {metrics['local_structure']:.3f}")
            
            # Save individual plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Original data
            ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=20, alpha=0.7)
            ax1.set_title(f'{dataset_name} (Original)', fontweight='bold')
            ax1.set_aspect('equal', adjustable='box')
            ax1.grid(True, alpha=0.3)
            
            # Embedded data
            ax2.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='viridis', s=20, alpha=0.7)
            ax2.set_title(f'{dataset_name} (NormalizedDynamics)', fontweight='bold')
            ax2.text(0.02, 0.98, f"Dist: {metrics['distortion']:.3f}, Local: {metrics['local_structure']:.3f}", 
                    transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax2.set_aspect('equal', adjustable='box')
            ax2.grid(True, alpha=0.3)
            
            # Add seed info to title
            fig.suptitle(f'{dataset_name} Test Results (Seed: {run_seed})', 
                        fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)  # Make room for title
            safe_name = dataset_name.replace(' ', '_').replace('-', '_')
            save_path = os.path.join(self.individual_dir, f'{safe_name}_test_{timestamp}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved individual plot: {save_path}")

    def test_comprehensive_comparison(self):
        """Generate comprehensive comparison like the reference image."""
        print("\n" + "="*60)
        print("Generating Comprehensive Comparison")
        print("="*60)
        
        # Set global seed for this test run
        run_seed = set_global_seed()
        print(f"Test run seed: {run_seed}")
        
        datasets = self.generate_datasets(run_seed)
        
        # Create timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.comprehensive_dir, f'comprehensive_comparison_{timestamp}.png')
        
        self.plot_comparison(datasets, save_path, run_seed)
        
        print(f"-> Comprehensive comparison saved: {save_path}")

    def test_performance_benchmark(self):
        """Run performance benchmarks across different data sizes."""
        print("\n" + "="*60)
        print("Running Performance Benchmarks")
        print("="*60)
        
        # Set global seed for this test run
        run_seed = set_global_seed()
        print(f"Benchmark run seed: {run_seed}")
        
        # Create timestamp for benchmark
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        sizes = [100, 500, 1000, 2000]
        times = []
        
        for size in sizes:
            print(f"\nBenchmarking with {size} samples...")
            X, _ = make_swiss_roll(n_samples=size, noise=0.1, random_state=run_seed)
            X_2d = X[:, [0, 2]]  # Use 2D projection
            
            start_time = time.time()
            self.model.fit_transform(X_2d)
            elapsed = time.time() - start_time
            times.append(elapsed)
            
            print(f"-> {size} samples: {elapsed:.2f}s")

        # Plot performance
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(sizes, times, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Samples', fontsize=12)
        ax.set_ylabel('Computation Time (seconds)', fontsize=12)
        ax.set_title(f'NormalizedDynamics Performance Benchmark (Seed: {run_seed})', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add annotations
        for size, time_val in zip(sizes, times):
            ax.annotate(f'{time_val:.2f}s', (size, time_val), 
                       textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.tight_layout()
        benchmark_path = os.path.join(self.comprehensive_dir, f'performance_benchmark_{timestamp}.png')
        plt.savefig(benchmark_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"-> Performance benchmark saved: {benchmark_path}")

if __name__ == '__main__':
    print("="*60)
    print("NormalizedDynamics Comprehensive Test Suite")
    print("="*60)
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"UMAP Available: {UMAP_AVAILABLE}")
    
    # Set master seed for the entire test session
    master_seed = set_global_seed()
    print(f"Master test session seed: {master_seed}")
    print("Starting comprehensive tests with reproducible seeding...")
    
    # Create test instance and run manually for better control
    test_instance = TestComprehensiveVisualizations()
    test_instance.setUp()
    
    try:
        # Run individual tests
        test_instance.test_individual_datasets()
        
        # Run comprehensive comparison
        test_instance.test_comprehensive_comparison()
        
        # Run performance benchmark
        test_instance.test_performance_benchmark()
        
        print("\n" + "="*60)
        print("All tests completed successfully!")
        print(f"All results generated with master seed: {master_seed}")
        print("Check the 'tests/results/' directory for generated visualizations.")
        print("="*60)
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc() 