"""
NormalizedDynamics with Smart K Parameter Adaptation
==================================================

Enhanced version with dynamic K adaptation optimized for:
1. Smart sampling scenarios (maintaining quality on subsampled data)
2. Varying dataset sizes (from 500 to 10,000+ cells)
3. Different data densities and structures
4. Biological trajectory preservation

Key innovations:
- Dataset size-aware K scaling
- Local density-aware adaptation  
- Biological structure preservation
- Computational efficiency optimization

Author: NormalizedDynamics Team
Date: 2024
"""

import torch
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
import warnings

class NormalizedDynamicsSmartK(torch.nn.Module):
    """
    Enhanced NormalizedDynamics with dynamic K parameter adaptation.
    
    This version automatically adapts the K parameter based on:
    1. Dataset size (larger datasets need more neighbors for stability)
    2. Local density patterns (dense regions need fewer, sparse regions need more)
    3. Data dimensionality (higher dimensions need more neighbors)
    4. Biological structure preservation requirements
    """
    
    def __init__(self, dim=2, k_base=None, alpha=1.0, max_iter=50, noise_scale=0.01, 
                 eta=0.01, target_local_structure=0.95, adaptive_params=True, 
                 k_adaptation_strategy='smart', device='cpu'):
        super(NormalizedDynamicsSmartK, self).__init__()
        
        self.dim = dim
        self.k_base = k_base  # If None, will be computed automatically
        self.alpha = torch.nn.Parameter(torch.tensor(alpha)) if adaptive_params else alpha
        self.max_iter = max_iter
        self.noise_scale = noise_scale
        self.eta = eta
        self.target_local_structure = target_local_structure
        self.adaptive_params = adaptive_params
        self.k_adaptation_strategy = k_adaptation_strategy
        self.device = device
        
        # K adaptation parameters
        self.k_history = []
        self.density_history = []
        
        # Optimization tracking
        self.cost_history = []
        self.alpha_history = []
    
    def compute_optimal_k_base(self, n_samples, n_features):
        """
        Compute optimal base K parameter based on dataset characteristics.
        
        Uses established heuristics from manifold learning literature:
        - Larger datasets can support more neighbors
        - Higher dimensional data needs more neighbors
        - Balance between local structure and global connectivity
        """
        if self.k_base is not None:
            return self.k_base
        
        # Dataset size component
        if n_samples <= 500:
            size_k = 8   # Small datasets: conservative K
        elif n_samples <= 1000:
            size_k = 12  # Medium datasets: moderate K
        elif n_samples <= 2000:
            size_k = 18  # Standard size: balanced K
        elif n_samples <= 5000:
            size_k = 25  # Large datasets: higher K for stability
        else:
            size_k = 35  # Very large: maximum connectivity
        
        # Dimensionality component
        if n_features <= 50:
            dim_k = 0    # Low dim: no adjustment needed
        elif n_features <= 500:
            dim_k = 3    # Medium dim: slight increase
        elif n_features <= 2000:
            dim_k = 6    # High dim: moderate increase
        else:
            dim_k = 10   # Very high dim: significant increase
        
        # Biological structure component (for developmental data)
        bio_k = 5  # Extra neighbors for trajectory preservation
        
        base_k = size_k + dim_k + bio_k
        
        # Ensure reasonable bounds
        min_k = max(5, int(np.sqrt(n_samples)) // 4)
        max_k = min(50, n_samples // 10)
        
        optimal_k = np.clip(base_k, min_k, max_k)
        
        print(f"📊 Computed optimal K: {optimal_k} (size: {size_k}, dim: {dim_k}, bio: {bio_k})")
        return optimal_k
    
    def adaptive_k_selection(self, x, dists):
        """
        Adaptive K selection based on local data characteristics.
        
        Strategies:
        1. 'smart': Full adaptive strategy considering multiple factors
        2. 'density': Adapt based on local density only  
        3. 'size': Adapt based on dataset size only
        4. 'fixed': Use base K without adaptation
        """
        n_samples = x.size(0)
        base_k = self.compute_optimal_k_base(n_samples, x.size(1))
        
        if self.k_adaptation_strategy == 'fixed':
            return base_k
        
        elif self.k_adaptation_strategy == 'size':
            # Simple size-based adaptation
            return base_k
        
        elif self.k_adaptation_strategy == 'density':
            # Density-based adaptation only
            return self._density_adaptive_k(dists, base_k)
        
        elif self.k_adaptation_strategy == 'smart':
            # Full smart adaptation
            return self._smart_adaptive_k(x, dists, base_k)
        
        else:
            warnings.warn(f"Unknown K adaptation strategy: {self.k_adaptation_strategy}")
            return base_k
    
    def _density_adaptive_k(self, dists, base_k):
        """Adapt K based on local density patterns."""
        # Compute local density indicators
        knn_dists = torch.topk(dists, min(15, dists.size(0)), largest=False)[0]
        local_density = 1.0 / (torch.mean(knn_dists, dim=1) + 1e-8)
        
        # Normalize density
        density_normalized = (local_density - local_density.min()) / (local_density.max() - local_density.min() + 1e-8)
        
        # Adaptive K: high density regions use fewer neighbors, low density use more
        k_adaptive = base_k + 10 * (1.0 - density_normalized)  # Inverse relationship
        k_adaptive = torch.clamp(k_adaptive, 5, min(50, dists.size(0) - 1))
        
        # Use median K for stability
        final_k = int(torch.median(k_adaptive).item())
        
        return final_k
    
    def _smart_adaptive_k(self, x, dists, base_k):
        """
        Smart adaptive K considering multiple factors:
        1. Local density patterns
        2. Data manifold curvature  
        3. Neighborhood consistency
        4. Computational efficiency
        """
        n_samples = x.size(0)
        
        # Factor 1: Local density adaptation
        density_k = self._density_adaptive_k(dists, base_k)
        
        # Factor 2: Manifold curvature estimation
        curvature_k = self._curvature_adaptive_k(x, dists, base_k)
        
        # Factor 3: Neighborhood consistency
        consistency_k = self._consistency_adaptive_k(dists, base_k)
        
        # Factor 4: Computational efficiency consideration
        efficiency_k = self._efficiency_adaptive_k(n_samples, base_k)
        
        # Weighted combination of factors
        weights = {
            'density': 0.4,      # Primary factor
            'curvature': 0.3,    # Important for trajectory preservation
            'consistency': 0.2,  # Stability factor
            'efficiency': 0.1    # Computational constraint
        }
        
        smart_k = (weights['density'] * density_k + 
                  weights['curvature'] * curvature_k +
                  weights['consistency'] * consistency_k + 
                  weights['efficiency'] * efficiency_k)
        
        # Ensure integer and reasonable bounds
        smart_k = int(round(smart_k))
        smart_k = np.clip(smart_k, 5, min(50, n_samples - 1))
        
        # Store for analysis
        self.k_history.append(smart_k)
        
        return smart_k
    
    def _curvature_adaptive_k(self, x, dists, base_k):
        """Estimate local manifold curvature and adapt K accordingly."""
        try:
            # Simple curvature estimation using PCA on local neighborhoods
            k_sample = min(20, x.size(0) - 1)
            _, nn_indices = torch.topk(dists, k_sample, largest=False)
            
            curvatures = []
            for i in range(0, x.size(0), max(1, x.size(0) // 50)):  # Sample points
                neighbors = x[nn_indices[i]]
                
                # Center the neighborhood
                centered = neighbors - neighbors.mean(dim=0, keepdim=True)
                
                # SVD for local PCA
                try:
                    U, S, V = torch.svd(centered)
                    
                    # Curvature indicator: ratio of explained variance
                    if len(S) > 1:
                        curvature = 1.0 - (S[0] / (S.sum() + 1e-8))
                        curvatures.append(curvature.item())
                except:
                    curvatures.append(0.5)  # Default moderate curvature
            
            mean_curvature = np.mean(curvatures) if curvatures else 0.5
            
            # High curvature regions need more neighbors for stability
            curvature_factor = 1.0 + 0.5 * mean_curvature
            curvature_k = base_k * curvature_factor
            
            return int(np.clip(curvature_k, base_k * 0.7, base_k * 1.5))
            
        except Exception as e:
            # Fallback to base K if curvature estimation fails
            return base_k
    
    def _consistency_adaptive_k(self, dists, base_k):
        """Adapt K based on neighborhood consistency."""
        try:
            # Test different K values for consistency
            k_candidates = [base_k - 5, base_k, base_k + 5]
            k_candidates = [k for k in k_candidates if 5 <= k < dists.size(0)]
            
            consistencies = []
            for k_test in k_candidates:
                _, nn_indices = torch.topk(dists, k_test, largest=False)
                
                # Measure consistency: how much do neighborhoods overlap?
                overlaps = []
                for i in range(0, dists.size(0), max(1, dists.size(0) // 20)):
                    for j in range(i + 1, min(i + 10, dists.size(0))):
                        overlap = len(set(nn_indices[i].tolist()) & set(nn_indices[j].tolist()))
                        overlaps.append(overlap / k_test)
                
                consistency = np.mean(overlaps) if overlaps else 0.5
                consistencies.append(consistency)
            
            # Choose K with best consistency
            if consistencies:
                best_idx = np.argmax(consistencies)
                return k_candidates[best_idx]
            else:
                return base_k
                
        except Exception as e:
            return base_k
    
    def _efficiency_adaptive_k(self, n_samples, base_k):
        """Adapt K considering computational efficiency."""
        # For large datasets, limit K to maintain performance
        if n_samples > 5000:
            efficiency_k = min(base_k, 30)  # Cap for large datasets
        elif n_samples > 2000:
            efficiency_k = min(base_k, 40)  # Moderate cap
        else:
            efficiency_k = base_k  # No constraint for small datasets
        
        return efficiency_k
    
    def forward(self, x):
        """
        Forward pass with smart K adaptation.
        """
        # Store original statistics for scale preservation
        original_mean = torch.mean(x, dim=0, keepdim=True)
        original_std = torch.std(x, dim=0, keepdim=True)
        
        # Center the data
        x_centered = x - original_mean
        
        # Compute pairwise distances
        dists = torch.cdist(x_centered, x_centered)
        
        # Smart K selection
        k = self.adaptive_k_selection(x, dists)
        
        # Extract k-th nearest neighbor distances for bandwidth calibration
        kth_dists, _ = torch.topk(dists, k, dim=1, largest=False)
        sigma = kth_dists[:, -1].view(-1, 1)  # Adaptive bandwidth parameter
        
        # Global kernel computation with adaptive bandwidth
        kernel = torch.exp(-dists / (2 * sigma**2 + 1e-8))
        kernel = kernel / (torch.sum(kernel, dim=1, keepdim=True) + 1e-8)
        
        # Comprehensive drift calculation with global information integration
        drift = torch.matmul(kernel, x_centered)
        
        # Dynamic step size calculation
        alpha_val = self.alpha if isinstance(self.alpha, float) else self.alpha.item()
        step_size = self.dim**(-alpha_val)
        
        # Stochastic update with noise
        noise = torch.randn_like(x_centered) * self.noise_scale
        h = x_centered + step_size * (drift - x_centered) + noise
        
        # Scale preservation
        current_std = torch.std(h, dim=0, keepdim=True)
        h = h * (original_std / (current_std + 1e-8))
        
        # Add back the mean
        h = h + original_mean
        
        return h
    
    def fit_transform(self, X):
        """
        Fit transform with smart K adaptation and detailed logging.
        """
        # Convert input to PyTorch tensor and move to device
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32)
        X = X.to(self.device)
        
        n_samples, n_features = X.shape
        print(f"🧠 Smart K adaptation for dataset: {n_samples} cells × {n_features} features")
        
        # Better initialization
        if X.shape[1] <= self.dim:
            # If input dim <= target dim, pad with noise
            embedding = torch.cat([X, torch.randn(n_samples, self.dim - X.shape[1]) * 0.1], dim=1)
        else:
            # PCA-like initialization for better starting point
            U, S, V = torch.svd(X - X.mean(dim=0, keepdim=True))
            embedding = U[:, :self.dim] * S[:self.dim].sqrt()
        
        embedding = embedding.to(self.device)
        
        # Iterative updates with smart K adaptation
        prev_cost = float('inf')
        patience = 5
        patience_counter = 0
        
        print(f"🔄 Starting optimization with {self.k_adaptation_strategy} K adaptation...")
        
        for iteration in range(self.max_iter):
            embedding_old = embedding.clone()
            embedding = self.forward(embedding)
            
            # Print K adaptation info
            if iteration == 0 or iteration % 10 == 0:
                current_k = self.k_history[-1] if self.k_history else "Computing..."
                print(f"   Iteration {iteration}: K = {current_k}")
            
            # Compute cost for early stopping and alpha adaptation
            if iteration % 5 == 0:
                try:
                    cost, metrics = self.compute_cost(X.cpu().numpy(), embedding.cpu().detach().numpy())
                    
                    # Store history
                    self.cost_history.append(cost)
                    alpha_val = self.alpha if isinstance(self.alpha, float) else self.alpha.item()
                    self.alpha_history.append(alpha_val)
                    
                    # Adaptive alpha adjustment
                    if self.adaptive_params and isinstance(self.alpha, torch.nn.Parameter):
                        error = self.target_local_structure - metrics['local_structure']
                        with torch.no_grad():
                            self.alpha += self.eta * error
                            self.alpha.clamp_(0.01, 2.0)
                    
                    # Early stopping
                    if iteration > 10:
                        if cost > prev_cost or abs(prev_cost - cost) < 1e-5:
                            patience_counter += 1
                        else:
                            patience_counter = 0
                        
                        if patience_counter >= patience:
                            print(f"   Early stopping at iteration {iteration}, cost: {cost:.4f}")
                            break
                    
                    prev_cost = cost
                    
                except Exception as e:
                    print(f"   Warning: Cost computation failed at iteration {iteration}: {e}")
            
            # Convergence check
            if torch.norm(embedding - embedding_old) < 1e-6:
                print(f"   Convergence at iteration {iteration}")
                break
        
        # Print final K adaptation summary
        if self.k_history:
            k_stats = {
                'mean': np.mean(self.k_history),
                'min': np.min(self.k_history),
                'max': np.max(self.k_history),
                'final': self.k_history[-1]
            }
            print(f"📈 K adaptation summary: mean={k_stats['mean']:.1f}, "
                  f"range=[{k_stats['min']}-{k_stats['max']}], final={k_stats['final']}")
        
        return embedding.cpu().detach().numpy()
    
    def compute_cost(self, original, embedded):
        """Compute cost function for optimization."""
        try:
            from src.normalized_dynamics_optimized import compute_metrics_optimized
            metrics = compute_metrics_optimized(original, embedded)
            cost = 0.3 * metrics['distortion'] + 0.7 * (1 - metrics['local_structure'])
            return cost, metrics
        except:
            return 1.0, {'distortion': 1.0, 'local_structure': 0.0}
    
    def get_k_adaptation_info(self):
        """Return detailed information about K adaptation."""
        return {
            'strategy': self.k_adaptation_strategy,
            'k_history': self.k_history.copy(),
            'k_statistics': {
                'mean': np.mean(self.k_history) if self.k_history else None,
                'std': np.std(self.k_history) if self.k_history else None,
                'min': np.min(self.k_history) if self.k_history else None,
                'max': np.max(self.k_history) if self.k_history else None
            },
            'base_k': self.k_base
        }

# Convenience function for easy testing
def create_smart_k_algorithm(dataset_size, strategy='smart', **kwargs):
    """
    Create NormalizedDynamicsSmartK with optimal settings for dataset size.
    
    Args:
        dataset_size: Number of samples in dataset
        strategy: K adaptation strategy ('smart', 'density', 'size', 'fixed')
        **kwargs: Additional parameters
    """
    # Default parameters optimized for smart sampling scenarios
    defaults = {
        'dim': 2,
        'alpha': 1.1,
        'max_iter': 100,
        'eta': 0.003,
        'target_local_structure': 0.96,
        'adaptive_params': True,
        'k_adaptation_strategy': strategy,
        'device': 'cpu'
    }
    
    # Override with user parameters
    params = {**defaults, **kwargs}
    
    # Adjust parameters based on dataset size
    if dataset_size <= 1000:
        params['max_iter'] = 80   # Fewer iterations for small datasets
    elif dataset_size >= 5000:
        params['max_iter'] = 120  # More iterations for large datasets
    
    print(f"🚀 Creating SmartK algorithm for {dataset_size} samples with '{strategy}' adaptation")
    
    return NormalizedDynamicsSmartK(**params)

if __name__ == "__main__":
    # Test smart K adaptation
    print("Testing Smart K Adaptation...")
    
    # Create test data of different sizes
    test_sizes = [500, 1000, 2000, 5000]
    
    for size in test_sizes:
        print(f"\n{'='*50}")
        print(f"Testing dataset size: {size}")
        print(f"{'='*50}")
        
        # Generate synthetic data
        np.random.seed(42)
        X_test = np.random.randn(size, 100)
        
        # Test different K strategies
        strategies = ['fixed', 'density', 'smart']
        
        for strategy in strategies:
            print(f"\n--- Testing {strategy} strategy ---")
            
            model = create_smart_k_algorithm(size, strategy=strategy, max_iter=20)
            
            import time
            start_time = time.time()
            embedding = model.fit_transform(X_test)
            runtime = time.time() - start_time
            
            k_info = model.get_k_adaptation_info()
            print(f"Runtime: {runtime:.2f}s")
            print(f"K adaptation: {k_info['k_statistics']}")
    
    print("\n✅ Smart K adaptation testing complete!") 