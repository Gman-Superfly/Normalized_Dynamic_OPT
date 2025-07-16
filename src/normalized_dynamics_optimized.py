import torch
import numpy as np
from scipy.spatial.distance import cdist

class NormalizedDynamicsOptimized(torch.nn.Module):
    """
    Optimized NormalizedDynamics model with comprehensive connectivity and adaptive bandwidth.
    
    This implementation utilizes global connectivity through complete pairwise analysis,
    enabling superior geometric preservation and accuracy. The algorithm maintains
    comprehensive information integration while adapting kernel bandwidth based on
    local density patterns.
    
    Key features:
    - Adaptive kernel bandwidth calculation using k-th nearest neighbor distances
    - Global connectivity with complete pairwise distance computation
    - Proper data centering and scale preservation
    - Dynamic step size adaptation for stable convergence
    - Adaptive bandwidth mechanism responding to local density variance
    - Stochastic exploration for improved optimization
    - Cost-driven early stopping for computational efficiency
    
    Performance characteristics:
    - Excellent efficiency for small to medium datasets (≤2000 samples)
    - Superior accuracy through comprehensive connectivity
    - Enhanced geometric preservation capabilities
    - Scalable with appropriate computational resources
    """
    
    def __init__(self, dim=2, k=20, alpha=1.0, max_iter=50, noise_scale=0.01, eta=0.01,
                 target_local_structure=0.95, adaptive_params=True, device='cpu'):
        super(NormalizedDynamicsOptimized, self).__init__()
        self.dim = dim
        self.k = k
        self.alpha = torch.nn.Parameter(torch.tensor(alpha)) if adaptive_params else alpha
        self.max_iter = max_iter
        self.noise_scale = noise_scale
        self.eta = eta
        self.target_local_structure = target_local_structure
        self.adaptive_params = adaptive_params
        self.device = device
        
        # Optimization tracking
        self.cost_history = []
        self.alpha_history = []
        
    def forward(self, x):
        """
        Perform one iteration of the embedding update using the corrected algorithm.
        """
        # Store original statistics for scale preservation
        original_mean = torch.mean(x, dim=0, keepdim=True)
        original_std = torch.std(x, dim=0, keepdim=True)
        
        # Center the data
        x_centered = x - original_mean
        
        # Compute pairwise distances
        dists = torch.cdist(x_centered, x_centered)
        
        # Adaptive bandwidth selection for local density adaptation
        # Adjusts kernel width while maintaining global connectivity
        base_k = min(self.k, x.size(0) - 1)
        if self.adaptive_params and x.size(0) > 20:
            # Adaptive bandwidth mechanism based on local distance variance
            # Functions as adaptive blur radius for kernel computation
            density = torch.std(dists, dim=1)
            density_factor = density / (density.max() + 1e-8)
            k_adaptive = torch.clamp(5 + 10 * density_factor, 5, 20).int()
            # Use averaged k for consistent bandwidth adaptation
            k = int(torch.mean(k_adaptive.float()).item())
        else:
            k = base_k
            
        # Extract k-th nearest neighbor distances for bandwidth calibration
        # Enables adaptive kernel scaling based on local density patterns
        kth_dists, _ = torch.topk(dists, k, dim=1, largest=False)
        sigma = kth_dists[:, -1].view(-1, 1)  # Adaptive bandwidth parameter
        
        # Global kernel computation with comprehensive connectivity
        # Creates complete interaction matrix with adaptive bandwidth weighting
        kernel = torch.exp(-dists / (2 * sigma**2 + 1e-8))
        kernel = kernel / (torch.sum(kernel, dim=1, keepdim=True) + 1e-8)
        
        # Comprehensive drift calculation with global information integration
        # Weighted position averaging incorporating all data points
        drift = torch.matmul(kernel, x_centered)
        
        # Dynamic step size calculation (CORRECTED)
        alpha_val = self.alpha if isinstance(self.alpha, float) else self.alpha.item()
        step_size = self.dim**(-alpha_val)
        
        # Stochastic update with noise (OPTIMIZATION)
        noise = torch.randn_like(x_centered) * self.noise_scale
        h = x_centered + step_size * (drift - x_centered) + noise
        
        # Scale preservation (CORRECTED - this was completely missing)
        current_std = torch.std(h, dim=0, keepdim=True)
        h = h * (original_std / (current_std + 1e-8))
        
        # Add back the mean
        h = h + original_mean
        
        return h
    
    def compute_cost(self, original, embedded):
        """
        Compute cost function for optimization.
        """
        try:
            metrics = compute_metrics_optimized(original, embedded)
            # Weighted cost: balance distortion and local structure
            cost = 0.3 * metrics['distortion'] + 0.7 * (1 - metrics['local_structure'])
            return cost, metrics
        except:
            return 1.0, {'distortion': 1.0, 'local_structure': 0.0}
    
    def fit_transform(self, X):
        """
        Fit the model with optimizations: early stopping, adaptive parameters.
        """
        # Convert input to PyTorch tensor and move to device
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32)
        X = X.to(self.device)
        
        # Better initialization (OPTIMIZATION)
        n_samples = X.shape[0]
        if X.shape[1] <= self.dim:
            # If input dim <= target dim, pad with noise
            embedding = torch.cat([X, torch.randn(n_samples, self.dim - X.shape[1]) * 0.1], dim=1)
        else:
            # PCA-like initialization for better starting point
            U, S, V = torch.svd(X - X.mean(dim=0, keepdim=True))
            embedding = U[:, :self.dim] * S[:self.dim].sqrt()
        
        embedding = embedding.to(self.device)
        
        # Iterative updates with optimizations
        prev_cost = float('inf')
        patience = 5
        patience_counter = 0
        
        for iteration in range(self.max_iter):
            embedding_old = embedding.clone()
            embedding = self.forward(embedding)
            
            # Compute cost for early stopping and alpha adaptation
            if iteration % 5 == 0:  # Check cost every 5 iterations for efficiency
                try:
                    cost, metrics = self.compute_cost(X.cpu().numpy(), embedding.cpu().detach().numpy())
                    
                    # Store history
                    self.cost_history.append(cost)
                    alpha_val = self.alpha if isinstance(self.alpha, float) else self.alpha.item()
                    self.alpha_history.append(alpha_val)
                    
                    # Adaptive alpha adjustment (OPTIMIZATION)
                    if self.adaptive_params and isinstance(self.alpha, torch.nn.Parameter):
                        error = self.target_local_structure - metrics['local_structure']
                        # Gradient-based alpha update
                        with torch.no_grad():
                            self.alpha += self.eta * error
                            self.alpha.clamp_(0.01, 2.0)  # Reasonable bounds
                    
                    # Early stopping (OPTIMIZATION)
                    if iteration > 10:
                        if cost > prev_cost or abs(prev_cost - cost) < 1e-5:
                            patience_counter += 1
                        else:
                            patience_counter = 0
                        
                        if patience_counter >= patience:
                            print(f"Early stopping at iteration {iteration}, cost: {cost:.4f}")
                            break
                    
                    prev_cost = cost
                    
                except Exception as e:
                    print(f"Warning: Cost computation failed at iteration {iteration}: {e}")
            
            # Standard early stopping based on embedding change
            if torch.norm(embedding - embedding_old) < 1e-6:
                print(f"Convergence at iteration {iteration}")
                break
        
        return embedding.cpu().detach().numpy()
    
    def update_embedding(self, new_point, max_history=500):
        """
        Update existing embedding with a new data point for real-time streaming.
        
        Args:
            new_point: New data point to add (array-like)
            max_history: Maximum number of points to keep in memory
        
        Returns:
            Updated embedding as numpy array
        """
        # Convert new point to tensor
        if not torch.is_tensor(new_point):
            new_point = torch.tensor(new_point, dtype=torch.float32)
        
        # Ensure new_point is 2D
        if new_point.dim() == 1:
            new_point = new_point.unsqueeze(0)
        
        new_point = new_point.to(self.device)
        
        # Initialize if this is the first point
        if not hasattr(self, 'streaming_data') or not hasattr(self, 'streaming_embedding'):
            self.streaming_data = new_point
            self.streaming_embedding = torch.randn(1, self.dim) * 0.1
            return self.streaming_embedding.cpu().detach().numpy()
        
        # Add new point to streaming data
        self.streaming_data = torch.cat([self.streaming_data, new_point], dim=0)
        
        # Manage memory - keep only last max_history points
        if self.streaming_data.size(0) > max_history:
            self.streaming_data = self.streaming_data[-max_history:]
            self.streaming_embedding = self.streaming_embedding[-max_history:]
        
        # Quick embedding update for the new point
        # Option 1: Full recomputation (more accurate but slower)
        # self.streaming_embedding = torch.tensor(self.fit_transform(self.streaming_data.cpu().numpy()))
        
        # Option 2: Incremental update (faster for real-time)
        with torch.no_grad():
            # Initialize new embedding point near similar existing points
            if self.streaming_embedding.size(0) > 1:
                # Find similar points in input space
                distances = torch.cdist(new_point, self.streaming_data[:-1])
                _, nearest_idx = torch.topk(distances, k=min(3, self.streaming_data.size(0)-1), largest=False)
                
                # Initialize new embedding near nearest neighbors
                new_embedding = torch.mean(self.streaming_embedding[nearest_idx.flatten()], dim=0, keepdim=True)
                new_embedding += torch.randn_like(new_embedding) * 0.1  # Add some noise
            else:
                new_embedding = torch.randn(1, self.dim) * 0.1
            
            # Add to embedding
            self.streaming_embedding = torch.cat([self.streaming_embedding, new_embedding], dim=0)
            
            # Quick refinement - few iterations of the algorithm
            for _ in range(3):  # Just a few iterations for real-time performance
                self.streaming_embedding = self.forward(self.streaming_embedding)
        
        return self.streaming_embedding.cpu().detach().numpy()
    
    def reset_streaming(self):
        """Reset streaming state"""
        if hasattr(self, 'streaming_data'):
            del self.streaming_data
        if hasattr(self, 'streaming_embedding'):
            del self.streaming_embedding


def compute_metrics_optimized(original, embedded):
    """
    Optimized metrics computation with better numerical stability.
    """
    # Normalize data for better comparison
    from sklearn.preprocessing import StandardScaler
    
    scaler_orig = StandardScaler()
    scaler_emb = StandardScaler()
    
    original_norm = scaler_orig.fit_transform(original)
    embedded_norm = scaler_emb.fit_transform(embedded)
    
    # Compute distance matrices
    dist_orig = cdist(original_norm, original_norm)
    dist_emb = cdist(embedded_norm, embedded_norm)
    
    # Distortion as normalized difference
    distortion = np.mean(np.abs(dist_orig - dist_emb)) / (np.mean(dist_orig) + 1e-8)
    
    # Local structure preservation with more neighbors for stability
    k = min(10, len(original) - 1)
    nn_orig = np.argsort(dist_orig, axis=1)[:, 1:k+1]  # k nearest neighbors
    nn_emb = np.argsort(dist_emb, axis=1)[:, 1:k+1]
    
    local_structure = np.mean([len(np.intersect1d(nn_orig[i], nn_emb[i])) / k
                               for i in range(len(original))])
    
    return {'distortion': distortion, 'local_structure': local_structure}


# For backward compatibility, create a corrected version of the original
class NormalizedDynamicsCorrected(torch.nn.Module):
    """
    Corrected version of the original NormalizedDynamics with proper implementation.
    """
    
    def __init__(self, dim=2, alpha=1.0, max_iter=50):
        super().__init__()
        self.dim = dim
        self.alpha = alpha
        self.max_iter = max_iter

    def forward(self, x):
        # CORRECTED implementation based on discussion document
        original_mean = torch.mean(x, dim=0, keepdim=True)
        original_std = torch.std(x, dim=0, keepdim=True)

        x_centered = x - original_mean
        dists = torch.cdist(x_centered, x_centered)

        # CORRECTED: Use k-th nearest neighbor, not median
        k = min(15, x.size(0) - 1)
        kth_dists, _ = torch.topk(dists, k, dim=1, largest=False)
        sigma = kth_dists[:, -1].view(-1, 1)

        kernel = torch.exp(-dists / (2 * sigma**2))
        kernel = kernel / torch.sum(kernel, dim=1, keepdim=True)

        drift = torch.matmul(kernel, x_centered)

        # CORRECTED: Proper step size calculation
        step_size = self.dim**(-self.alpha)
        h = x_centered + step_size * (drift - x_centered)

        # CORRECTED: Scale preservation (this was missing!)
        h = h * (original_std / torch.std(h, dim=0, keepdim=True))
        h = h + original_mean

        return h

    def fit_transform(self, X):
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32)

        X_embedded = X.clone()
        for _ in range(self.max_iter):
            X_embedded = self.forward(X_embedded)
        return X_embedded.detach().numpy() 