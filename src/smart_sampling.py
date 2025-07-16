"""
Smart Sampling Strategies for Large-Scale Spatial Transcriptomics Data

This module provides various sampling strategies to reduce dataset size while
preserving biological structure and spatial relationships.
CURRENTLY IN TEST MODE UNUSED IN PUBLISHED SCRIPTS
"""

import numpy as np
import polars as pl
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import time

class BiologicalSampler:
    """
    Intelligent sampling strategies that preserve biological structure.
    """
    
    def __init__(self, target_size=15000, random_state=42):
        self.target_size = target_size
        self.random_state = random_state
        np.random.seed(random_state)
    
    def spatial_stratified_sample(self, data, spatial_coords, grid_size=50):
        """
        Sample evenly across spatial regions to preserve tissue architecture.
        
        Args:
            data: Gene expression matrix (n_cells x n_genes)
            spatial_coords: Spatial coordinates (n_cells x 2)
            grid_size: Number of spatial grid cells per dimension
            
        Returns:
            sampled_indices: Indices of selected cells
        """
        print(f"Performing spatial stratified sampling from {data.shape[0]} cells...")
        
        # Create spatial grid
        x_min, x_max = spatial_coords[:, 0].min(), spatial_coords[:, 0].max()
        y_min, y_max = spatial_coords[:, 1].min(), spatial_coords[:, 1].max()
        
        x_bins = np.linspace(x_min, x_max, grid_size + 1)
        y_bins = np.linspace(y_min, y_max, grid_size + 1)
        
        # Assign cells to grid cells
        x_indices = np.digitize(spatial_coords[:, 0], x_bins) - 1
        y_indices = np.digitize(spatial_coords[:, 1], y_bins) - 1
        
        # Sample evenly from each grid cell
        sampled_indices = []
        cells_per_grid = max(1, self.target_size // (grid_size * grid_size))
        
        for i in range(grid_size):
            for j in range(grid_size):
                mask = (x_indices == i) & (y_indices == j)
                cell_indices = np.where(mask)[0]
                
                if len(cell_indices) > 0:
                    # Sample from this grid cell
                    n_sample = min(cells_per_grid, len(cell_indices))
                    selected = np.random.choice(cell_indices, n_sample, replace=False)
                    sampled_indices.extend(selected)
        
        # If we haven't reached target size, randomly sample more
        if len(sampled_indices) < self.target_size:
            remaining = self.target_size - len(sampled_indices)
            all_indices = set(range(data.shape[0]))
            unused_indices = list(all_indices - set(sampled_indices))
            
            if len(unused_indices) > 0:
                additional = np.random.choice(unused_indices, 
                                            min(remaining, len(unused_indices)), 
                                            replace=False)
                sampled_indices.extend(additional)
        
        sampled_indices = np.array(sampled_indices[:self.target_size])
        print(f"   Selected {len(sampled_indices)} cells preserving spatial structure")
        
        return sampled_indices
    
    def expression_diversity_sample(self, data, n_clusters=100):
        """
        Sample to preserve gene expression diversity using clustering.
        
        Args:
            data: Gene expression matrix (n_cells x n_genes)
            n_clusters: Number of expression clusters to identify
            
        Returns:
            sampled_indices: Indices of selected cells
        """
        print(f"Performing expression diversity sampling from {data.shape[0]} cells...")
        
        # Use highly variable genes for clustering (faster)
        if data.shape[1] > 2000:
            # Simple variance-based gene selection
            gene_vars = np.var(data, axis=0)
            top_genes = np.argsort(gene_vars)[-2000:]
            data_subset = data[:, top_genes]
        else:
            data_subset = data
        
        # Standardize data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_subset)
        
        # MiniBatch K-means for speed
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, 
                                random_state=self.random_state,
                                batch_size=1000)
        cluster_labels = kmeans.fit_predict(data_scaled)
        
        # Sample evenly from each cluster
        sampled_indices = []
        cells_per_cluster = max(1, self.target_size // n_clusters)
        
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) > 0:
                n_sample = min(cells_per_cluster, len(cluster_indices))
                selected = np.random.choice(cluster_indices, n_sample, replace=False)
                sampled_indices.extend(selected)
        
        # Fill to target size if needed
        if len(sampled_indices) < self.target_size:
            remaining = self.target_size - len(sampled_indices)
            all_indices = set(range(data.shape[0]))
            unused_indices = list(all_indices - set(sampled_indices))
            
            if len(unused_indices) > 0:
                additional = np.random.choice(unused_indices,
                                            min(remaining, len(unused_indices)),
                                            replace=False)
                sampled_indices.extend(additional)
        
        sampled_indices = np.array(sampled_indices[:self.target_size])
        print(f"   Selected {len(sampled_indices)} cells preserving expression diversity")
        
        return sampled_indices
    
    def hybrid_sample(self, data, spatial_coords, spatial_weight=0.7):
        """
        Combine spatial and expression diversity sampling.
        
        Args:
            data: Gene expression matrix
            spatial_coords: Spatial coordinates
            spatial_weight: Weight for spatial vs expression sampling (0-1)
            
        Returns:
            sampled_indices: Indices of selected cells
        """
        print(f"Performing hybrid sampling with {spatial_weight:.1%} spatial weighting...")
        
        # Split target size between strategies
        spatial_target = int(self.target_size * spatial_weight)
        expression_target = self.target_size - spatial_target
        
        # Spatial sampling
        if spatial_target > 0:
            spatial_indices = self.spatial_stratified_sample(
                data, spatial_coords, grid_size=int(np.sqrt(spatial_target))
            )[:spatial_target]
        else:
            spatial_indices = np.array([], dtype=int)
        
        # Expression sampling from remaining cells
        if expression_target > 0:
            remaining_mask = np.ones(data.shape[0], dtype=bool)
            if len(spatial_indices) > 0:
                remaining_mask[spatial_indices] = False
            
            remaining_data = data[remaining_mask]
            remaining_original_indices = np.where(remaining_mask)[0]
            
            if len(remaining_data) > 0:
                # Temporarily adjust target for remaining data
                temp_sampler = BiologicalSampler(target_size=expression_target, 
                                               random_state=self.random_state)
                expression_relative_indices = temp_sampler.expression_diversity_sample(
                    remaining_data, n_clusters=min(50, expression_target)
                )
                expression_indices = remaining_original_indices[expression_relative_indices]
            else:
                expression_indices = np.array([], dtype=int)
        else:
            expression_indices = np.array([], dtype=int)
        
        # Combine indices
        sampled_indices = np.concatenate([spatial_indices, expression_indices])
        
        print(f"   Combined sampling: {len(spatial_indices)} spatial + {len(expression_indices)} expression")
        
        return sampled_indices

def smart_sample_visium_data(adata, target_size=15000, method='hybrid', spatial_weight=0.7):
    """
    High-level function to intelligently sample Visium data.
    
    Args:
        adata: AnnData object with expression and spatial data
        target_size: Target number of cells to sample
        method: 'spatial', 'expression', 'hybrid', or 'random'
        spatial_weight: For hybrid method, weight of spatial vs expression
        
    Returns:
        adata_sampled: Subsampled AnnData object
        sample_info: Dictionary with sampling information
    """
    start_time = time.time()
    
    if adata.shape[0] <= target_size:
        print(f"Dataset already small enough ({adata.shape[0]} cells), no sampling needed")
        return adata, {"method": "none", "original_size": adata.shape[0], "final_size": adata.shape[0]}
    
    print(f"Smart sampling {adata.shape[0]} cells → {target_size} cells using '{method}' method")
    
    # Get expression data
    if hasattr(adata.X, 'toarray'):
        X = adata.X.toarray()
    else:
        X = adata.X
    
    # Initialize sampler
    sampler = BiologicalSampler(target_size=target_size)
    
    # Apply sampling method
    if method == 'random':
        sampled_indices = np.random.choice(adata.shape[0], target_size, replace=False)
        print(f"   Random sampling: {len(sampled_indices)} cells")
        
    elif method == 'spatial':
        if 'spatial' not in adata.obsm:
            raise ValueError("Spatial coordinates not found in adata.obsm['spatial']")
        spatial_coords = adata.obsm['spatial']
        sampled_indices = sampler.spatial_stratified_sample(X, spatial_coords)
        
    elif method == 'expression':
        sampled_indices = sampler.expression_diversity_sample(X)
        
    elif method == 'hybrid':
        if 'spatial' not in adata.obsm:
            print("   Warning: No spatial coordinates found, falling back to expression sampling")
            sampled_indices = sampler.expression_diversity_sample(X)
        else:
            spatial_coords = adata.obsm['spatial']
            sampled_indices = sampler.hybrid_sample(X, spatial_coords, spatial_weight)
            
    else:
        raise ValueError(f"Unknown sampling method: {method}")
    
    # Create subsampled dataset
    adata_sampled = adata[sampled_indices].copy()
    
    sampling_time = time.time() - start_time
    
    sample_info = {
        "method": method,
        "original_size": adata.shape[0],
        "final_size": len(sampled_indices),
        "sampling_time": sampling_time,
        "sampled_indices": sampled_indices
    }
    
    print(f"   Sampling complete in {sampling_time:.2f}s")
    print(f"   Size reduction: {adata.shape[0]:,} → {len(sampled_indices):,} ({len(sampled_indices)/adata.shape[0]*100:.1f}%)")
    
    return adata_sampled, sample_info

# Example usage and testing
if __name__ == "__main__":
    # Test with synthetic data
    print("Testing smart sampling strategies...")
    
    # Create synthetic spatial transcriptomics data
    n_cells = 50000
    n_genes = 2000
    
    # Synthetic expression data
    X_synthetic = np.random.lognormal(0, 1, (n_cells, n_genes))
    
    # Synthetic spatial coordinates with structure
    spatial_synthetic = np.column_stack([
        np.random.normal(0, 100, n_cells),
        np.random.normal(0, 100, n_cells)
    ])
    
    print(f"Created synthetic dataset: {X_synthetic.shape}")
    
    # Test sampling methods
    sampler = BiologicalSampler(target_size=5000)
    
    # Test spatial sampling
    spatial_indices = sampler.spatial_stratified_sample(X_synthetic, spatial_synthetic)
    print(f"Spatial sampling result: {len(spatial_indices)} cells")
    
    # Test expression sampling  
    expression_indices = sampler.expression_diversity_sample(X_synthetic)
    print(f"Expression sampling result: {len(expression_indices)} cells")
    
    # Test hybrid sampling
    hybrid_indices = sampler.hybrid_sample(X_synthetic, spatial_synthetic)
    print(f"Hybrid sampling result: {len(hybrid_indices)} cells")
    
    print("All sampling methods tested successfully!") 