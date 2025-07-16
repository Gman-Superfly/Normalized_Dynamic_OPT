"""
Biological Metrics for Continuous Trajectory Preservation
========================================================

This module implements novel metrics specifically designed to evaluate how well
dimensionality reduction algorithms preserve continuous biological processes,
particularly developmental trajectories and spatial gradients.

These metrics address key limitations in current evaluation approaches that focus
on mathematical properties rather than biological accuracy.

Author: NormalizedDynamics Team
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.stats import spearmanr, pearsonr
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class BiologicalMetrics:
    """
    A comprehensive suite of biological accuracy metrics for evaluating
    dimensionality reduction algorithms on developmental biology data.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the biological metrics calculator.
        
        Args:
            verbose: Whether to print detailed metric explanations
        """
        self.verbose = verbose
        self.results_cache = {}
    
    def trajectory_coherence_score(self, 
                                 embedding: np.ndarray,
                                 developmental_time: np.ndarray,
                                 cell_types: Optional[np.ndarray] = None,
                                 n_neighbors: int = 10) -> Dict[str, float]:
        """
        Trajectory Coherence Score (TCS)
        
        Measures how well the embedding preserves known developmental ordering.
        High scores indicate that cells close in embedding space have similar
        developmental timing, preserving temporal relationships.
        
        Args:
            embedding: 2D embedding coordinates (n_cells, 2)
            developmental_time: Developmental time/pseudotime for each cell (n_cells,)
            cell_types: Optional cell type labels (n_cells,)
            n_neighbors: Number of neighbors to consider for local coherence
            
        Returns:
            Dictionary containing:
            - tcs_global: Global trajectory coherence (0-1, higher better)
            - tcs_local: Local neighborhood coherence (0-1, higher better)
            - time_correlation: Correlation between embedding distance and time difference
            - smoothness_index: Ratio of intra-stage to inter-stage distances
        """
        n_cells = len(embedding)
        
        # 1. Global coherence: correlation between embedding distance and time difference
        embedding_distances = pdist(embedding)
        time_differences = pdist(developmental_time.reshape(-1, 1))
        
        # Use Spearman correlation (robust to non-linear relationships)
        time_correlation, _ = spearmanr(embedding_distances, time_differences)
        
        # 2. Local coherence: for each cell, check if neighbors have similar developmental time
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(embedding)
        _, neighbor_indices = nbrs.kneighbors(embedding)
        
        local_coherences = []
        for i in range(n_cells):
            cell_time = developmental_time[i]
            neighbor_times = developmental_time[neighbor_indices[i][1:]]  # Exclude self
            
            # Calculate temporal coherence in neighborhood
            time_variance = np.var(neighbor_times)
            max_possible_variance = np.var(developmental_time)
            
            # Coherence is inverse of normalized variance
            coherence = 1 - (time_variance / max_possible_variance) if max_possible_variance > 0 else 1
            local_coherences.append(coherence)
        
        tcs_local = np.mean(local_coherences)
        
        # 3. Smoothness index: intra-stage vs inter-stage distances
        if cell_types is not None:
            intra_distances = []
            inter_distances = []
            
            for cell_type in np.unique(cell_types):
                type_mask = cell_types == cell_type
                type_embedding = embedding[type_mask]
                
                if len(type_embedding) > 1:
                    # Intra-type distances
                    intra_dist = pdist(type_embedding)
                    intra_distances.extend(intra_dist)
                
                # Inter-type distances
                other_mask = cell_types != cell_type
                if np.any(other_mask):
                    other_embedding = embedding[other_mask]
                    inter_dist = cdist(type_embedding, other_embedding).flatten()
                    inter_distances.extend(inter_dist)
            
            if inter_distances and intra_distances:
                smoothness_index = np.mean(intra_distances) / np.mean(inter_distances)
            else:
                smoothness_index = 1.0
        else:
            smoothness_index = None
        
        # 4. Global TCS: weighted combination of coherence measures
        tcs_global = 0.6 * max(0, time_correlation) + 0.4 * tcs_local
        
        results = {
            'tcs_global': tcs_global,
            'tcs_local': tcs_local,
            'time_correlation': time_correlation,
            'smoothness_index': smoothness_index
        }
        
        if self.verbose:
            print(f"Trajectory Coherence Score: {tcs_global:.3f}")
            print(f"  - Local coherence: {tcs_local:.3f}")
            print(f"  - Time correlation: {time_correlation:.3f}")
            if smoothness_index is not None:
                print(f"  - Smoothness index: {smoothness_index:.3f}")
        
        return results
    
    def bifurcation_preservation_score(self,
                                     embedding: np.ndarray,
                                     cell_types: np.ndarray,
                                     bifurcation_hierarchy: Dict[str, List[str]],
                                     developmental_time: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Bifurcation Preservation Score (BPS)
        
        Measures how well the embedding preserves branching developmental processes.
        Good algorithms should maintain smooth branching topology rather than
        creating discrete clusters.
        
        Args:
            embedding: 2D embedding coordinates (n_cells, 2)
            cell_types: Cell type labels (n_cells,)
            bifurcation_hierarchy: Dict mapping parent types to child types
                e.g., {'progenitor': ['neuron', 'glia'], 'neuron': ['excitatory', 'inhibitory']}
            developmental_time: Optional developmental time for each cell
            
        Returns:
            Dictionary containing:
            - bps_global: Global bifurcation preservation score (0-1, higher better)
            - branch_connectivity: How well branches are connected to parents
            - transition_smoothness: Smoothness of transitions at bifurcation points
            - topology_preservation: Overall topology preservation score
        """
        # 1. Identify bifurcation points and branches
        branch_connectivities = []
        transition_smoothnesses = []
        
        for parent_type, child_types in bifurcation_hierarchy.items():
            if parent_type not in cell_types or len(child_types) < 2:
                continue
                
            parent_mask = cell_types == parent_type
            parent_embedding = embedding[parent_mask]
            
            if len(parent_embedding) == 0:
                continue
            
            # Calculate connectivity between parent and each child
            for child_type in child_types:
                if child_type not in cell_types:
                    continue
                    
                child_mask = cell_types == child_type
                child_embedding = embedding[child_mask]
                
                if len(child_embedding) == 0:
                    continue
                
                # Measure connectivity: minimum distance between parent and child populations
                distances = cdist(parent_embedding, child_embedding)
                min_distance = np.min(distances)
                
                # Normalize by typical intra-population distance
                parent_intra_dist = np.mean(pdist(parent_embedding)) if len(parent_embedding) > 1 else 1.0
                child_intra_dist = np.mean(pdist(child_embedding)) if len(child_embedding) > 1 else 1.0
                typical_intra_dist = (parent_intra_dist + child_intra_dist) / 2
                
                connectivity = np.exp(-min_distance / typical_intra_dist) if typical_intra_dist > 0 else 0
                branch_connectivities.append(connectivity)
        
        # 2. Measure transition smoothness at bifurcation points
        for parent_type, child_types in bifurcation_hierarchy.items():
            if parent_type not in cell_types or len(child_types) < 2:
                continue
                
            # Find cells near bifurcation point
            parent_mask = cell_types == parent_type
            child_masks = [cell_types == child_type for child_type in child_types if child_type in cell_types]
            
            if not child_masks or len(child_masks) < 2:
                continue
            
            # Calculate smoothness: how gradually do we transition from parent to children
            parent_center = np.mean(embedding[parent_mask], axis=0) if np.any(parent_mask) else np.array([0, 0])
            child_centers = [np.mean(embedding[mask], axis=0) for mask in child_masks if np.any(mask)]
            
            if len(child_centers) >= 2:
                # Measure angle between branches (wider angles = better preserved bifurcation)
                vectors = [center - parent_center for center in child_centers]
                angles = []
                for i in range(len(vectors)):
                    for j in range(i+1, len(vectors)):
                        v1, v2 = vectors[i], vectors[j]
                        if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                            cos_angle = np.clip(cos_angle, -1, 1)
                            angle = np.arccos(cos_angle)
                            angles.append(angle)
                
                if angles:
                    # Good bifurcations have angles close to optimal (e.g., 120° for 3 branches)
                    mean_angle = np.mean(angles)
                    optimal_angle = np.pi / len(child_types)  # Rough estimate
                    smoothness = 1 - abs(mean_angle - optimal_angle) / np.pi
                    transition_smoothnesses.append(max(0, smoothness))
        
        # 3. Calculate overall scores
        branch_connectivity = np.mean(branch_connectivities) if branch_connectivities else 0
        transition_smoothness = np.mean(transition_smoothnesses) if transition_smoothnesses else 0
        
        # Topology preservation: combination of connectivity and smoothness
        topology_preservation = 0.6 * branch_connectivity + 0.4 * transition_smoothness
        
        # Global BPS
        bps_global = topology_preservation
        
        results = {
            'bps_global': bps_global,
            'branch_connectivity': branch_connectivity,
            'transition_smoothness': transition_smoothness,
            'topology_preservation': topology_preservation
        }
        
        if self.verbose:
            print(f"Bifurcation Preservation Score: {bps_global:.3f}")
            print(f"  - Branch connectivity: {branch_connectivity:.3f}")
            print(f"  - Transition smoothness: {transition_smoothness:.3f}")
        
        return results
    
    def spatial_gradient_preservation(self,
                                    embedding: np.ndarray,
                                    spatial_coordinates: np.ndarray,
                                    gene_expression: Optional[np.ndarray] = None,
                                    layer_labels: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Spatial Gradient Preservation (SGP)
        
        Measures how well the embedding preserves spatial gradients, particularly
        important for spatial transcriptomics data with known tissue architecture.
        
        Args:
            embedding: 2D embedding coordinates (n_spots, 2)
            spatial_coordinates: Original spatial coordinates (n_spots, 2)
            gene_expression: Optional gene expression matrix (n_spots, n_genes)
            layer_labels: Optional layer/region labels (n_spots,)
            
        Returns:
            Dictionary containing:
            - sgp_global: Global spatial gradient preservation (0-1, higher better)
            - distance_correlation: Correlation between spatial and embedding distances
            - gradient_preservation: How well expression gradients are preserved
            - layer_organization: How well spatial layers are organized
        """
        n_spots = len(embedding)
        
        # 1. Distance correlation: spatial vs embedding distances
        spatial_distances = pdist(spatial_coordinates)
        embedding_distances = pdist(embedding)
        
        distance_correlation, _ = spearmanr(spatial_distances, embedding_distances)
        distance_correlation = max(0, distance_correlation)  # Only positive correlations are meaningful
        
        # 2. Gene expression gradient preservation
        gradient_preservation = 0.0
        if gene_expression is not None:
            # For each gene, measure how well its spatial gradient is preserved
            gene_correlations = []
            
            # Sample a subset of genes to avoid computational overload
            n_genes = gene_expression.shape[1]
            sample_genes = np.random.choice(n_genes, min(50, n_genes), replace=False)
            
            for gene_idx in sample_genes:
                gene_values = gene_expression[:, gene_idx]
                
                # Skip genes with no variation
                if np.var(gene_values) == 0:
                    continue
                
                # Calculate pairwise expression differences
                expr_differences = pdist(gene_values.reshape(-1, 1))
                
                # Correlation between embedding distance and expression difference
                expr_corr, _ = spearmanr(embedding_distances, expr_differences)
                if not np.isnan(expr_corr):
                    gene_correlations.append(abs(expr_corr))  # Use absolute correlation
            
            gradient_preservation = np.mean(gene_correlations) if gene_correlations else 0
        
        # 3. Layer organization (if layer labels provided)
        layer_organization = 0.0
        if layer_labels is not None:
            # Measure how well layers are spatially organized in embedding
            unique_layers = np.unique(layer_labels)
            if len(unique_layers) > 1:
                layer_scores = []
                
                for layer in unique_layers:
                    layer_mask = layer_labels == layer
                    layer_embedding = embedding[layer_mask]
                    
                    if len(layer_embedding) < 2:
                        continue
                    
                    # Calculate compactness of layer in embedding
                    layer_center = np.mean(layer_embedding, axis=0)
                    layer_distances = np.linalg.norm(layer_embedding - layer_center, axis=1)
                    layer_compactness = 1 / (1 + np.mean(layer_distances))
                    
                    layer_scores.append(layer_compactness)
                
                layer_organization = np.mean(layer_scores) if layer_scores else 0
        
        # 4. Global SGP score
        weights = [0.4, 0.3, 0.3]  # distance_correlation, gradient_preservation, layer_organization
        values = [distance_correlation, gradient_preservation, layer_organization]
        
        # Only use available metrics
        available_weights = []
        available_values = []
        
        if distance_correlation > 0:
            available_weights.append(weights[0])
            available_values.append(values[0])
        
        if gene_expression is not None and gradient_preservation > 0:
            available_weights.append(weights[1])
            available_values.append(values[1])
        
        if layer_labels is not None:
            available_weights.append(weights[2])
            available_values.append(values[2])
        
        if available_weights:
            # Normalize weights
            available_weights = np.array(available_weights)
            available_weights = available_weights / np.sum(available_weights)
            sgp_global = np.sum(np.array(available_values) * available_weights)
        else:
            sgp_global = distance_correlation
        
        results = {
            'sgp_global': sgp_global,
            'distance_correlation': distance_correlation,
            'gradient_preservation': gradient_preservation,
            'layer_organization': layer_organization
        }
        
        if self.verbose:
            print(f"Spatial Gradient Preservation: {sgp_global:.3f}")
            print(f"  - Distance correlation: {distance_correlation:.3f}")
            print(f"  - Gradient preservation: {gradient_preservation:.3f}")
            print(f"  - Layer organization: {layer_organization:.3f}")
        
        return results
    
    def fragmentation_penalty(self,
                            embedding: np.ndarray,
                            true_trajectories: Union[np.ndarray, List[np.ndarray]],
                            cell_types: Optional[np.ndarray] = None,
                            developmental_time: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Fragmentation Penalty (FP)
        
        Penalizes methods that artificially fragment continuous biological processes
        into discrete clusters. Lower scores indicate better preservation of continuity.
        
        Args:
            embedding: 2D embedding coordinates (n_cells, 2)
            true_trajectories: True continuous trajectories (labels or trajectory assignments)
            cell_types: Optional cell type labels
            developmental_time: Optional developmental time
            
        Returns:
            Dictionary containing:
            - fragmentation_penalty: Global fragmentation penalty (0-1, lower better)
            - artificial_clusters: Number of artificial clusters detected
            - continuity_breaks: Number of continuity breaks in trajectories
            - cluster_purity: Purity of clusters with respect to true trajectories
        """
        n_cells = len(embedding)
        
        # 1. Detect artificial clusters using density-based clustering
        from sklearn.cluster import DBSCAN
        
        # Use DBSCAN to find dense regions (potential artificial clusters)
        eps_values = np.percentile(pdist(embedding), [10, 20, 30])
        best_clustering = None
        best_score = -1
        
        for eps in eps_values:
            clustering = DBSCAN(eps=eps, min_samples=max(3, n_cells // 50)).fit(embedding)
            n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
            
            if n_clusters > 1:
                # Score based on silhouette-like metric
                if hasattr(true_trajectories, '__len__') and len(true_trajectories) == n_cells:
                    from sklearn.metrics import adjusted_rand_score
                    score = adjusted_rand_score(true_trajectories, clustering.labels_)
                    if score > best_score:
                        best_score = score
                        best_clustering = clustering
        
        artificial_clusters = 0
        if best_clustering is not None:
            artificial_clusters = len(set(best_clustering.labels_)) - (1 if -1 in best_clustering.labels_ else 0)
        
        # 2. Count continuity breaks
        continuity_breaks = 0
        if developmental_time is not None:
            # Sort cells by developmental time
            time_order = np.argsort(developmental_time)
            ordered_embedding = embedding[time_order]
            
            # Calculate consecutive distances in embedding
            consecutive_distances = np.linalg.norm(np.diff(ordered_embedding, axis=0), axis=1)
            
            # Identify breaks (distances much larger than typical)
            median_distance = np.median(consecutive_distances)
            break_threshold = median_distance * 3  # 3x median as threshold
            continuity_breaks = np.sum(consecutive_distances > break_threshold)
        
        # 3. Cluster purity with respect to true trajectories
        cluster_purity = 0.0
        if best_clustering is not None and hasattr(true_trajectories, '__len__'):
            cluster_labels = best_clustering.labels_
            unique_clusters = set(cluster_labels) - {-1}  # Exclude noise points
            
            if unique_clusters:
                purities = []
                for cluster_id in unique_clusters:
                    cluster_mask = cluster_labels == cluster_id
                    cluster_trajectories = np.array(true_trajectories)[cluster_mask]
                    
                    if len(cluster_trajectories) > 0:
                        # Calculate purity as fraction of most common trajectory
                        unique_trajs, counts = np.unique(cluster_trajectories, return_counts=True)
                        purity = np.max(counts) / len(cluster_trajectories)
                        purities.append(purity)
                
                cluster_purity = np.mean(purities) if purities else 0
        
        # 4. Calculate fragmentation penalty
        # Higher artificial clusters = higher penalty
        # More continuity breaks = higher penalty
        # Lower cluster purity = higher penalty
        
        cluster_penalty = min(1.0, artificial_clusters / max(1, len(np.unique(true_trajectories)) if hasattr(true_trajectories, '__len__') else 1))
        break_penalty = min(1.0, continuity_breaks / max(1, n_cells // 10))
        purity_penalty = 1 - cluster_purity
        
        fragmentation_penalty = (cluster_penalty + break_penalty + purity_penalty) / 3
        
        results = {
            'fragmentation_penalty': fragmentation_penalty,
            'artificial_clusters': artificial_clusters,
            'continuity_breaks': continuity_breaks,
            'cluster_purity': cluster_purity
        }
        
        if self.verbose:
            print(f"Fragmentation Penalty: {fragmentation_penalty:.3f} (lower is better)")
            print(f"  - Artificial clusters: {artificial_clusters}")
            print(f"  - Continuity breaks: {continuity_breaks}")
            print(f"  - Cluster purity: {cluster_purity:.3f}")
        
        return results
    
    def comprehensive_biological_evaluation(self,
                                          embedding: np.ndarray,
                                          developmental_time: Optional[np.ndarray] = None,
                                          cell_types: Optional[np.ndarray] = None,
                                          spatial_coordinates: Optional[np.ndarray] = None,
                                          gene_expression: Optional[np.ndarray] = None,
                                          bifurcation_hierarchy: Optional[Dict[str, List[str]]] = None,
                                          layer_labels: Optional[np.ndarray] = None) -> Dict[str, Dict[str, float]]:
        """
        Run comprehensive biological evaluation using all available metrics.
        
        Args:
            embedding: 2D embedding coordinates
            developmental_time: Developmental time/pseudotime
            cell_types: Cell type labels
            spatial_coordinates: Original spatial coordinates (for spatial transcriptomics)
            gene_expression: Gene expression matrix
            bifurcation_hierarchy: Bifurcation hierarchy for developmental data
            layer_labels: Layer/region labels for spatial data
            
        Returns:
            Dictionary containing results from all applicable metrics
        """
        results = {}
        
        print("Running Comprehensive Biological Evaluation...")
        print("=" * 50)
        
        # 1. Trajectory Coherence Score
        if developmental_time is not None:
            print("\n1. Trajectory Coherence Analysis:")
            results['trajectory_coherence'] = self.trajectory_coherence_score(
                embedding, developmental_time, cell_types
            )
        
        # 2. Bifurcation Preservation Score
        if bifurcation_hierarchy is not None and cell_types is not None:
            print("\n2. Bifurcation Preservation Analysis:")
            results['bifurcation_preservation'] = self.bifurcation_preservation_score(
                embedding, cell_types, bifurcation_hierarchy, developmental_time
            )
        
        # 3. Spatial Gradient Preservation
        if spatial_coordinates is not None:
            print("\n3. Spatial Gradient Preservation Analysis:")
            results['spatial_gradient'] = self.spatial_gradient_preservation(
                embedding, spatial_coordinates, gene_expression, layer_labels
            )
        
        # 4. Fragmentation Penalty
        if developmental_time is not None or cell_types is not None:
            print("\n4. Fragmentation Analysis:")
            true_trajectories = developmental_time if developmental_time is not None else cell_types
            results['fragmentation'] = self.fragmentation_penalty(
                embedding, true_trajectories, cell_types, developmental_time
            )
        
        # 5. Generate summary
        print("\n" + "=" * 50)
        print("BIOLOGICAL EVALUATION SUMMARY:")
        print("=" * 50)
        
        for metric_name, metric_results in results.items():
            if metric_name == 'trajectory_coherence':
                score = metric_results.get('tcs_global', 0)
                print(f"Trajectory Coherence Score: {score:.3f}")
            elif metric_name == 'bifurcation_preservation':
                score = metric_results.get('bps_global', 0)
                print(f"Bifurcation Preservation Score: {score:.3f}")
            elif metric_name == 'spatial_gradient':
                score = metric_results.get('sgp_global', 0)
                print(f"Spatial Gradient Preservation: {score:.3f}")
            elif metric_name == 'fragmentation':
                score = metric_results.get('fragmentation_penalty', 1)
                print(f"Fragmentation Penalty: {score:.3f} (lower better)")
        
        return results


def generate_synthetic_developmental_data(n_cells: int = 1000, 
                                        n_genes: int = 100,
                                        n_branches: int = 3,
                                        noise_level: float = 0.1,
                                        random_seed: int = 42) -> Dict[str, np.ndarray]:
    """
    Generate synthetic developmental biology data for testing biological metrics.
    
    Args:
        n_cells: Number of cells to generate
        n_genes: Number of genes
        n_branches: Number of developmental branches
        noise_level: Amount of noise to add
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing synthetic data components
    """
    np.random.seed(random_seed)
    
    # Generate developmental trajectory
    t = np.linspace(0, 1, n_cells)
    
    # Create branching structure
    branch_points = np.linspace(0.2, 0.8, n_branches - 1)
    cell_types = np.zeros(n_cells, dtype=int)
    
    for i, branch_point in enumerate(branch_points):
        branch_mask = t > branch_point
        cell_types[branch_mask] = i + 1
    
    # Generate 2D trajectory with branches
    x = np.zeros(n_cells)
    y = np.zeros(n_cells)
    
    for branch in range(n_branches):
        branch_mask = cell_types == branch
        branch_t = t[branch_mask]
        
        if len(branch_t) > 0:
            # Different trajectory for each branch
            if branch == 0:  # Main trunk
                x[branch_mask] = branch_t
                y[branch_mask] = 0.1 * np.sin(4 * np.pi * branch_t)
            else:  # Branches
                angle = 2 * np.pi * branch / n_branches
                x[branch_mask] = branch_t * np.cos(angle)
                y[branch_mask] = branch_t * np.sin(angle)
    
    # Add noise
    x += np.random.normal(0, noise_level, n_cells)
    y += np.random.normal(0, noise_level, n_cells)
    
    # Generate gene expression correlated with trajectory
    gene_expression = np.zeros((n_cells, n_genes))
    
    for i in range(n_genes):
        # Some genes correlate with time
        if i < n_genes // 3:
            gene_expression[:, i] = t + np.random.normal(0, 0.1, n_cells)
        # Some genes are branch-specific
        elif i < 2 * n_genes // 3:
            branch = i % n_branches
            gene_expression[:, i] = (cell_types == branch).astype(float) + np.random.normal(0, 0.1, n_cells)
        # Rest are random
        else:
            gene_expression[:, i] = np.random.normal(0, 1, n_cells)
    
    # Create bifurcation hierarchy
    bifurcation_hierarchy = {
        '0': [str(i) for i in range(1, n_branches)]
    }
    
    return {
        'coordinates': np.column_stack([x, y]),
        'developmental_time': t,
        'cell_types': cell_types.astype(str),
        'gene_expression': gene_expression,
        'bifurcation_hierarchy': bifurcation_hierarchy
    }


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Biological Metrics Suite")
    print("=" * 50)
    
    # Generate synthetic data
    data = generate_synthetic_developmental_data(n_cells=500, random_seed=42)
    
    # Initialize metrics calculator
    metrics = BiologicalMetrics(verbose=True)
    
    # Run comprehensive evaluation
    results = metrics.comprehensive_biological_evaluation(
        embedding=data['coordinates'],
        developmental_time=data['developmental_time'],
        cell_types=data['cell_types'],
        gene_expression=data['gene_expression'],
        bifurcation_hierarchy=data['bifurcation_hierarchy']
    )
    
    print("\nTest completed successfully!") 