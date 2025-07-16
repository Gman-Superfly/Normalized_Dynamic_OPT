"""
Synthetic Developmental Trajectory Datasets
==========================================

This module creates biologically realistic synthetic developmental datasets with known ground truth
to demonstrate algorithmic superiority in developmental biology applications.

These datasets are specifically designed to highlight NormalizedDynamics' strengths:
1. Global connectivity preservation during branching events
2. Smooth gradient maintenance across developmental time  
3. Continuous trajectory preservation (vs. artificial clustering)
4. Multi-scale structure handling (local + global relationships)

Author: NormalizedDynamics Team
Date: 2024
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class SyntheticDevelopmentalDatasets:
    """
    Generator for synthetic developmental biology datasets with known ground truth.
    
    Each dataset is designed to test specific aspects of developmental trajectory preservation:
    - Hematopoietic: Multi-lineage branching from single progenitor
    - Neural Crest: Complex multi-branching with spatial organization
    - Spatial Layers: Gradient-based spatial transcriptomics simulation
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the synthetic dataset generator.
        
        Args:
            random_seed: Random seed for reproducible dataset generation
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def create_hematopoietic_differentiation(self, 
                                           n_cells: int = 3000,
                                           n_genes: int = 2000,
                                           noise_level: float = 0.15) -> Dict[str, np.ndarray]:
        """
        Create synthetic hematopoietic stem cell differentiation dataset.
        
        Models the well-established hematopoietic hierarchy:
        HSC → Common Myeloid Progenitor (CMP) / Common Lymphoid Progenitor (CLP)
            → Specialized cells (neutrophils, monocytes, T-cells, B-cells, etc.)
        
        This dataset tests:
        - Bifurcation preservation during lineage commitment
        - Global trajectory coherence across multiple branches
        - Proper temporal ordering maintenance
        
        Args:
            n_cells: Number of cells to generate
            n_genes: Number of genes/features
            noise_level: Amount of biological noise to add
            
        Returns:
            Dictionary containing:
            - X: Gene expression matrix (n_cells, n_genes)
            - cell_types: Cell type annotations
            - true_pseudotime: Ground truth developmental time
            - lineage_labels: Lineage assignment for each cell
            - spatial_coordinates: True underlying trajectory coordinates
            - bifurcation_tree: Hierarchical structure of development
        """
        np.random.seed(self.random_seed)
        
        # Define hematopoietic hierarchy
        lineage_structure = {
            'HSC': {'proportion': 0.15, 'pseudotime_range': (0.0, 0.2), 'parent': None},
            'CMP': {'proportion': 0.12, 'pseudotime_range': (0.15, 0.4), 'parent': 'HSC'},
            'CLP': {'proportion': 0.12, 'pseudotime_range': (0.15, 0.4), 'parent': 'HSC'},
            'Neutrophil': {'proportion': 0.18, 'pseudotime_range': (0.35, 0.8), 'parent': 'CMP'},
            'Monocyte': {'proportion': 0.15, 'pseudotime_range': (0.35, 0.8), 'parent': 'CMP'},
            'T_cell': {'proportion': 0.14, 'pseudotime_range': (0.35, 0.8), 'parent': 'CLP'},
            'B_cell': {'proportion': 0.14, 'pseudotime_range': (0.35, 0.8), 'parent': 'CLP'}
        }
        
        # Generate cells for each lineage
        X = np.zeros((n_cells, n_genes))
        cell_types = []
        true_pseudotime = []
        lineage_labels = []
        spatial_coords = []
        
        cell_idx = 0
        
        for cell_type, info in lineage_structure.items():
            n_cells_type = int(n_cells * info['proportion'])
            if cell_idx + n_cells_type > n_cells:
                n_cells_type = n_cells - cell_idx
            
            if n_cells_type == 0:
                continue
                
            # Generate pseudotime for this cell type
            pt_min, pt_max = info['pseudotime_range']
            pseudotime = np.random.uniform(pt_min, pt_max, n_cells_type)
            
            # Generate spatial trajectory coordinates
            if cell_type == 'HSC':
                # Root at origin
                x_coord = np.zeros(n_cells_type)
                y_coord = np.zeros(n_cells_type)
            elif cell_type in ['CMP', 'CLP']:
                # Early branching
                angle = 0.3 if cell_type == 'CMP' else -0.3
                x_coord = pseudotime * np.cos(angle) + np.random.normal(0, 0.1, n_cells_type)
                y_coord = pseudotime * np.sin(angle) + np.random.normal(0, 0.1, n_cells_type)
            else:
                # Terminal differentiation
                lineage = 'myeloid' if info['parent'] == 'CMP' else 'lymphoid'
                if lineage == 'myeloid':
                    base_angle = 0.5 if cell_type == 'Neutrophil' else 0.1
                else:
                    base_angle = -0.1 if cell_type == 'T_cell' else -0.5
                
                x_coord = pseudotime * np.cos(base_angle) + np.random.normal(0, 0.15, n_cells_type)
                y_coord = pseudotime * np.sin(base_angle) + np.random.normal(0, 0.15, n_cells_type)
            
            spatial_coords.extend(list(zip(x_coord, y_coord)))
            
            # Generate gene expression profiles
            for i in range(n_cells_type):
                expr_profile = self._generate_hematopoietic_expression(
                    cell_type, pseudotime[i], n_genes, noise_level
                )
                X[cell_idx + i] = expr_profile
            
            # Store annotations
            cell_types.extend([cell_type] * n_cells_type)
            true_pseudotime.extend(pseudotime.tolist())
            
            # Determine lineage
            if cell_type in ['HSC']:
                lineage = 'stem'
            elif cell_type in ['CMP', 'Neutrophil', 'Monocyte']:
                lineage = 'myeloid'
            else:
                lineage = 'lymphoid'
            lineage_labels.extend([lineage] * n_cells_type)
            
            cell_idx += n_cells_type
            
            if cell_idx >= n_cells:
                break
        
        # Ensure we have exactly n_cells
        if cell_idx < n_cells:
            # Fill remaining with HSCs
            remaining = n_cells - cell_idx
            expr_profiles = [self._generate_hematopoietic_expression('HSC', 0.1, n_genes, noise_level) 
                           for _ in range(remaining)]
            X[cell_idx:] = np.array(expr_profiles)
            cell_types.extend(['HSC'] * remaining)
            true_pseudotime.extend([0.1] * remaining)
            lineage_labels.extend(['stem'] * remaining)
            spatial_coords.extend([(0, 0)] * remaining)
        
        # Create bifurcation tree structure
        bifurcation_tree = {
            'HSC': ['CMP', 'CLP'],
            'CMP': ['Neutrophil', 'Monocyte'],
            'CLP': ['T_cell', 'B_cell']
        }
        
        # Convert spatial coordinates to array
        spatial_coordinates = np.array(spatial_coords)
        
        return {
            'X': X,
            'cell_types': np.array(cell_types),
            'true_pseudotime': np.array(true_pseudotime),
            'lineage_labels': np.array(lineage_labels),
            'spatial_coordinates': spatial_coordinates,
            'bifurcation_tree': bifurcation_tree,
            'dataset_name': 'hematopoietic_differentiation',
            'description': 'Synthetic hematopoietic stem cell differentiation with multi-lineage branching'
        }
    
    def create_neural_crest_development(self,
                                      n_cells: int = 2500,
                                      n_genes: int = 1800,
                                      noise_level: float = 0.18) -> Dict[str, np.ndarray]:
        """
        Create synthetic neural crest cell development dataset.
        
        Models neural crest cell migration and differentiation:
        Neural Crest → Multiple specialized cell types (neurons, glia, melanocytes, etc.)
        
        This dataset tests:
        - Complex multi-branching preservation
        - Spatial organization during migration
        - Multiple terminal cell fate maintenance
        
        Args:
            n_cells: Number of cells to generate
            n_genes: Number of genes/features
            noise_level: Amount of biological noise to add
            
        Returns:
            Dictionary with same structure as hematopoietic dataset
        """
        np.random.seed(self.random_seed + 1)  # Slightly different seed
        
        # Define neural crest hierarchy
        cell_fate_structure = {
            'Neural_Crest': {'proportion': 0.20, 'pseudotime_range': (0.0, 0.25), 'parent': None},
            'Sensory_Neuron': {'proportion': 0.18, 'pseudotime_range': (0.2, 0.8), 'parent': 'Neural_Crest'},
            'Sympathetic_Neuron': {'proportion': 0.15, 'pseudotime_range': (0.2, 0.8), 'parent': 'Neural_Crest'},
            'Schwann_Cell': {'proportion': 0.14, 'pseudotime_range': (0.25, 0.85), 'parent': 'Neural_Crest'},
            'Melanocyte': {'proportion': 0.12, 'pseudotime_range': (0.3, 0.9), 'parent': 'Neural_Crest'},
            'Smooth_Muscle': {'proportion': 0.11, 'pseudotime_range': (0.25, 0.85), 'parent': 'Neural_Crest'},
            'Adrenal_Medulla': {'proportion': 0.10, 'pseudotime_range': (0.3, 0.9), 'parent': 'Neural_Crest'}
        }
        
        X = np.zeros((n_cells, n_genes))
        cell_types = []
        true_pseudotime = []
        spatial_coords = []
        migration_paths = []
        
        cell_idx = 0
        
        for cell_type, info in cell_fate_structure.items():
            n_cells_type = int(n_cells * info['proportion'])
            if cell_idx + n_cells_type > n_cells:
                n_cells_type = n_cells - cell_idx
                
            if n_cells_type == 0:
                continue
            
            # Generate pseudotime
            pt_min, pt_max = info['pseudotime_range']
            pseudotime = np.random.uniform(pt_min, pt_max, n_cells_type)
            
            # Generate spatial migration trajectories
            if cell_type == 'Neural_Crest':
                # Start at neural tube
                x_coord = np.random.normal(0, 0.1, n_cells_type)
                y_coord = np.random.normal(0, 0.1, n_cells_type)
                migration_path = 'origin'
            else:
                # Different migration paths for different cell types
                path_angles = {
                    'Sensory_Neuron': 0.0,       # Dorsal migration
                    'Sympathetic_Neuron': 0.8,   # Ventral migration
                    'Schwann_Cell': -0.3,        # Lateral migration
                    'Melanocyte': 1.2,           # Ventro-lateral migration
                    'Smooth_Muscle': -0.8,       # Medial migration
                    'Adrenal_Medulla': 1.5       # Deep ventral migration
                }
                
                angle = path_angles[cell_type]
                migration_distance = pseudotime * 2  # Scale migration with time
                
                x_coord = migration_distance * np.cos(angle) + np.random.normal(0, 0.2, n_cells_type)
                y_coord = migration_distance * np.sin(angle) + np.random.normal(0, 0.2, n_cells_type)
                migration_path = f'path_{cell_type.lower()}'
            
            spatial_coords.extend(list(zip(x_coord, y_coord)))
            migration_paths.extend([migration_path] * n_cells_type)
            
            # Generate gene expression
            for i in range(n_cells_type):
                expr_profile = self._generate_neural_crest_expression(
                    cell_type, pseudotime[i], n_genes, noise_level
                )
                X[cell_idx + i] = expr_profile
            
            cell_types.extend([cell_type] * n_cells_type)
            true_pseudotime.extend(pseudotime.tolist())
            cell_idx += n_cells_type
            
            if cell_idx >= n_cells:
                break
        
        # Fill remaining cells if needed
        if cell_idx < n_cells:
            remaining = n_cells - cell_idx
            expr_profiles = [self._generate_neural_crest_expression('Neural_Crest', 0.1, n_genes, noise_level) 
                           for _ in range(remaining)]
            X[cell_idx:] = np.array(expr_profiles)
            cell_types.extend(['Neural_Crest'] * remaining)
            true_pseudotime.extend([0.1] * remaining)
            spatial_coords.extend([(0, 0)] * remaining)
            migration_paths.extend(['origin'] * remaining)
        
        # Create complex bifurcation tree
        bifurcation_tree = {
            'Neural_Crest': ['Sensory_Neuron', 'Sympathetic_Neuron', 'Schwann_Cell', 
                           'Melanocyte', 'Smooth_Muscle', 'Adrenal_Medulla']
        }
        
        spatial_coordinates = np.array(spatial_coords)
        
        return {
            'X': X,
            'cell_types': np.array(cell_types),
            'true_pseudotime': np.array(true_pseudotime),
            'lineage_labels': np.array(migration_paths),
            'spatial_coordinates': spatial_coordinates,
            'bifurcation_tree': bifurcation_tree,
            'dataset_name': 'neural_crest_development',
            'description': 'Synthetic neural crest development with complex multi-branching and migration'
        }
    
    def create_spatial_cortical_layers(self,
                                     n_cells: int = 2800,
                                     n_genes: int = 1500,
                                     noise_level: float = 0.12) -> Dict[str, np.ndarray]:
        """
        Create synthetic spatial transcriptomics dataset simulating cortical layer development.
        
        Models cortical layer formation with spatial gradients:
        Layer I (superficial) → Layer VI (deep) with smooth transitions
        
        This dataset tests:
        - Spatial gradient preservation
        - Layer-specific gene expression maintenance
        - Smooth spatial transitions
        
        Args:
            n_cells: Number of cells to generate
            n_genes: Number of genes/features
            noise_level: Amount of spatial noise to add
            
        Returns:
            Dictionary with spatial transcriptomics data structure
        """
        np.random.seed(self.random_seed + 2)
        
        # Define cortical layer structure
        layer_structure = {
            'Layer_I': {'proportion': 0.10, 'depth_range': (0.0, 0.15), 'spatial_y': (0.85, 1.0)},
            'Layer_II_III': {'proportion': 0.25, 'depth_range': (0.1, 0.35), 'spatial_y': (0.65, 0.9)},
            'Layer_IV': {'proportion': 0.20, 'depth_range': (0.3, 0.55), 'spatial_y': (0.45, 0.7)},
            'Layer_V': {'proportion': 0.25, 'depth_range': (0.5, 0.75), 'spatial_y': (0.25, 0.5)},
            'Layer_VI': {'proportion': 0.20, 'depth_range': (0.7, 1.0), 'spatial_y': (0.0, 0.3)}
        }
        
        X = np.zeros((n_cells, n_genes))
        cell_types = []
        spatial_coordinates = []
        depth_positions = []
        layer_gradients = []
        
        cell_idx = 0
        
        for layer, info in layer_structure.items():
            n_cells_layer = int(n_cells * info['proportion'])
            if cell_idx + n_cells_layer > n_cells:
                n_cells_layer = n_cells - cell_idx
                
            if n_cells_layer == 0:
                continue
            
            # Generate spatial positions
            depth_min, depth_max = info['depth_range']
            y_min, y_max = info['spatial_y']
            
            # Depth represents developmental time/position
            depth = np.random.uniform(depth_min, depth_max, n_cells_layer)
            
            # Spatial coordinates with realistic tissue organization
            x_coord = np.random.uniform(-1, 1, n_cells_layer)  # Horizontal spread
            y_coord = np.random.uniform(y_min, y_max, n_cells_layer)  # Layer position
            
            # Add spatial noise
            x_coord += np.random.normal(0, noise_level, n_cells_layer)
            y_coord += np.random.normal(0, noise_level * 0.5, n_cells_layer)  # Less vertical noise
            
            spatial_coordinates.extend(list(zip(x_coord, y_coord)))
            depth_positions.extend(depth.tolist())
            
            # Generate layer-specific gene expression
            for i in range(n_cells_layer):
                expr_profile = self._generate_spatial_layer_expression(
                    layer, depth[i], y_coord[i], n_genes, noise_level
                )
                X[cell_idx + i] = expr_profile
            
            cell_types.extend([layer] * n_cells_layer)
            
            # Create gradient labels for each layer
            gradient_value = (depth_max + depth_min) / 2  # Average depth for layer
            layer_gradients.extend([gradient_value] * n_cells_layer)
            
            cell_idx += n_cells_layer
            
            if cell_idx >= n_cells:
                break
        
        # Fill remaining cells
        if cell_idx < n_cells:
            remaining = n_cells - cell_idx
            expr_profiles = [self._generate_spatial_layer_expression('Layer_II_III', 0.2, 0.75, n_genes, noise_level) 
                           for _ in range(remaining)]
            X[cell_idx:] = np.array(expr_profiles)
            cell_types.extend(['Layer_II_III'] * remaining)
            spatial_coordinates.extend([(0, 0.75)] * remaining)
            depth_positions.extend([0.2] * remaining)
            layer_gradients.extend([0.2] * remaining)
        
        # Spatial transcriptomics doesn't have traditional bifurcation tree
        # Instead, we have layer organization
        layer_organization = {
            'superficial': ['Layer_I', 'Layer_II_III'],
            'middle': ['Layer_IV'],
            'deep': ['Layer_V', 'Layer_VI']
        }
        
        spatial_coordinates = np.array(spatial_coordinates)
        
        return {
            'X': X,
            'cell_types': np.array(cell_types),
            'true_pseudotime': np.array(depth_positions),  # Depth as pseudotime
            'lineage_labels': np.array(layer_gradients),   # Gradient values
            'spatial_coordinates': spatial_coordinates,
            'bifurcation_tree': layer_organization,
            'dataset_name': 'spatial_cortical_layers',
            'description': 'Synthetic spatial transcriptomics with cortical layer gradients'
        }
    
    def _generate_hematopoietic_expression(self, cell_type: str, pseudotime: float, 
                                         n_genes: int, noise_level: float) -> np.ndarray:
        """Generate realistic gene expression profile for hematopoietic cells."""
        expression = np.random.normal(0, 1, n_genes)
        
        # Cell type-specific expression patterns
        if cell_type == 'HSC':
            # Stem cell markers (high expression in early genes)
            expression[:100] += 2 * (1 - pseudotime) + np.random.normal(0, 0.2, 100)
        elif cell_type in ['CMP', 'CLP']:
            # Progenitor markers
            expression[100:200] += 1.5 + np.random.normal(0, 0.3, 100)
        elif cell_type in ['Neutrophil', 'Monocyte']:
            # Myeloid markers
            expression[200:350] += 2 * pseudotime + np.random.normal(0, 0.3, 150)
        else:  # Lymphoid cells
            # Lymphoid markers
            expression[350:500] += 2 * pseudotime + np.random.normal(0, 0.3, 150)
        
        # Add temporal dynamics
        expression += pseudotime * np.random.normal(0, 0.5, n_genes)
        
        # Add noise
        expression += np.random.normal(0, noise_level, n_genes)
        
        return expression
    
    def _generate_neural_crest_expression(self, cell_type: str, pseudotime: float,
                                        n_genes: int, noise_level: float) -> np.ndarray:
        """Generate realistic gene expression profile for neural crest cells."""
        expression = np.random.normal(0, 1, n_genes)
        
        # Neural crest-specific patterns
        if cell_type == 'Neural_Crest':
            # Neural crest markers
            expression[:150] += 2 * (1 - pseudotime) + np.random.normal(0, 0.2, 150)
        elif 'Neuron' in cell_type:
            # Neuronal markers
            expression[150:300] += 2 * pseudotime + np.random.normal(0, 0.3, 150)
        elif cell_type == 'Schwann_Cell':
            # Glial markers
            expression[300:450] += 1.8 * pseudotime + np.random.normal(0, 0.3, 150)
        elif cell_type == 'Melanocyte':
            # Pigmentation markers
            expression[450:600] += 2.2 * pseudotime + np.random.normal(0, 0.3, 150)
        else:
            # Other derivatives
            expression[600:750] += 1.5 * pseudotime + np.random.normal(0, 0.3, 150)
        
        # Migration-related expression
        expression += pseudotime * np.random.normal(0, 0.4, n_genes)
        expression += np.random.normal(0, noise_level, n_genes)
        
        return expression
    
    def _generate_spatial_layer_expression(self, layer: str, depth: float, y_position: float,
                                         n_genes: int, noise_level: float) -> np.ndarray:
        """Generate realistic gene expression profile for spatial layer data."""
        expression = np.random.normal(0, 1, n_genes)
        
        # Layer-specific expression
        layer_genes = {
            'Layer_I': (0, 100),
            'Layer_II_III': (100, 300),
            'Layer_IV': (300, 500),
            'Layer_V': (500, 700),
            'Layer_VI': (700, 900)
        }
        
        if layer in layer_genes:
            start, end = layer_genes[layer]
            expression[start:end] += 2 + np.random.normal(0, 0.3, end - start)
        
        # Spatial gradients
        gradient_genes = n_genes // 3
        expression[:gradient_genes] += y_position * 1.5  # Superficial-deep gradient
        expression[gradient_genes:2*gradient_genes] += (1 - y_position) * 1.5  # Deep-superficial gradient
        
        # Depth-related expression (developmental time)
        expression += depth * np.random.normal(0, 0.4, n_genes)
        expression += np.random.normal(0, noise_level, n_genes)
        
        return expression


def create_all_synthetic_datasets(random_seed: int = 42) -> Dict[str, Dict]:
    """
    Create all synthetic developmental datasets for comprehensive testing.
    
    Args:
        random_seed: Random seed for reproducible generation
        
    Returns:
        Dictionary containing all generated datasets
    """
    generator = SyntheticDevelopmentalDatasets(random_seed=random_seed)
    
    datasets = {}
    
    print("Generating synthetic developmental datasets...")
    
    # Hematopoietic differentiation
    print("  Creating hematopoietic differentiation dataset...")
    datasets['hematopoietic'] = generator.create_hematopoietic_differentiation()
    
    # Neural crest development
    print("  Creating neural crest development dataset...")
    datasets['neural_crest'] = generator.create_neural_crest_development()
    
    # Spatial cortical layers
    print("  Creating spatial cortical layers dataset...")
    datasets['spatial_layers'] = generator.create_spatial_cortical_layers()
    
    print("All synthetic datasets generated successfully!")
    
    return datasets


if __name__ == "__main__":
    # Test dataset generation
    datasets = create_all_synthetic_datasets(random_seed=42)
    
    for name, data in datasets.items():
        print(f"\n{name.upper()} DATASET:")
        print(f"  Shape: {data['X'].shape}")
        print(f"  Cell types: {len(np.unique(data['cell_types']))}")
        print(f"  Pseudotime range: {data['true_pseudotime'].min():.3f} - {data['true_pseudotime'].max():.3f}")
        print(f"  Description: {data['description']}") 