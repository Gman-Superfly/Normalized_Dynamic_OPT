# Smart Sampling for Spatial Transcriptomics Data

## Overview

Smart Sampling is an intelligent data reduction system designed specifically for large-scale spatial transcriptomics datasets. Instead of randomly removing data points, it preserves important biological structure and spatial relationships while reducing computational complexity.

## The Problem

Large spatial transcriptomics datasets (50,000+ cells) present several challenges:

- **Computational Bottlenecks**: Analysis algorithms become slow and memory-intensive
- **Visualization Issues**: Too many points create cluttered, unreadable plots
- **Storage Constraints**: Large datasets are difficult to store and transfer
- **Interactive Analysis**: Real-time exploration becomes impossible

Traditional random sampling loses important biological information and spatial structure, making it unsuitable for biological data analysis.

## Smart Sampling Solutions

### Spatial stratified sampling

**Purpose**: Preserve tissue architecture and spatial relationships

**How it Works**:
- Divides the tissue into a spatial grid (default: 50×50 regions)
- Samples cells evenly from each spatial region
- Ensures no tissue areas are completely lost
- Maintains spatial gradients and boundaries

**Best For**:
- Datasets where spatial location is critical
- Tissue architecture analysis
- Spatial pattern discovery

```python
# Example: Sample 15,000 cells preserving spatial structure
sampler = BiologicalSampler(target_size=15000)
indices = sampler.spatial_stratified_sample(data, spatial_coords)
```

### Expression diversity sampling

**Purpose**: Preserve biological cell type diversity and gene expression patterns

**How it Works**:
- Performs clustering on gene expression data
- Uses top 2,000 most variable genes for efficiency
- Samples representative cells from each expression cluster
- Ensures rare cell types are retained

**Best For**:
- Cell type discovery
- Gene expression analysis
- When spatial information is not available

```python
# Example: Sample preserving expression diversity
indices = sampler.expression_diversity_sample(data, n_clusters=100)
```

### Hybrid sampling

**Purpose**: Combine the benefits of both spatial and expression sampling

**How it Works**:
- **Configurable weighting** between spatial and expression strategies (task-adaptive)
- **Default: 70% spatial, 30% expression sampling** (adjustable via `spatial_weight` parameter)
- First applies spatial sampling, then expression sampling on remaining cells
- Provides optimal balance for most use cases
- **Note**: The 70%/30% ratio can be adjusted to best suit your specific analysis task

**Best For**:
- General-purpose sampling
- When both spatial structure and cell diversity matter
- Most spatial transcriptomics analyses

```python
# Example: Hybrid sampling with custom weighting
indices = sampler.hybrid_sample(data, spatial_coords, spatial_weight=0.7)
```

## Usage Examples

### Quick Start

```python
from src.smart_sampling import smart_sample_visium_data

# Simple usage - automatically selects best method
adata_sampled, info = smart_sample_visium_data(
    adata,
    target_size=15000,
    method='hybrid'
)

print(f"Reduced from {info['original_size']} to {info['final_size']} cells")
```

### Advanced Usage

```python
from src.smart_sampling import BiologicalSampler

# Custom sampler with specific parameters
sampler = BiologicalSampler(target_size=10000, random_state=42)

# Method 1: Spatial sampling only
spatial_indices = sampler.spatial_stratified_sample(
    data, spatial_coords, grid_size=30
)

# Method 2: Expression sampling only  
expression_indices = sampler.expression_diversity_sample(
    data, n_clusters=50
)

# Method 3: Custom hybrid weighting
hybrid_indices = sampler.hybrid_sample(
    data, spatial_coords, spatial_weight=0.8  # 80% spatial
)
```

## Method Comparison

| Method | Preserves Spatial Structure | Preserves Cell Diversity | Speed | Best Use Case |
|--------|----------------------------|--------------------------|-------|---------------|
| **Random** | No | Poor | Fastest | Quick testing only |
| **Spatial** | Excellent | Moderate | Medium | Spatial analysis |
| **Expression** | No | Excellent | Medium | Cell type analysis |
| **Hybrid** | Good | Good | Medium | **General purpose** |

## Performance benefits

### Computational Speedup
- **Memory Usage**: Reduces RAM requirements proportionally
- **Algorithm Speed**: NormalizedDynamics runs ~3-5x faster on sampled data
- **Visualization**: Interactive plots become responsive

### Quality Preservation
- **Spatial Patterns**: Maintains tissue architecture and gradients
- **Cell Types**: Preserves rare and common cell populations
- **Biological Signal**: Retains meaningful gene expression patterns

## Integration with NormalizedDynamics

Smart sampling is particularly beneficial when combined with the NormalizedDynamics algorithm:

```python
# Workflow: Smart sampling + NormalizedDynamics
from src.smart_sampling import smart_sample_visium_data
from src.normalized_dynamics_optimized import NormalizedDynamicsOptimized

# 1. Smart sample large dataset
adata_sampled, info = smart_sample_visium_data(
    large_dataset, 
    target_size=15000,
    method='hybrid'
)

# 2. Apply NormalizedDynamics to sampled data
model = NormalizedDynamicsOptimized(dim=2, device='cuda')
embedding = model.fit_transform(adata_sampled.X)

# 3. Fast, high-quality results
print(f"Processed {info['final_size']} cells in record time!")
```

## Configuration Options

### BiologicalSampler Parameters

- **`target_size`**: Number of cells to sample (default: 15,000)
- **`random_state`**: Random seed for reproducibility (default: 42)

### Spatial Sampling Parameters

- **`grid_size`**: Spatial grid resolution (default: 50×50)
- Higher values = finer spatial resolution but smaller samples per region

### Expression Sampling Parameters  

- **`n_clusters`**: Number of expression clusters (default: 100)
- Higher values = better diversity preservation but smaller samples per cluster

### Hybrid Sampling Parameters

- **`spatial_weight`**: Balance between spatial/expression (default: 0.7)
- 0.0 = pure expression sampling
- 1.0 = pure spatial sampling
- 0.7 = 70% spatial, 30% expression (recommended)

## File Structure

```
src/
└── smart_sampling.py          # Main smart sampling implementation
    ├── BiologicalSampler      # Core sampling class
    ├── smart_sample_visium_data()  # High-level interface
    └── Testing code           # Built-in testing with synthetic data
```

## Dependencies

- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **scikit-learn**: Clustering and preprocessing
- **AnnData**: Spatial transcriptomics data format (for high-level interface)

## Best Practices

### When to Use Smart Sampling

Use when:
- Dataset has >20,000 cells
- Analysis is slow or memory-intensive
- You need interactive visualization
- Spatial structure is important

Do not use when:
- Dataset is already small (<10,000 cells)
- You need every single cell for analysis
- Random sampling is sufficient for your use case

### Recommended Workflows

1. **Exploratory Analysis**: Start with hybrid sampling (15k cells)
2. **Spatial Focus**: Use spatial sampling for tissue architecture studies
3. **Cell Type Focus**: Use expression sampling for cell classification
4. **Production**: Use hybrid sampling with optimized parameters

### Parameter Tuning

- **Small datasets** (20k-50k cells): `target_size=10000`
- **Medium datasets** (50k-100k cells): `target_size=15000` 
- **Large datasets** (100k+ cells): `target_size=20000`

Adjust `spatial_weight` based on your analysis goals:
- Spatial analysis: 0.8-0.9
- Cell type analysis: 0.3-0.5
- General analysis: 0.6-0.7

## Future Enhancements

Potential improvements to the smart sampling system:

- **Adaptive Sampling**: Automatically determine optimal target size
- **Multi-Modal Integration**: Incorporate protein expression data
- **Temporal Sampling**: Handle time-series spatial data
- **Interactive Selection**: GUI for manual region selection
- **Quality Metrics**: Automated assessment of sampling quality

---

**Note**: Smart sampling integrates with the NormalizedDynamics algorithm and other spatial transcriptomics analysis tools in this project through standard data interfaces. 