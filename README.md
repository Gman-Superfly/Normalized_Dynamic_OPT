# NormalizedDynamics: A Kernel-Based Manifold Learning Algorithm

A specialized dimensionality reduction algorithm designed for preserving continuous relationships in scientific data, with particular focus on trajectory analysis and self stabilization.

## Overview

NormalizedDynamics is a kernel-based iterative manifold learning algorithm that addresses specific challenges in dimensionality reduction, for example  in biological data maintaining continuous trajectories is important,  it offers a specialized approach to dimensionality reduction that is useful for certain scientific applications.

### Key Characteristics

- **Studied Application**: Single-cell developmental biology and continuous biological processes
- **Broader Applications**: RL, CS, Astronomical surveys, time-series analysis, scientific data visualization
- **Approach**: Adaptive kernel bandwidth with global connectivity preservation
- **Scale**: Optimized for realtime and small to medium datasets (100-5000 samples) for speed, larger datasets will take long processing time but yeald lowest distrotion.
- **Philosophy**: Prioritizes geometric relationship preservation with self correction.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/normdyn.git
cd normdyn

# Install dependencies
pip install -r requirements.txt
```

### Core Dependencies
- PyTorch (GPU support recommended)
- NumPy, SciPy, scikit-learn
- Flask (for web interface)
- Polars (efficient data processing)
- Matplotlib, Seaborn (visualization)

### Extras for Test Dependencies
- scanpy (biological analysis)
- UMAP (for comparisons)
- astroquery/astropy (GAIA data)

## Quick Start

### Basic Usage

```python
from src.normalized_dynamics_optimized import NormalizedDynamicsOptimized

# Initialize with default parameters
nd = NormalizedDynamicsOptimized(
    dim=2,                    # Target dimensions
    k=20,                     # Base number of neighbors change according to your data
    alpha=1.0,                # Bandwidth parameter
    max_iter=50,              # Maximum iterations
    adaptive_params=True,     # Enable adaptive optimization
    device='cpu'              # or 'cuda' for GPU
)

# Fit and transform your data
X_embedded = nd.fit_transform(X)
```

### Smart K Adaptation

```python
from src.normalized_dynamics_smart_k import create_smart_k_algorithm

# Automatically configure based on dataset characteristics
nd_smart = create_smart_k_algorithm(
    dataset_size=len(X),
    strategy='smart',
    device='cpu'
)

X_embedded = nd_smart.fit_transform(X)
```

### Real-Time Streaming

For sensor data or real-time applications:

```python
# Initialize for streaming
nd = NormalizedDynamicsOptimized(dim=2)

# Process streaming data
for new_point in data_stream:
    embedding = nd.update_embedding(new_point, max_history=500)
```

## Algorithm Overview

### Mathematical Foundation

The algorithm uses iterative dynamics with kernel-weighted drift:

```
h^(t+1) = h^(t) + α × Δt × (G[h^(t)] - h^(t)) + η
```

Where:
- `G[h]`: Kernel-weighted drift function (neighborhood consensus)
- `α`: Adaptive step size parameter
- `Δt`: Dimension-dependent time step
- `η`: Optional exploration noise

### Mathematical Foundation

**Problem Formulation**: Given high-dimensional data X ∈ ℝ^(n×d), find embedding H ∈ ℝ^(n×k) where k < d that preserves:
- Geometric relationships: ||x_i - x_j|| ≈ ||h_i - h_j||
- Local neighborhoods: N_K(x_i) ≈ N_K(h_i) for K-nearest neighbors

The core update rule follows a kernel-weighted drift equation:

```
h^(t+1) = h^(t) + α × Δt × (G[h^(t)] - h^(t)) + η
```

Where:
- `G[h]`: Kernel-weighted drift function (neighborhood consensus)
- `α`: Adaptive step size parameter
- `Δt`: Dimension-dependent time step = d^(-α)
- `η`: Optional stochastic noise term

**Drift Function**: Kernel-weighted center of mass incorporating all data points:
```
G[h_i] = Σⱼ K(h_i, h_j) × h_j / Σⱼ K(h_i, h_j)
```
This ensures comprehensive information integration across the entire dataset.

### Key Technical Features

1. **Adaptive Kernel Bandwidth**: Uses k-th nearest neighbor distances for local density adaptation
   ```
   σ_i = ||h_i - h_i^(k)||₂
   K(h_i, h_j) = exp(-||h_i - h_j||²/(2σ_i²))
   ```

2. **Scale Preservation**: Maintains feature-wise standard deviations during iteration
   ```
   h ← h × (σ_original / σ_current)
   ```

3. **Global Connectivity**: Considers all pairwise relationships (O(n²) complexity)
   - Comprehensive information integration across entire dataset
   - Complete interaction matrix with adaptive bandwidth weighting

4. **Multi-Criteria Convergence**: Cost-based and stability-based stopping criteria

5. **Free Energy Principle Emergence**: The algorithm naturally implements FEP through its mathematical structure

### Theoretical Foundation: Free Energy Principle

The algorithm naturally implements the Free Energy Principle through gradient descent on:

```math
F[H] = U[H] - T·S[H]
```

Where:
- **Energy term**: `U[H] = ½∑ᵢ ||hᵢ - δᵢ||²` (prediction error)
- **Entropy term**: `S[H] = -∑ᵢⱼ p(j|i) log p(j|i)` (neighborhood uncertainty)

**Mathematical Correspondence**:
1. **Probabilistic Belief Formation**: `p(j|i) = exp(-||h_i - h_j||²/(2σ_i²)) / Z_i`
2. **Prediction Generation**: `δ_i = Σⱼ p(j|i) h_j`
3. **Error Minimization**: `h ← h + α(δ - h)` minimizes prediction error

This emergence provides theoretical foundation for understanding why the algorithm works well for biological trajectory preservation.

### Computational Characteristics

**Performance by Dataset Size**:

- **Small datasets (≤2000 samples)**: Excellent efficiency with rapid convergence, real-time capability
- **Medium datasets (2000-5000 samples)**: Enhanced accuracy through global connectivity
- **Large datasets (>5000 samples)**: Maximum precision with quadratic computational requirements

**Complexity**:
- Time: O(n²d) per iteration
- Space: O(n²) for distance and kernel matrices
- Scaling: Prioritizes accuracy through comprehensive connectivity

## Comprehensive Test Suite

We've implemented extensive testing to validate the algorithm across multiple domains and ensure reproducible results.

### Test Coverage

```
tests/
├── test_normalized_dynamics.py          # Core algorithm validation
├── test_comprehensive_visualizations.py # Visual comparison testing
├── test_biological_metrics.py          # Standard biological evaluation
├── test_enhanced_biological_metrics.py # Advanced DPT-based metrics
├── test_pancreas_endocrinogenesis.py   # Single-cell developmental data
├── test_gaia_data.py                   # Astronomical survey data
├── test_wine_dataset.py                # Chemical classification
├── test_mouse_brain_cortical.py        # Neural tissue analysis
├── test_synthetic_developmental.py     # Ground truth validation
├── synthetic_developmental_datasets.py # Synthetic biology generators
└── smart_sampling_enhanced_analysis.py # Sampling strategy analysis
```

### Biological Validation

- **Pancreas Endocrinogenesis**: Real single-cell RNA-seq developmental trajectory
- **Diffusion Pseudotime (DPT)**: Standard computational biology evaluation using scanpy
- **Multi-Scale Analysis**: Trajectory coherence across multiple neighborhood scales
- **Bifurcation Preservation**: Assessment of developmental branching point accuracy

### Benchmark Datasets

- **Synthetic**: Swiss Roll, Two Moons, Multi-Scale Circles, Clustered Data
- **Biological**: Pancreas development, Mouse brain cortical tissue
- **Astronomical**: GAIA stellar survey data
- **Chemical**: Wine classification dataset

### Running Tests

```bash
# Run all tests
python src/run_tests.py

# Run specific test categories
python tests/test_pancreas_endocrinogenesis.py     # Biological validation
python tests/test_gaia_data.py                     # Astronomical data
python tests/test_comprehensive_visualizations.py  # Visual comparisons
```

Results are automatically saved in `static/results/` with timestamp-based naming.

## Performance Characteristics

Based on comprehensive testing across multiple datasets:

### Empirical Performance Characteristics
- **Geometric Distortion**: 0.001-0.024 (good preservation of pairwise distances)
- **Local Structure Preservation**: 46-85% depending on data complexity and intrinsic dimensionality
- **Computational Efficiency**: Sub-second to few-second embedding times for datasets ≤2000 samples
- **Biological Trajectories**: 0.660 trajectory smoothness on pancreas endocrinogenesis data
- **Optimal Domain**: Small to medium-scale datasets where precision is prioritized

### Strengths
- **Global Structure**: Good preservation of large-scale spatial relationships
- **Continuous Processes**: Avoids artificial fragmentation common in t-SNE/UMAP
- **Adaptive Behavior**: Responds to local data density and manifold characteristics
- **Real-Time Capability**: Interactive applications and live monitoring support
- **Global Structure Preservation**: Maintains physically meaningful relationships in scientific data
  - Preserves smooth gradients and continuous distributions
  - Adapts embedding complexity to data density (intrinsic dimensionality detection)
  - Suitable for applications where global geometric relationships are scientifically important

### Limitations
- **Computational Complexity**: O(n²) scaling limits applicability to large datasets
- **Local Structure**: 46-85% preservation (dataset dependent, sometimes lower than t-SNE/UMAP)
- **3D Manifolds**: Challenging for complex 3D → 2D unfolding (e.g., Swiss Roll)
- **Runtime**: Slower than t-SNE/UMAP for large datasets due to global connectivity

### When to Use
- **Single-cell developmental biology**: RNA-seq trajectory analysis, stem cell research
- **Astronomical data**: Stellar surveys with continuous distributions (GAIA)
- **Real-time applications**: Sensor monitoring, interactive visualization (<2000 samples)
- **Scientific datasets**: Any data requiring continuous relationship preservation
- **Time-series analysis**: Biological processes, temporal dynamics

### When to Consider Alternatives
- Large datasets (>5000 samples) where speed is critical
- Maximum local structure preservation is required
- Complex 3D manifold unfolding tasks
- Image or text embeddings (specialized methods more appropriate)

## Web Interface

Launch the interactive demonstration:

```bash
python app.py
```

Navigate to `http://localhost:5000` for:
- **Algorithm Comparisons**: Side-by-side evaluation against t-SNE and UMAP
- **Dataset Analyses**: Pancreas, GAIA, Wine, and more
- **Real-Time Demos**: Streaming sensor simulations
- **Smart Sampling**: Impact of different sampling strategies
- **Interactive Visualizations**: Explore embeddings in real-time
- **and more in web UI...
    

## Smart Sampling for Large Datasets

For large spatial transcriptomics datasets (50,000+ cells), we provide intelligent sampling strategies that preserve biological structure:

### Sampling Strategies

1. **Spatial Stratified Sampling**: Preserves tissue architecture by sampling evenly across spatial regions
2. **Expression Diversity Sampling**: Maintains biological cell type diversity and rare cell populations
3. **Hybrid Sampling**: Combines spatial and expression strategies for optimal preservation

### Usage

```python
from src.smart_sampling import BiologicalSampler

# Initialize sampler
sampler = BiologicalSampler(target_size=15000)

# Apply smart sampling strategies
spatial_indices = sampler.spatial_stratified_sample(data, spatial_coords)
expression_indices = sampler.expression_diversity_sample(data)
hybrid_indices = sampler.hybrid_sample(data, spatial_coords)
```

### Performance Results

Smart sampling with dynamic K adaptation shows measurable improvements:

| Strategy | Trajectory Smoothness | Local Structure | Runtime |
|----------|----------------------|-----------------|---------|
| Random Sampling | 0.447 | 0.993 | 106.1s |
| Smart + Dynamic K | **0.480** | 0.993 | 130.4s |
| Hybrid + Dynamic K | **0.429** | 0.993 | 128.4s |

## Results and Evaluation

### Evaluation Methodology

We employ sound evaluation metrics following computational biology standards:

- **Diffusion Pseudotime (DPT)**: Gold-standard trajectory inference (Haghverdi et al. 2016)
- **Geometric Distortion**: Normalized mean squared error of distance matrices
- **Local Structure Preservation**: k-nearest neighbor overlap preservation
- **Trajectory Smoothness**: Correlation between embedding distances and developmental time

### Sample Results

| Dataset | Geometric Distortion | Local Structure | Trajectory Smoothness |
|---------|---------------------|-----------------|----------------------|
| Pancreas Development | 0.0089 | 0.710 | 0.660 |
| GAIA Stellar Data | 0.0156 | 0.680 | N/A |
| Wine Classification | 0.0034 | 0.850 | N/A |
| Multi-Scale Circles | 0.0012 | 0.920 | N/A |

*Note: Results may vary depending on dataset characteristics and parameter settings.*

## Algorithm Parameters

### Core Parameters
- `dim`: Target embedding dimensions (default: 2)
- `k`: Base number of neighbors (automatically adapted)
- `alpha`: Bandwidth scaling (default: 1.0, adaptive)
- `max_iter`: Maximum iterations (default: 50)
- `noise_scale`: Stochastic noise level (default: 0.01)

### Adaptive Features
- **Smart K**: Automatic adjustment based on dataset size and density
- **Dynamic Alpha**: Adapts to achieve target local structure preservation
- **Early Stopping**: Cost-based convergence detection

## Project Structure

```
normdyn/
├── src/                    # Core algorithm implementations
├── tests/                  # Comprehensive test suite
├── templates/              # Web interface templates
├── static/                 # CSS, JS, and results
├── data/                   # Datasets (biological, astronomical)
├── docs/                   # Documentation
├── app.py                  # Flask web application
└── requirements.txt        # Dependencies
```

## Documentation

- **Technical Documentation**: [`docs/NormalizedDynamics_Technical_Documentation.py`](docs/NormalizedDynamics_Technical_Documentation.py) - Complete mathematical specification
- **Methodology**: [`docs/METHODOLOGY_TRANSPARENCY.md`](docs/METHODOLOGY_TRANSPARENCY.md) - Evaluation framework
- **Test Documentation**: [`docs/README_tests.md`](docs/README_tests.md) - Test infrastructure overview
- **Project Organization**: [`docs/PROJECT_ORGANIZATION_PLAN.md`](docs/PROJECT_ORGANIZATION_PLAN.md) - Repository structure

## Future Directions

- **Theoretical Analysis**: K-independence properties need investingating as they are interesting and self stabilizing ... and mathematical foundations
- **Extended Applications**: Hi-C genomics, spatial transcriptomics, network dynamics, Reinforcement Learning, Oprimisation algos....
- **Free Energy Principle**: Connections to information theory and physics
- **Performance Optimization**: Further scalability improvements

## Contributing

They are welcome... just wait until it's all uploaded, July 2025... so after...

## Citation

If you find this work useful in your research, please cite:

```bibtex
@misc{normalizedynamics2024,
  title={NormalizedDynamics: A Kernel-Based Algorithm for Biological Trajectory Preservation},
  author={Oscar Goldman},
  year={2024},
  url={https://github.com/Gman-Superfly/NormalizedDynamics},
  note={Implementation available at https://github.com/Gman-Superfly/Normalized_Dynamic_OPT}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

We thank the computational biology community for feedback and guidance. Special recognition to:
- The developers of scanpy, UMAP, and t-SNE for providing comparison baselines
- The Gaia consortium for astronomical data access
- The single-cell genomics community for biological datasets and evaluation frameworks

---

**Note**: This is a research implementation aimed at exploring alternative approaches to manifold learning. While we've conducted comprehensive testing, users should evaluate the algorithm's suitability for their specific applications and compare against established methods like t-SNE and UMAP, ND is useful for specific datasets and problems, it's not a one size fits all solution and it has many parameters to balance before it self balances and error corrects on it's own.


## 🏁 Summary ##

NormalizedDynamics represents a **specialized contribution to manifold learning** with particular strengths in geometric preservation and computational efficiency. The algorithm demonstrates competitive performance on standard benchmarks while excelling in specific scenarios involving multi-scale, hierarchical, and structured data with distinct classes.

### ✅ **Key Contributions**
- Kernel-based iterative approach with adaptive bandwidth
- Comprehensive empirical characterization across multiple datasets
- Efficient implementation with clear scope documentation
- Honest assessment of algorithmic strengths and limitations

### 🎯 **Scientific Value**
This work contributes to the manifold learning literature by providing:
- A well-characterized specialized algorithm
- Comprehensive comparative analysis
- Open implementation for research use
- Clear guidance on appropriate applications

**This implementation is suitable for research use, practical applications within its scope, and as a reference for comparative studies in manifold learning.**

---
