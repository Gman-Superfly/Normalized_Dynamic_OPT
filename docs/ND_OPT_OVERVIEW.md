# NormalizedDynamics

A kernel-based dimensionality reduction algorithm designed for preserving continuous trajectories in biological and scientific data.

## Overview

NormalizedDynamics addresses a fundamental challenge in computational biology: existing dimensionality reduction methods like t-SNE and UMAP often fragment continuous biological processes into artificial clusters. Our algorithm maintains global connectivity through adaptive kernel computation, making it particularly suitable for:

- Single-cell RNA sequencing developmental trajectories
- Astronomical surveys with continuous stellar distributions
- Time-series biological processes
- Any scientific data requiring preservation of continuous relationships

## Key algo

The algorithm employs adaptive kernel bandwidth selection that adjusts the number of neighbors based on local data density. This adaptive mechanism helps maintain global connectivity while preserving local structure.

## Installation

```bash
# Clone the repository
git clone https://github.com/Gman-Superfly/Normalized_Dynamic_OPT.git
cd Normalized_Dynamic_OPT

# Install dependencies
pip install -r requirements.txt
```

### Core dependencies
- PyTorch (GPU support recommended)
- NumPy, SciPy, scikit-learn
- Flask (for web interface)
- Polars (efficient data processing)
- Matplotlib, Seaborn (visualization)

### Optional dependencies
- scanpy (biological analysis)
- UMAP (for comparisons)
- astroquery/astropy (GAIA data)

## Quick start

### Basic usage

```python
from src.normalized_dynamics_optimized import NormalizedDynamicsOptimized

# Initialize the algorithm
nd = NormalizedDynamicsOptimized(
    dim=2,                    # Target dimensions
    k=20,                     # Base number of neighbors
    alpha=1.0,                # Bandwidth parameter
    max_iter=50,              # Maximum iterations
    adaptive_params=True,     # Enable adaptive parameters
    device='cpu'              # or 'cuda' for GPU
)

# Fit and transform your data
X_embedded = nd.fit_transform(X)
```

### Smart K adaptation

For automatic parameter tuning based on dataset characteristics:

```python
from src.normalized_dynamics_smart_k import create_smart_k_algorithm

# Automatically configure based on dataset size
nd_smart = create_smart_k_algorithm(
    dataset_size=len(X),
    strategy='smart',  # Adaptive strategy
    device='cpu'
)

X_embedded = nd_smart.fit_transform(X)
```

### Real-time streaming

For sensor data or real-time applications:

```python
# Initialize for streaming
nd = NormalizedDynamicsOptimized(dim=2)

# Process streaming data
for new_point in data_stream:
    embedding = nd.update_embedding(new_point, max_history=500)
```

## Web interface

Launch the interactive web application:

```bash
python app.py
```

Navigate to `http://localhost:5000` to access:

- **Algorithm Comparisons**: Side-by-side evaluation against t-SNE and UMAP
- **Dataset Analyses**: Pancreas, GAIA, Wine, and more
- **Real-Time Demos**: Streaming sensor simulations
- **Smart Sampling**: Impact of different sampling strategies
- **Interactive Visualizations**: Explore embeddings in real-time

## Comprehensive test suite

### Run all tests
```bash
python src/run_tests.py
```

### Specific test categories

**Biological Validation:**
```bash
python tests/test_pancreas_endocrinogenesis.py
python tests/test_biological_metrics.py
python tests/test_enhanced_biological_metrics.py
```

**Domain-Specific Tests:**
```bash
python tests/test_gaia_data.py      # Astronomical data
python tests/test_wine_dataset.py   # Chemical classification
python tests/test_synthetic_developmental.py  # Ground truth
```

**Algorithm Validation:**
```bash
python tests/test_normalized_dynamics.py
python tests/test_comprehensive_visualizations.py
```

Results are saved in `tests/results/` and `static/results/`.

## Key results

### Biological data (Pancreas development)
- **Trajectory Smoothness**: 0.660 (ND) vs 0.696 (t-SNE)
- **Bifurcation Preservation**: Superior branching point accuracy
- **Global Connectivity**: Maintains developmental continuity

### Astronomical data (GAIA)
- **Geometric Distortion**: 0.0089 (ND) vs 0.0156 (t-SNE)
- **H-R Diagram Structure**: Better preservation of stellar relationships

### Performance
- **Speed**: Sub-second for datasets <2000 samples
- **Scalability**: Efficient up to 10,000 samples
- **GPU Support**: Full PyTorch acceleration

## Algorithm parameters

### Core parameters
- `dim`: Target embedding dimensions (default: 2)
- `k`: Base number of neighbors (automatically adapted)
- `alpha`: Bandwidth scaling (default: 1.0, adaptive)
- `max_iter`: Maximum iterations (default: 50)
- `noise_scale`: Stochastic noise level (default: 0.01)

### Adaptive features
- **Smart K**: Automatic adjustment based on dataset size and density
- **Dynamic Alpha**: Adapts to achieve target local structure preservation
- **Early Stopping**: Cost-based convergence detection

## Project structure

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

- [Technical Documentation](docs/NormalizedDynamics_OG_Technical_Documentation_deprecated_FEB_2025.py) - Complete algorithm specification
- [Methodology](docs/METHODOLOGY_TRANSPARENCY.md) - Evaluation framework and metrics
- [Project Organization](docs/repo_plans/PROJECT_ORGANIZATION_PLAN.md) - Repository structure and future directions
- [Smart Sampling](docs/smart_sampling/SMART_SAMPLING_RESULTS.md) - Sampling strategy analysis

## Citation

If you use this repository in your research, please cite it, this is ongoing work we would like to know your opions and experiments, thank you.

Authors: Oscar Goldman - Shogu research Group @ Datamutant.ai subsidiary of 温心重工業

## Future directions

- **Theoretical Analysis**: K-independence properties and mathematical foundations
- **Extended Applications**: Hi-C genomics, spatial transcriptomics, network dynamics
- **Free Energy Principle**: Connections to information theory and physics
- **Performance Optimization**: Further scalability improvements
- **Nystrom approximation**: Implement and validate a landmark-based kernel approximation against the full-kernel baseline. This is the next scaling experiment after smart sampling.

## Contributing

We welcome contributions! Areas of interest:
- Additional biological datasets
- Performance optimizations
- Theoretical analysis
- New application domains

## License

MIT License - see LICENSE file for details.

## Acknowledgments

We thank the computational biology community for feedback and the developers of scanpy, UMAP, and t-SNE for comparison baselines. Special thanks to the Gaia consortium for astronomical data access.

---

*For detailed information about the algorithm, evaluation methodology, and future research directions, please see the documentation folder.* 