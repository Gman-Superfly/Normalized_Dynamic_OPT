# NormalizedDynamics: A Self-Adapting Kernel-Based Manifold Learning Algorithm

NormalizedDynamics combines kernel-based dynamics with adaptive bandwidths and early-stopping controls to preserve continuous relationships. It targets small and medium datasets (larger sets possible with scaling) where trajectory continuity matters, the demos include biological and astronomical data, one of the main features is support for real-time streaming of embedding updates (try the multi sensor real-time demo).

## Context
- This repository is under active refinement. Documentation files under `docs/` are present in the current tree, and this README is the primary entry point.
- Reported performance figures are historical. Reproduce them on your data before relying on them.

## Problem
Preserving trajectories and local structure is difficult when data density varies and samples arrive over time. Traditional manifold learners struggle to keep global connectivity while avoiding geometric distortion in these settings.

## Solution
- Adaptive kernels with bandwidth from local neighbor distances while keeping full pairwise connectivity.
- Step-size control and multi-criteria early stopping to stabilize optimization.
- Optional smart sampling plus dynamic K selection for density-varying datasets.
- Streaming entry points that maintain a bounded history for incremental updates.

## Capabilities (operational claims)
- Smart sampling integration: historical internal runs reported about 83 percent size reduction while preserving observed cell type diversity, with 5–15 percent trajectory smoothness gains. Reproduce on your data; artifacts are not bundled.
- Dynamic K adaptation: size-aware and density-aware neighbor counts (examples from prior runs: 2000 cells → K≈28, 3000 cells → K≈35) to balance stability and locality.
- Real-time streaming: incremental embedding updates with optional history cap for sensor-style data.
- Tunable structure-geometry balance: cost weighting between local structure and distortion (default 70-30) plus convergence and stability checks.
- Free Energy Principle framing: energy-entropy balance guides adaptation; see algorithm overview for the explicit forms.
- Scale preservation: per-feature standard deviation is maintained after each update.

## Applications
- Single-cell developmental biology, RNA-seq trajectory analysis, stem cell studies.
- Astronomical surveys (e.g., GAIA), time-series with continuous trends.
- Sensor monitoring, interactive visualization, reinforcement learning diagnostics.
- Best suited for ≤2000 samples in real-time settings and up to roughly 5000 samples for higher-precision offline analysis.

**Note: the mouse cortical tests will be moved to another repo because they use a modal backend, which can be noisy if you only want to run the remaining tests.**

## Installation

```bash
# Clone the repository
git clone https://github.com/Gman-Superfly/Normalized_Dynamic_OPT.git
cd Normalized_Dynamic_OPT

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
- Core: PyTorch (GPU recommended), NumPy, SciPy, scikit-learn, Polars.
- Web/demo: Flask, Matplotlib, Seaborn.
- Extended analysis: scanpy, UMAP, astroquery/astropy when those comparisons or datasets are used.

**Core Requirements:**
- PyTorch (GPU support recommended)
- NumPy, SciPy, scikit-learn
- Polars (efficient data processing)

**Web Interface:**
- Flask, Matplotlib, Seaborn

**Extended Analysis:**
- scanpy (biological analysis), UMAP (comparisons), astroquery/astropy (GAIA data)

## Quick start

### Basic Usage

```python
from src.normalized_dynamics_optimized import NormalizedDynamicsOptimized

# Initialize with adaptive features enabled:
nd = NormalizedDynamicsOptimized(
    dim=2,                    # Target dimensions
    k=20,                     # Base neighbors (auto-adapted)
    alpha=1.0,                # Bandwidth parameter (adaptive)
    max_iter=50,              # Maximum iterations
    adaptive_params=True,     # Enable self-correction
    device='cpu',             # or 'cuda' for GPU
    kernel_type='exponential' # or 'gaussian' for standard RBF
)

# Fit and transform your data:
X_embedded = nd.fit_transform(X)
```

### Kernel type options

The algorithm supports two kernel functions for distance weighting:

- **`'exponential'`** (default): `K = exp(-d / (2σ²))` - Linear distance decay. Works well empirically on biological trajectory data.
- **`'gaussian'`**: `K = exp(-d² / (2σ²))` - Squared distance decay. Standard RBF kernel formulation.

```python
# Use Gaussian kernel (standard RBF)
nd_gaussian = NormalizedDynamicsOptimized(dim=2, kernel_type='gaussian')

# Use Exponential kernel (default, empirically effective)
nd_exponential = NormalizedDynamicsOptimized(dim=2, kernel_type='exponential')
```

# Smart sampling:
```python
from src.smart_sampling import BiologicalSampler
from src.normalized_dynamics_smart_k import create_smart_k_algorithm

sampler = BiologicalSampler(target_size=15000)
sampled_indices = sampler.hybrid_sample(X, spatial_coords)

# Dynamic K:
nd_smart = create_smart_k_algorithm(
    dataset_size=len(sampled_indices),
    strategy="smart",
    device="cpu"
)

# Process the optimized dataset :
X_embedded = nd_smart.fit_transform(X[sampled_indices])
```

### Smart K adaptation only

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

### Real-time streaming with memory management

```python
# Initialize for streaming applications with configurable parameters
nd = NormalizedDynamicsOptimized(
    dim=2,
    adaptive_params=True,     # Enable real-time adaptation
    alpha=1.0                 # Adjust for streaming responsiveness
)

# Process live data streams with memory management
for new_point in data_stream:
    # Incremental embedding updates with history buffer
    embedding = nd.update_embedding(
        new_point, 
        max_history=500,      # Configurable memory limit
        update_adaptive=True  # Real-time parameter adjustment
    )
    
    # Real-time visualization or processing
    visualize_embedding(embedding)
```

## Algorithm Overview

### Mathematical Foundation

The algorithm implements iterative dynamics with kernel-weighted drift and adaptive mechanisms:

```
h^(t+1) = h^(t) + α × Δt × (G[h^(t)] - h^(t)) + η

Where:
- `G[h]`: Kernel-weighted drift function (neighborhood consensus)
- `α`: Adaptive step size parameter with feedback control
- `Δt`: Dimension-dependent time step (d^(-α))
- `η`: Optional stochastic exploration noise
```

Core technical features:
- Adaptive kernel bandwidth: k-th neighbor distances set σ_i; Gaussian kernel on full pairwise distances.
- Early stopping and cost: cost = 0.3 × distortion + 0.7 × (1 - local_structure), evaluated periodically with patience.
- Free Energy Principle framing: `F[H] = U[H] - T·S[H]`, with energy from prediction error and entropy from neighborhood uncertainty.
- Scale preservation: rescale to original per-feature standard deviations each iteration.
- Smart sampling integration: spatial, expression, and hybrid strategies available; dynamic K scales with dataset size.
- Streaming architecture: incremental updates with a configurable history buffer.

In Depth:
**1. Adaptive Kernel Bandwidth**
- Uses k-th nearest neighbor distances: `σ_i = ||h_i - h_i^(k)||₂`
- Gaussian kernel: `K(h_i, h_j) = exp(-||h_i - h_j||²/(2σ_i²))`
- Local density adaptation for optimal information integration

**2. Multi-Criteria Early Stopping with Tunable Balance**
- **Cost-based**: Evaluates `cost = 0.3 × distortion + 0.7 × (1 - local_structure)` every 5 iterations
- **Configurable Weighting**: 70-30 balance between local structure preservation and geometric distortion (tunable for different datasets)
- **Stability-based**: Monitors embedding change norm for convergence detection
- **Patience mechanism**: Prevents premature termination while avoiding overfitting

**3. Free Energy Principle Implementation**
- Minimizes free energy functional: `F[H] = U[H] - T·S[H]`
- Energy term: `U[H] = ½∑ᵢ ||hᵢ - δᵢ||²` (prediction error)
- Entropy term: `S[H] = -∑ᵢⱼ p(j|i) log p(j|i)` (neighborhood uncertainty)
- Emergent temperature parameter through α and σ_i interaction

**4. Scale Preservation**
- Maintains feature-wise standard deviations: `h ← h × (σ_original / σ_current)`
- Prevents geometric distortion during iterative updates

**5. Smart Sampling Integration**
- **Biological Structure Preservation**: 83% size reduction maintaining 100% cell type diversity
- **Dynamic K Scaling**: Automatic parameter optimization (K ∝ √dataset_size)
- **Performance**: 5-15% trajectory smoothness improvements observed in testing
- **Sampling Strategies**: Spatial stratified, expression diversity, and hybrid approaches

**6. Real-time streaming architecture**
- **Incremental Embedding Updates**: Live data processing with `update_embedding()` method
- **Memory Management**: Configurable history buffer (default: 500 samples max_history)
- **Interactive Demonstrations**: Web-based streaming sensor simulations
- **Deployment scope**: Designed for sensor monitoring and live data visualization

### Computational characteristics

- **Time Complexity**: O(n²d) per iteration with global connectivity
- **Space Complexity**: O(n²) for distance and kernel matrices
- **Performance**: Sub-second processing for <=2000 samples, with stable behavior on medium datasets
- **Scalability**: Prioritizes accuracy through comprehensive pairwise analysis

## Comprehensive Evaluation

### Test Coverage

Our extensive test suite validates performance across multiple domains:

```
tests/
├── test_normalized_dynamics.py          # Core algorithm validation
├── test_convergence.py                  # Multi-criteria convergence validation
├── test_biological_metrics.py          # Standard biological evaluation
├── test_enhanced_biological_metrics.py # Advanced DPT-based metrics
├── test_pancreas_endocrinogenesis.py   # Single-cell developmental data
├── test_gaia_data.py                   # Astronomical survey data
├── test_synthetic_developmental.py     # Ground truth validation
└── smart_sampling_enhanced_analysis.py # Sampling strategy analysis
```

Benchmark results (historical; reproduce before use):
| Dataset | Geometric Distortion | Local Structure | Trajectory Smoothness |
|---------|---------------------|-----------------|----------------------|
| Pancreas Development | 0.0089 | 0.710 | 0.660 |
| GAIA Stellar Data | 0.0156 | 0.680 | N/A |
| Wine Classification | 0.0034 | 0.850 | N/A |
| Multi-Scale Circles | 0.0012 | 0.920 | N/A |

Running evaluations:
```bash
python src/run_tests.py

# Run specific evaluations
python tests/test_pancreas_endocrinogenesis.py     # Biological validation
python tests/test_gaia_data.py                     # Astronomical data
```
Outputs, when produced, are written to `static/results/` with timestamped names.

## Performance characteristics
- Global connectivity with adaptive bandwidths for density variation.
- Trajectory continuity and streaming-friendly updates with bounded history.
- Tunable balance between local structure preservation and geometric distortion.
- Free Energy Principle framing for energy/entropy trade-offs.

Limitations:
- O(n²) time and space; large datasets require sampling or batching.
- Local structure preservation can vary (historical range 46–85 percent) relative to methods that optimize locality only.
- 3D manifold unfolding and highly curved geometries may need specialized settings.

### Strengths
- **Global Structure Preservation**: Maintains physically meaningful spatial relationships
- **Trajectory Continuity**: Avoids artificial fragmentation in developmental processes  
- **Adaptive Behavior**: Self-adjusts to local data density and manifold characteristics
- **Real-Time Capability**: Interactive applications and live sensor monitoring


### Optimal use cases
- **Developmental Biology**: RNA-seq trajectory analysis, stem cell differentiation
- **Astronomical Data**: Stellar surveys with continuous distributions
- **Real-Time Applications**: Sensor monitoring, interactive visualization (≤2000 samples)
- **Scientific Datasets**: Applications requiring continuous relationship preservation

## Smart sampling and dynamic K results (historical)
- Size reduction: about 83 percent while keeping observed cell type diversity in prior internal tests.
- Trajectory smoothness: historical gains of 5–15 percent.
- Example K scaling: 2000 cells → K≈28, 3000 cells → K≈35.

## Nystrom approximation status
Nystrom approximation is the next scaling step to test. Smart sampling reduces the number of input points before optimization, while Nystrom would approximate the full kernel with a smaller set of landmark points. The expected target is to reduce the full `O(n^2)` kernel cost toward an `O(nm)` cross-kernel cost where `m` is the landmark count and `m << n`.

The current UI exposes Nystrom as an explicit on/off experimental setting so test runs can record the intended approximation mode. The numerical Nystrom path still needs implementation and validation against the full-kernel baseline before runtime or quality claims should be made.

Performance table (historical):
| Strategy | Trajectory Smoothness | Improvement | Runtime |
|----------|----------------------|-------------|---------|
| Random Sampling | 0.447 | baseline | 106.1s |
| Smart + Dynamic K | 0.480 | +7.4% | 130.4s |
| Hybrid + Dynamic K | 0.429 | +8.7% | 128.4s |

**Note on Performance Variability**: Results depend on dataset characteristics and 
geometric complexity. As a geometry-preserving algorithm with self-adapting, 
error-correcting mechanisms, NormalizedDynamics optimizes for geometric fidelity rather 
than uniformly maximizing trajectory smoothness. Some combinations may prioritize global 
structure preservation over local smoothness metrics, reflecting the algorithm's adaptive 
response to intrinsic data properties and manifold characteristics.

Sampling strategies:
1) Spatial stratified sampling: Preserves tissue architecture through grid-based 
spatial sampling
2) Expression diversity: Maintains cell type diversity using clustering-based 
selection
3) Hybrid: Combines spatial and expression strategies for optimal biological 
preservation

### Usage

```python
from src.smart_sampling import BiologicalSampler
from src.normalized_dynamics_smart_k import create_smart_k_algorithm

# Smart sampling with dynamic K adaptation
sampler = BiologicalSampler(target_size=15000)
hybrid_indices = sampler.hybrid_sample(data, spatial_coords)

# Apply NormalizedDynamics with dynamic K
nd_smart = create_smart_k_algorithm(dataset_size=len(hybrid_indices), strategy='smart')
embedding = nd_smart.fit_transform(data[hybrid_indices])
```

## Web interface

Launch the interactive demonstration:

```bash
python app.py
```

Navigate to `http://localhost:5000` for:
- **Algorithm Comparisons**: Side-by-side evaluation against t-SNE and UMAP
- **Dataset Analyses**: Pancreas, GAIA, Wine, and synthetic datasets
- **Real-Time Demos**: Streaming sensor simulations
- **Smart Sampling**: Impact analysis of different sampling strategies
- **Interactive Visualizations**: Explore embeddings with real-time parameter adjustment

## Algorithm parameters

### Core parameters
- `dim`: Target embedding dimensions (default: 2)
- `k`: Base number of neighbors (automatically adapted based on data density)
- `alpha`: Bandwidth scaling factor (default: 1.0, with adaptive adjustment)
- `max_iter`: Maximum iterations (default: 50)
- `noise_scale`: Stochastic exploration level (default: 0.01)
- `kernel_type`: Kernel function for distance weighting (default: 'exponential')
  - `'exponential'`: K = exp(-d / (2σ²)) - Linear distance decay, empirically effective
  - `'gaussian'`: K = exp(-d² / (2σ²)) - Squared distance decay, standard RBF kernel

### Adaptive features
- **Smart K**: Automatic adjustment using density factors and dataset characteristics
- **Dynamic Alpha**: Performance feedback-based adaptation for optimal convergence
- **Early Stopping**: Multi-criteria convergence detection with patience mechanisms
- **Kernel Selection**: Choose between exponential (default) and Gaussian kernels via UI or API

## Project structure

```
Normalized_Dynamic_OPT/
├── src/                    # Core algorithm implementations
├── tests/                  # Comprehensive evaluation suite
├── templates/              # Web interface templates
├── static/                 # Results, CSS, and JavaScript
├── data/                   # Biological and astronomical datasets
├── docs/                   # Technical documentation
├── app.py                  # Flask web application
└── requirements.txt        # Python dependencies
```

## Documentation

- **Technical Specification**: [`docs/NormalizedDynamics_OG_Technical_Documentation_deprecated_FEB_2025.py`](docs/NormalizedDynamics_OG_Technical_Documentation_deprecated_FEB_2025.py)
- **Evaluation Framework**: [`docs/METHODOLOGY_TRANSPARENCY.md`](docs/METHODOLOGY_TRANSPARENCY.md)
- **Test Infrastructure**: [`docs/tests/README_tests.md`](docs/tests/README_tests.md)
- **Project Organization**: [`docs/repo_plans/PROJECT_ORGANIZATION_PLAN.md`](docs/repo_plans/PROJECT_ORGANIZATION_PLAN.md)

## Future directions

- **Theoretical Analysis**: Investigation of K-independence properties and mathematical foundations
- **Extended Applications**: Hi-C genomics, spatial transcriptomics, network dynamics, reinforcement learning
- **Free Energy Principle**: Deeper connections to information theory and physics
- **Performance Optimization**: Scalability improvements for larger datasets

## Contributing

Contributions are welcome! Please submit pull requests or issues directly. The repository is actively maintained and open for community contributions.

## Citation

- Authors: Oscar Goldman - Shogu research Group @ Datamutant.ai (subsidiary of 温心重工業)  
- Message: "If you use this repository in your research, please cite it, this is ongoing work we would like to know your opions and experiments, thank you."

## License
Code MIT License (docs cc4).

## Acknowledgments

We thank the computational biology community for feedback and guidance. Special recognition to:
- The developers of scanpy, UMAP, and t-SNE for providing comparison baselines
- The Gaia consortium for astronomical data access
- The single-cell genomics community for biological datasets and evaluation frameworks

---

**Note**: This is a research implementation exploring manifold learning with a Free Energy Principle framing. Evaluate suitability on your data and compare against established methods before relying on results.

## Summary

We optimize this algorithm for:
1.  **Trajectory Analysis**: Continuous biological processes (e.g., cell differentiation).
2.  **Scientific Data**: Systems where theoretical grounding matters (e.g., astronomy).
3.  **Real-time Adaptation**: Scenarios requiring incremental updates.

We provide this implementation and the accompanying evaluation framework to enable reproducible comparisons with established methods.

