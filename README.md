# NormalizedDynamics: A Kernel-Based Manifold Learning Algorithm

A specialized dimensionality reduction algorithm designed for preserving continuous relationships in scientific data, with particular focus on trajectory analysis and self stabilization.

## more readme files will be added the coming week for all the tests ##

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

**Gaussian Kernel and Euclidean Distance**:
The Gaussian kernel is defined as:
```
K(h_i, h_j) = exp(-||h_i - h_j||² / (2σ_i²))
```
where ||h_i - h_j||² is the squared Euclidean distance between points h_i and h_j. Euclidean distance measures the straight-line separation in feature space, computed as:
```
||h_i - h_j|| = sqrt(Σ_k (h_ik - h_jk)²)
```
In sparse regions (fewer points), larger kernels smooth over broader areas; in dense regions (many points), smaller kernels preserve fine details. Close points yield kernel values near 1 (high similarity), while distant points approach 0 (low similarity).

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
   - Computes pairwise Euclidean distances using torch.cdist(x_centered, x_centered), where Euclidean distance is the straight-line measure: ||x_i - x_j|| = sqrt(Σ (x_ik - x_jk)^2), with squared form used in the Gaussian kernel.
   - Comprehensive information integration across entire dataset, enabling global structure preservation while adapting to local density.
   - Complete interaction matrix with adaptive bandwidth weighting via Gaussian kernel: K(h_i, h_j) = exp(-||h_i - h_j||² / (2σ_i²)), normalized to probabilities p(j|i).

4. **Multi-Criteria Convergence**: Cost-based and stability-based stopping criteria
   - Cost-based: Every 5 iterations (after iteration 10), computes cost = 0.3 * distortion + 0.7 * (1 - local_structure), where distortion is mean absolute difference in normalized distance matrices, and local_structure is average k-NN overlap (k=10). Increments patience counter on no improvement (<1e-5) or increase; stops at patience=5.
   - Stability-based: Every iteration, checks if norm(embedding - embedding_old) < 1e-6; stops on convergence.
   - Supports feedback loops for iterative refinement, including embedding updates, adaptive bandwidth (σ_i based on k-th neighbor distance), and alpha adjustment.

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

This emergence provides theoretical foundation for understanding why the algorithm works well for physically accurate trajectory preservation.

**Insights on the Temperature Parameter (T) Analog**:

This section recaps the key insights from discussions on how the Free Energy Principle (FEP) temperature parameter \( T \) is implicitly incorporated into our dimensionality reduction algorithm. 

The algorithm embeds high-dimensional data \( X \in \mathbb{R}^{n \times d} \) into a lower-dimensional space \( H \in \mathbb{R}^{n \times k} \), preserving geometric relationships and local neighborhoods via an adaptive Gaussian kernel, drift-based updates, and scale preservation. It naturally aligns with the FEP by minimizing a free energy functional \( \mathcal{F}[H] = U[H] - T \cdot S[H] \), where:
- \( U[H] = \frac{1}{2} \sum_i \|h_i - \delta_i\|^2 \) is the energy term (prediction error between embedding points \( h_i \) and predicted positions \( \delta_i \)).
- \( S[H] = -\sum_{i,j} p(j|i) \log p(j|i) \) is the entropy term (uncertainty in neighborhood probabilities \( p(j|i) \), derived from the normalized Gaussian kernel).

Since \( T \) (which balances precision/energy minimization and robustness/entropy maximization) is not explicitly defined, it emerges implicitly through other parameters. Below are the collected insights on this analog.

**Role of \( T \) in the FEP**:
- \( T \) scales the entropy term's contribution to the free energy gradient: \( \frac{\partial \mathcal{F}}{\partial h_i} = \frac{\partial U}{\partial h_i} - T \cdot \frac{\partial S}{\partial h_i} \).
- High \( T \): Prioritizes entropy, leading to broader, more uniform neighborhood probabilities (robust but less precise embeddings).
- Low \( T \): Prioritizes energy, focusing on tight, precise predictions (detailed local structure but potential overfitting).

**Implicit Absorption of \( T \) into Algorithm Parameters**:
The algorithm absorbs \( T \) through a combination of the step size \( \alpha \) and adaptive bandwidth \( \sigma_i \), with scale preservation acting as a stabilizer. This creates a dynamic balance without explicit tuning.

1. **Absorption into \( \alpha \) (Step Size)**:
   - \( \alpha \) (and the effective step size \( \Delta t = d^{-\alpha} \)) controls the update rule: \( h^{(t+1)} = h^{(t)} + \alpha \cdot \Delta t \cdot (G[h^{(t)}] - h^{(t)}) + \eta \).
   - **Mechanism**: \( \alpha \) scales the drift toward the predicted position \( \delta_i = G[h_i] \), directly minimizing the energy term \( U \).
     - Smaller \( \alpha \): Slower updates allow more iterations for entropy (via kernel adaptation) to influence the embedding, mimicking higher \( T \) (more exploration, higher entropy).
     - Larger \( \alpha \): Faster energy minimization, mimicking lower \( T \) (focus on precision, lower entropy).
   - **Adaptive Nature**: If enabled, \( \alpha \) is dynamically adjusted: \( \alpha += \eta \cdot (\text{target_local_structure} - \text{metrics['local_structure']}) \), clamped between 0.01 and 2.0. This feedback loop refines the energy-entropy trade-off based on embedding quality.
   - **Insight**: \( \alpha \) acts as a proxy for \( 1/T \), modulating how aggressively the algorithm reduces prediction error versus allowing entropy-driven robustness.

2. **Absorption into \( \sigma_i \) (Adaptive Bandwidth)**:
   - \( \sigma_i = \|h_i - h_i^{(k)}\|_2 \) (distance to the K-th nearest neighbor) controls the Gaussian kernel spread: \( K(h_i, h_j) = \exp\left(-\frac{\|h_i - h_j\|^2}{2\sigma_i^2}\right) \), normalized to \( p(j|i) \).
   - **Mechanism**: \( \sigma_i \) directly affects entropy \( S \):
     - Smaller \( \sigma_i \) (dense regions): Narrow kernel, concentrates \( p(j|i) \) on few points, lowers entropy (mimics low \( T \), high precision).
     - Larger \( \sigma_i \) (sparse regions): Wide kernel, spreads \( p(j|i) \) across more points, raises entropy (mimics high \( T \), high robustness).
   - **Density Adaptation**: \( K \) is adjusted via density factor \( = \frac{\text{distance_std}}{\text{distance_mean}} \):
     - High factor (sparse, high variance): Larger \( K \), larger \( \sigma_i \), higher effective \( T \).
     - Low factor (dense, low variance): Smaller \( K \), smaller \( \sigma_i \), lower effective \( T \).
     - Formula: \( k_{\text{adaptive}} = \text{clamp}(5 + 10 \times \text{density_factor}, \text{min_k}, \text{max_k}) \).
   - **Insight**: \( \sigma_i \) serves as the primary analog for \( T \), dynamically tuning the kernel's "blur radius" to balance local detail and global connectivity based on data density.

3. **Combined Effect and Scale Preservation**:
   - **Interplay**: \( \alpha \) (energy focus) and \( \sigma_i \) (entropy control) together emulate \( T \)'s trade-off:
     - Small \( \alpha \) + large \( \sigma_i \): High effective \( T \) (exploration in sparse areas).
     - Large \( \alpha \) + small \( \sigma_i \): Low effective \( T \) (precision in dense areas).
   - **Scale Preservation**: \( H \leftarrow H \cdot \frac{\sigma_{\text{original}}}{\sigma_{\text{current}}} \) stabilizes the balance, preventing entropy changes (via \( \sigma_i \)) from distorting global geometry, maintaining a consistent effective \( T \).
   - **Insight**: This implicit absorption makes the algorithm robust (e.g., K-independence observed in tests) without explicit \( T \) tuning, but experiments varying \( \alpha \) and \( \sigma_i \) can confirm its impact.

**Practical Implications**:
- **Robustness**: The algorithm's K-independence stems from \( \sigma_i \)'s density adaptation, ensuring stable entropy levels.
- **Testing \( T \)'s Analog**: Vary \( \alpha \) (e.g., small for higher entropy influence) or \( K \) bounds to observe effects on embeddings. Compute entropy \( S \) to verify correlations with \( \sigma_i \).
- **Limitations**: Without explicit \( T \), fine control requires indirect tuning of \( \alpha \) and \( \sigma_i \). Future enhancements could expose \( T \) for direct modulation.

This recap highlights the algorithm's elegant integration of FEP concepts, making it a theoretically grounded tool for dimensionality reduction. For implementation details, see the code in `NormalizedDynamicsOptimized`.

### Early Stopping Mechanism

Early stopping in the algorithm prevents unnecessary iterations by halting when improvements plateau or the embedding stabilizes. It uses two complementary checks:

1. **Cost-Based Stopping**:
   - Evaluated every 5 iterations after iteration 10.
   - Computes a cost: \( \text{cost} = 0.3 \times \text{distortion} + 0.7 \times (1 - \text{local_structure}) \), where distortion measures global distance preservation and local_structure assesses k-NN overlap (k=10).
   - If cost increases or improvement is < 1e-5, a patience counter increments; stops if counter reaches 5.

2. **Embedding Change-Based Stopping**:
   - Checked every iteration.
   - Stops if \( \text{norm}(H^{(t+1)} - H^{(t)}) < 1e-6 \), indicating stability.

These mechanisms ensure efficient convergence while maintaining quality.

### Parameter Sensitivity and K-Independence for certain settings and datasets

The algorithm exhibits robustness to K (number of neighbors for bandwidth), often showing K-independence in tests. Reasons include:
- **Uniform Data Density**: Similar σ_i across K values in evenly distributed data.
- **Scale Preservation**: Normalizes variations in σ_i, standardizing the embedding.
- **Robust Drift Function**: Weighted center of mass is insensitive to moderate σ_i changes.
- **Noise and Convergence**: Stochastic noise (η) smooths differences.
- **Embedding Space**: Low intrinsic dimensionality reduces σ_i variability.

This reduces hyperparameter tuning needs, making the algorithm practical.

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
- **and more in web UI...**
    

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

##THIS DATA NEEDS TO BE UPDATED##
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

- **Theoretical Analysis**: K-independence properties need investigating as they are interesting and self stabilizing ... and mathematical foundations
- **Extended Applications**: Hi-C genomics, spatial transcriptomics, network dynamics, Reinforcement Learning, Optimization algos....
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


### ** Value**
This work contributes to the manifold learning literature by providing:
- A well-characterized specialized algorithm
- Comprehensive comparative analysis
- Open implementation for research use
- Clear guidance on appropriate applications
