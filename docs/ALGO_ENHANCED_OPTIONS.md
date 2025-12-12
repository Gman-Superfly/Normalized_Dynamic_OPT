## NormalizedDynamics and enhanced kernel options

### Context
This document describes the NormalizedDynamics update rule as it is implemented in this repository, and then lists the optional kernel families and kernel parameters that were added so you can compare behavior from the UI.

This repository has two main variants:
- `src/normalized_dynamics_optimized.py`: `NormalizedDynamicsOptimized`
- `src/normalized_dynamics_smart_k.py`: `NormalizedDynamicsSmartK` (adds K selection logic and uses the same kernel update rule)

The notation below uses:
- \(n\): number of samples (cells, points)
- \(d\): embedding dimension (typically 2)
- \(h_i \in \mathbb{R}^d\): embedding coordinate for sample \(i\)
- \(D_{ij} = \|h_i - h_j\|_2\): pairwise distance in the current embedding

### Problem the algorithm solves
We want an embedding \(H \in \mathbb{R}^{n \times d}\) that preserves geometry and continuous processes. The implementation does this by repeatedly moving each point toward a kernel-weighted neighborhood consensus, while controlling the embedding scale.

## The baseline algorithm (as implemented)

### Step 0. Initialization (in `fit_transform`)
The implementation constructs an initial embedding before iterative updates:
- If input features \(m\) satisfy \(m \le d\), it pads to \(d\) with small noise.
- Otherwise it uses a PCA-like SVD initialization and keeps the top \(d\) components.

After initialization, the algorithm iterates only on the embedding coordinates \(H\).

### Step 1. Centering and distances (per iteration)
Given the current embedding \(X \in \mathbb{R}^{n \times d}\) for this iteration:
- Compute per-dimension mean \(\mu\) and standard deviation \(s\) of \(X\).
- Center: \(\tilde{X} = X - \mu\).
- Compute pairwise distances: \(D_{ij} = \|\tilde{x}_i - \tilde{x}_j\|_2\).

### Step 2. Adaptive bandwidth \(\sigma_i\) from a k-th neighbor distance
The implementation assigns each point \(i\) a local bandwidth based on the distance to its k-th nearest neighbor in the current embedding.

Let \(k\) be the neighbor count used for bandwidth selection.
- Compute the sorted neighbor distances for each row \(i\), and take the k-th smallest distance (excluding self implicitly via sorting):

\[
\sigma_i = D_{i,(k)}
\]

In `NormalizedDynamicsOptimized`, k is chosen like this:
- Start from \(k_{base} = \min(k_{configured}, n-1)\).
- If adaptive parameters are enabled and \(n > 20\), it computes a simple per-point heterogeneity proxy from distance variability and then uses the mean of a clamped per-point k suggestion.

In `NormalizedDynamicsSmartK`, k comes from `adaptive_k_selection(...)` and can follow strategies like `fixed`, `size`, `density`, or `smart`.

### Step 3. Kernel weights and row-normalization
The algorithm computes an unnormalized weight \(w_{ij}\) from \(D_{ij}\) and \(\sigma_i\), then row-normalizes:

\[
 p(j\mid i) = \frac{w_{ij}}{\sum_{\ell=1}^{n} w_{i\ell} + \varepsilon}
\]

This makes each row a probability distribution over neighbors.

The specific formula for \(w_{ij}\) depends on `kernel_type` and optional parameters, described below.

### Step 4. Drift (kernel-weighted neighborhood consensus)
The drift for point \(i\) is the kernel-weighted average of all points:

\[
\delta_i = \sum_{j=1}^{n} p(j\mid i)\, \tilde{x}_j
\]

In code this is a matrix multiply: `drift = kernel @ x_centered`.

### Step 5. Update rule (consensus drift plus noise)
The update moves each point toward its drift:

\[
\tilde{h}_i^{(t+1)} = \tilde{h}_i^{(t)} + \Delta t\, (\delta_i - \tilde{h}_i^{(t)}) + \xi_i
\]

Where:
- \(\Delta t = d^{-\alpha}\) (dimension-dependent step size)
- \(\alpha\) is either fixed or adaptively adjusted
- \(\xi_i\) is isotropic Gaussian noise with scale `noise_scale`

### Step 6. Per-dimension scale preservation
After the drift update, the implementation rescales each embedding dimension to preserve the previous iteration’s per-dimension standard deviation.

Let \(s\) be the standard deviation vector computed at the start of the iteration from \(X\), and let \(\hat{s}\) be the standard deviation of the updated \(\tilde{H}\). The rescale step is:

\[
\tilde{H} \leftarrow \tilde{H} \odot \frac{s}{\hat{s} + \varepsilon}
\]

Then it adds back the mean \(\mu\):

\[
H \leftarrow \tilde{H} + \mu
\]

This keeps the embedding from collapsing or blowing up in a single dimension.

### Optional: cost-based monitoring and early stopping
Both algorithm variants periodically compute a cost based on:
- Distortion between pairwise distances in original space vs embedding space
- Local structure overlap

The cost is:

\[
\text{cost} = 0.3\,\text{distortion} + 0.7\,(1 - \text{local\_structure})
\]

This cost is used for early stopping and for parameter adaptation.

## Enhanced kernel families (new options)

### Shared pieces of notation
All kernels use:
- \(D_{ij}\): current embedding distance
- \(\sigma_i\): per-point bandwidth
- \(\varepsilon\): small constant for stability
- \(\beta\): kernel slope (new), multiplies the distance term

In all cases, define \(w_{ij}\) and then row-normalize to get \(p(j\mid i)\).

### 1) Exponential kernel (existing default)
`kernel_type='exponential'`

\[
 w_{ij} = \exp\left( -\frac{\beta\, D_{ij}}{2\sigma_i^2 + \varepsilon} \right)
\]

Notes:
- This is the repository default because it has been empirically useful.

### 2) Gaussian / RBF kernel
`kernel_type='gaussian'`

\[
 w_{ij} = \exp\left( -\frac{\beta\, D_{ij}^2}{2\sigma_i^2 + \varepsilon} \right)
\]

Notes:
- This is the standard RBF form in many kernel methods.

### 3) Generalized exponential family (shape parameter p)
`kernel_type='generalized'`, with `kernel_p = p`

\[
 w_{ij} = \exp\left( -\frac{\beta\, D_{ij}^{p}}{2\sigma_i^2 + \varepsilon} \right)
\]

Interpretation:
- \(p=1\) behaves like the exponential option.
- \(p=2\) behaves like the Gaussian option.
- \(p\) lets you interpolate between heavier or sharper decay behaviors.

### 4) Student-t kernel (degrees of freedom ν)
`kernel_type='student_t'`, with `kernel_nu = \nu`

\[
 w_{ij} = \left(1 + \frac{\beta\, D_{ij}^2}{\nu\,\sigma_i^2 + \varepsilon}\right)^{-\frac{\nu + 1}{2}}
\]

Interpretation:
- Heavier tails than Gaussian.
- Smaller \(\nu\) produces heavier tails.

### 5) Rational quadratic kernel (shape α)
`kernel_type='rational_quadratic'`, with `kernel_alpha = \alpha`

\[
 w_{ij} = \left(1 + \frac{\beta\, D_{ij}^2}{2\alpha\,\sigma_i^2 + \varepsilon}\right)^{-\alpha}
\]

Interpretation:
- It can be seen as a mixture of Gaussians with different scales.
- Smaller \(\alpha\) produces heavier tails.

## Enhanced kernel slope: learned β (what you asked for)

### What β does
\(\beta\) controls the decay rate of the kernel with distance.
- Larger \(\beta\): faster decay, more local neighborhoods
- Smaller \(\beta\): slower decay, more global influence

### Implementation
Parameters:
- `kernel_beta`: initial value of \(\beta\) (default 1.0)
- `learn_kernel_beta`: enable auto-tuning (default False)
- `kernel_beta_eta`: update learning rate (default 0.01)

When `learn_kernel_beta=True`, the code updates \(\beta\) during optimization using the same periodic evaluation loop used for adaptive \(\alpha\):

\[
\text{error} = \text{target\_local\_structure} - \text{local\_structure}
\]
\[
\beta \leftarrow \mathrm{clip}(\beta + \eta_\beta\,\text{error},\ 0.1,\ 10.0)
\]

Important note:
- This is not gradient descent on \(\beta\) through autograd. It is a simple feedback controller driven by the measured local structure metric.

## How this is exposed in the web UI
Each demo page now supports:
- `kernel_type`: exponential, gaussian, generalized, student_t, rational_quadratic
- One kernel shape parameter field (context-dependent):
  - p for generalized
  - ν for Student-t
  - α for rational quadratic
- `kernel_beta` (Slope β)
- `learn_kernel_beta` (Auto-tune β)

Defaults are preserved:
- `kernel_type='exponential'`
- `kernel_beta=1.0`
- `learn_kernel_beta=False`

## Verification
To verify which kernel is active:
- Use the UI selectors and run the same dataset twice with different kernels.
- Compare runtime and metrics in the page output.
- For kernels with a shape parameter, keep everything constant except the single parameter (p, ν, or α).
- For learned β, compare “Auto-tune β” on vs off while holding the initial β fixed.
