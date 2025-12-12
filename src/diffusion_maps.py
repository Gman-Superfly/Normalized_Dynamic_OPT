"""Diffusion maps embedding.

This module provides a small Diffusion Maps implementation with a sklearn-like
fit_transform interface so it can be used alongside t-SNE/UMAP in the demos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors


@dataclass
class DiffusionMaps:
    """Compute a Diffusion Maps embedding.

    Args:
        n_components: Number of diffusion coordinates to return (excluding trivial component).
        n_neighbors: Number of nearest neighbors used to build the sparse affinity graph.
        alpha: Density normalization exponent (0.0 to 1.0). 0.5 is a common default.
        t: Diffusion time. t=1 is the standard embedding.
        epsilon: Kernel scale for the affinity kernel. If None, it is estimated from kNN distances.
        random_state: Seed used by neighbor search where applicable.
    """

    n_components: int = 2
    n_neighbors: int = 30
    alpha: float = 0.5
    t: int = 1
    epsilon: Optional[float] = None
    random_state: int = 42

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit diffusion maps and return embedding."""
        assert X is not None, "X is required"
        assert isinstance(X, np.ndarray), f"Expected np.ndarray, got {type(X)}"
        assert X.ndim == 2, f"Expected 2D X, got shape {X.shape}"
        n_samples = int(X.shape[0])
        assert n_samples >= 3, f"Need at least 3 samples, got {n_samples}"
        assert isinstance(self.n_components, int) and self.n_components >= 1, "n_components must be >= 1"

        k = int(self.n_neighbors)
        if k < 2:
            k = 2
        if k >= n_samples:
            k = n_samples - 1

        # Build kNN graph.
        nn = NearestNeighbors(n_neighbors=k + 1)
        nn.fit(X)
        dists, inds = nn.kneighbors(X)

        # Exclude self-neighbor at column 0.
        dists = dists[:, 1:]
        inds = inds[:, 1:]

        # Choose epsilon from kNN distances if not provided.
        if self.epsilon is None:
            # Use median of squared k-th neighbor distance.
            kth = dists[:, -1]
            eps = float(np.median(kth * kth))
            if not np.isfinite(eps) or eps <= 0:
                eps = 1.0
        else:
            eps = float(self.epsilon)
            if not np.isfinite(eps) or eps <= 0:
                eps = 1.0

        # Affinity weights.
        w = np.exp(-(dists * dists) / (eps + 1e-12))

        # Sparse affinity matrix W (symmetrized).
        rows = np.repeat(np.arange(n_samples, dtype=int), k)
        cols = inds.reshape(-1).astype(int)
        data = w.reshape(-1).astype(float)
        W = sparse.csr_matrix((data, (rows, cols)), shape=(n_samples, n_samples))
        W = 0.5 * (W + W.T)

        # Density normalization: K = D_q^{-alpha} W D_q^{-alpha}
        q = np.asarray(W.sum(axis=1)).reshape(-1)
        q = np.maximum(q, 1e-12)
        alpha = float(self.alpha)
        if not np.isfinite(alpha):
            alpha = 0.5
        if alpha < 0.0:
            alpha = 0.0
        if alpha > 1.0:
            alpha = 1.0
        q_alpha = q ** (-alpha)
        Dq = sparse.diags(q_alpha, format="csr")
        K = Dq @ W @ Dq

        # Row-stochastic Markov operator: P = D^{-1} K
        d = np.asarray(K.sum(axis=1)).reshape(-1)
        d = np.maximum(d, 1e-12)
        d_inv_sqrt = d ** (-0.5)
        D_inv_sqrt = sparse.diags(d_inv_sqrt, format="csr")
        # Symmetric conjugate to P for stable eigen computation.
        S = D_inv_sqrt @ K @ D_inv_sqrt

        # Compute leading eigenpairs. Largest eigenvalue ~ 1 (trivial).
        n_eigs = min(n_samples - 1, self.n_components + 1)
        evals, evecs = eigsh(S, k=n_eigs, which="LA")

        # Sort descending.
        order = np.argsort(evals)[::-1]
        evals = evals[order]
        evecs = evecs[:, order]

        # Map back to right eigenvectors of P: psi = D^{-1/2} v
        psi = (d_inv_sqrt[:, None] * evecs)

        # Drop trivial component.
        evals = evals[1 : self.n_components + 1]
        psi = psi[:, 1 : self.n_components + 1]

        # Diffusion time scaling.
        t = int(self.t)
        if t < 1:
            t = 1
        coords = psi * (evals ** t)[None, :]

        assert coords.shape[0] == n_samples, "Unexpected embedding row count"
        assert coords.shape[1] == self.n_components, "Unexpected embedding dimension"
        return coords.astype(np.float32)

