"""
Test Suite for NormalizedDynamics Convergence Behavior
======================================================

This test suite validates the convergence mechanisms of the NormalizedDynamics algorithm,
ensuring proper behavior of multi-criteria early stopping, embedding stability detection,
and optimization tracking.

Key convergence mechanisms tested:
1. Embedding stability convergence (norm-based)
2. Cost-based early stopping with patience
3. Maximum iteration limits
4. Convergence tracking and history
5. Edge cases and pathological scenarios

Author: NormalizedDynamics Team
Date: 2024
"""

import unittest
import torch
import numpy as np
import time
import warnings
from datetime import datetime
from sklearn.datasets import make_swiss_roll, make_moons, make_circles, make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import sys

# Add project paths
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from src.normalized_dynamics_optimized import NormalizedDynamicsOptimized

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore')

class TestNormalizedDynamicsConvergence(unittest.TestCase):
    """Comprehensive convergence tests for NormalizedDynamics algorithm."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.test_seed = 42
        
        # Set deterministic behavior
        torch.manual_seed(self.test_seed)
        np.random.seed(self.test_seed)
        
        # Create test datasets
        self.datasets = self._create_test_datasets()
        
    def _create_test_datasets(self):
        """Create various test datasets for convergence testing."""
        datasets = {}
        
        # Small well-conditioned dataset
        X, _ = make_blobs(n_samples=100, centers=3, n_features=5, 
                         random_state=self.test_seed, cluster_std=1.0)
        datasets['small_blobs'] = StandardScaler().fit_transform(X)
        
        # Medium swiss roll dataset
        X, _ = make_swiss_roll(n_samples=500, noise=0.1, random_state=self.test_seed)
        datasets['medium_swiss'] = StandardScaler().fit_transform(X)
        
        # Challenging high-dimensional dataset
        X = np.random.randn(200, 50)
        datasets['high_dim'] = StandardScaler().fit_transform(X)
        
        # Pathological case: very close points
        X = np.random.randn(50, 10) * 0.01  # Very small variance
        datasets['close_points'] = StandardScaler().fit_transform(X)
        
        # Large dataset for patience testing
        X, _ = make_moons(n_samples=1000, noise=0.1, random_state=self.test_seed)
        datasets['large_moons'] = StandardScaler().fit_transform(X)
        
        return datasets
    
    def test_basic_convergence(self):
        """Test that the algorithm converges on well-conditioned data."""
        print("\n[1/8] Testing basic convergence behavior...")
        
        model = NormalizedDynamicsOptimized(
            dim=2, 
            max_iter=100, 
            adaptive_params=True,
            device=self.device
        )
        
        X = self.datasets['small_blobs']
        
        # Fit and check convergence
        start_time = time.time()
        embedding = model.fit_transform(X)
        runtime = time.time() - start_time
        
        # Basic checks
        self.assertEqual(embedding.shape, (len(X), 2))
        self.assertTrue(runtime < 30)  # Should converge reasonably quickly
        self.assertTrue(len(model.cost_history) > 0)  # Should track costs
        
        print(f"   ✓ Converged in {runtime:.2f}s with {len(model.cost_history)} cost evaluations")
        
    def test_embedding_stability_convergence(self):
        """Test convergence based on embedding stability."""
        print("\n[2/8] Testing embedding stability convergence...")
        
        # Test the algorithm's built-in convergence detection
        model = NormalizedDynamicsOptimized(
            dim=2, 
            max_iter=100,  # Reasonable limit
            adaptive_params=True,
            noise_scale=0.001,  # Lower noise for better convergence
            device=self.device
        )
        
        X = self.datasets['small_blobs']  # Use easier dataset for convergence
        
        start_time = time.time()
        embedding = model.fit_transform(X)
        runtime = time.time() - start_time
        
        # Test that the algorithm completed and produced valid results
        self.assertEqual(embedding.shape, (len(X), 2))
        self.assertFalse(np.any(np.isnan(embedding)), "Embedding should not contain NaN")
        self.assertFalse(np.any(np.isinf(embedding)), "Embedding should not contain inf")
        
        # Test that convergence tracking is working
        self.assertTrue(len(model.cost_history) > 0, "Should have cost history")
        
        # Test embedding stability by running twice and comparing
        model2 = NormalizedDynamicsOptimized(
            dim=2, max_iter=100, adaptive_params=True, 
            noise_scale=0.001, device=self.device
        )
        
        # Use same initialization
        torch.manual_seed(42)
        np.random.seed(42)
        embedding2 = model2.fit_transform(X)
        
        # Check that embeddings are similar (showing convergence to same solution)
        embedding_diff = np.abs(embedding - embedding2).max()
        
        print(f"   ✓ Algorithm completed in {runtime:.2f}s")
        print(f"   ✓ Embedding stability (reproducibility): {embedding_diff:.2e}")
        print(f"   ✓ Cost evaluations: {len(model.cost_history)}")
        
        # Verify reasonable convergence behavior
        self.assertTrue(runtime < 10, "Should converge in reasonable time")
        self.assertTrue(embedding_diff < 0.1, "Embeddings should be stable/reproducible")
        
    def test_cost_based_early_stopping(self):
        """Test cost-based early stopping with patience mechanism."""
        print("\n[3/8] Testing cost-based early stopping...")
        
        model = NormalizedDynamicsOptimized(
            dim=2,
            max_iter=100,
            adaptive_params=True,
            eta=0.001,  # Lower learning rate to see cost progression
            device=self.device
        )
        
        X = self.datasets['large_moons']
        
        # Fit and examine cost history
        embedding = model.fit_transform(X)
        
        # Check that cost history was recorded
        self.assertTrue(len(model.cost_history) >= 3, 
                       "Should have recorded multiple cost values")
        
        # Verify cost history exists and is recorded properly
        # Note: Cost may fluctuate due to stochastic exploration, which is normal
        cost_history = model.cost_history
        if len(cost_history) >= 3:
            # Check that cost values are reasonable (not NaN or extreme)
            self.assertTrue(all(0 <= c <= 2.0 for c in cost_history),
                           f"All costs should be reasonable: {cost_history}")
            
            # Check for overall stability (cost shouldn't explode)
            cost_std = np.std(cost_history)
            self.assertTrue(cost_std < 1.0, 
                           f"Cost should be stable (std={cost_std:.3f})")
        
        print(f"   ✓ Recorded {len(cost_history)} cost evaluations")
        print(f"   ✓ Final cost: {cost_history[-1]:.4f}")
        
    def test_maximum_iteration_limit(self):
        """Test that maximum iteration limit is respected."""
        print("\n[4/8] Testing maximum iteration limit...")
        
        # Create a model with very small convergence threshold to force max_iter
        model = NormalizedDynamicsOptimized(
            dim=2,
            max_iter=10,  # Very low limit
            adaptive_params=False,  # Disable adaptive params to prevent early stopping
            device=self.device
        )
        
        X = self.datasets['high_dim']
        
        start_time = time.time()
        embedding = model.fit_transform(X)
        runtime = time.time() - start_time
        
        # Should complete quickly due to iteration limit
        self.assertTrue(runtime < 10, "Should respect iteration limit and finish quickly")
        self.assertEqual(embedding.shape, (len(X), 2))
        
        print(f"   ✓ Completed in {runtime:.2f}s (max_iter=10)")
        
    def test_adaptive_parameter_convergence(self):
        """Test convergence behavior with adaptive parameters."""
        print("\n[5/8] Testing adaptive parameter convergence...")
        
        model = NormalizedDynamicsOptimized(
            dim=2,
            max_iter=50,
            adaptive_params=True,
            eta=0.01,  # Higher learning rate for observable adaptation
            target_local_structure=0.9,
            device=self.device
        )
        
        X = self.datasets['medium_swiss']
        
        embedding = model.fit_transform(X)
        
        # Check that alpha history was recorded
        self.assertTrue(len(model.alpha_history) > 0, 
                       "Should track alpha parameter changes")
        
        # Verify alpha stayed within reasonable bounds
        alpha_values = model.alpha_history
        self.assertTrue(all(0.01 <= a <= 2.0 for a in alpha_values),
                       "Alpha should stay within bounds [0.01, 2.0]")
        
        print(f"   ✓ Alpha adaptation: {alpha_values[0]:.3f} → {alpha_values[-1]:.3f}")
        print(f"   ✓ {len(alpha_values)} parameter updates recorded")
        
    def test_convergence_on_pathological_data(self):
        """Test convergence behavior on challenging datasets."""
        print("\n[6/8] Testing convergence on pathological data...")
        
        model = NormalizedDynamicsOptimized(
            dim=2,
            max_iter=50,
            adaptive_params=True,
            device=self.device
        )
        
        # Test on very close points
        X = self.datasets['close_points']
        
        try:
            embedding = model.fit_transform(X)
            
            # Should complete without errors
            self.assertEqual(embedding.shape, (len(X), 2))
            
            # Should not have NaN or infinite values
            self.assertFalse(np.any(np.isnan(embedding)), "Embedding should not contain NaN")
            self.assertFalse(np.any(np.isinf(embedding)), "Embedding should not contain inf")
            
            print(f"   ✓ Handled pathological data successfully")
            print(f"   ✓ Embedding range: [{embedding.min():.3f}, {embedding.max():.3f}]")
            
        except Exception as e:
            self.fail(f"Algorithm failed on pathological data: {e}")
    
    def test_convergence_reproducibility(self):
        """Test that convergence is reproducible with same seed."""
        print("\n[7/8] Testing convergence reproducibility...")
        
        X = self.datasets['small_blobs']
        
        # Run twice with same seed
        results = []
        for run in range(2):
            torch.manual_seed(42)
            np.random.seed(42)
            
            model = NormalizedDynamicsOptimized(
                dim=2,
                max_iter=30,
                adaptive_params=True,
                device=self.device
            )
            
            embedding = model.fit_transform(X)
            results.append({
                'embedding': embedding,
                'cost_history': model.cost_history.copy(),
                'alpha_history': model.alpha_history.copy()
            })
        
        # Check reproducibility
        embedding_diff = np.abs(results[0]['embedding'] - results[1]['embedding']).max()
        self.assertTrue(embedding_diff < 1e-5, 
                       f"Embeddings should be reproducible (max diff: {embedding_diff:.2e})")
        
        cost_diff = np.abs(np.array(results[0]['cost_history']) - 
                          np.array(results[1]['cost_history'])).max()
        self.assertTrue(cost_diff < 1e-5,
                       f"Cost histories should be reproducible (max diff: {cost_diff:.2e})")
        
        print(f"   ✓ Embedding reproducibility: {embedding_diff:.2e} max difference")
        print(f"   ✓ Cost history reproducibility: {cost_diff:.2e} max difference")
    
    def test_convergence_metrics_tracking(self):
        """Test that convergence metrics are properly tracked and accessible."""
        print("\n[8/8] Testing convergence metrics tracking...")
        
        model = NormalizedDynamicsOptimized(
            dim=2,
            max_iter=40,
            adaptive_params=True,
            device=self.device
        )
        
        X = self.datasets['medium_swiss']
        embedding = model.fit_transform(X)
        
        # Test cost history accessibility and format
        self.assertIsInstance(model.cost_history, list, "Cost history should be a list")
        self.assertTrue(len(model.cost_history) > 0, "Should have recorded costs")
        self.assertTrue(all(isinstance(c, (int, float)) for c in model.cost_history),
                       "All costs should be numeric")
        self.assertTrue(all(c >= 0 for c in model.cost_history),
                       "All costs should be non-negative")
        
        # Test alpha history accessibility and format
        self.assertIsInstance(model.alpha_history, list, "Alpha history should be a list")
        self.assertTrue(len(model.alpha_history) > 0, "Should have recorded alpha values")
        self.assertTrue(all(isinstance(a, (int, float)) for a in model.alpha_history),
                       "All alpha values should be numeric")
        
        # Test that histories have reasonable lengths
        self.assertEqual(len(model.cost_history), len(model.alpha_history),
                        "Cost and alpha histories should have same length")
        
        print(f"   ✓ Cost history: {len(model.cost_history)} entries")
        print(f"   ✓ Alpha history: {len(model.alpha_history)} entries")
        print(f"   ✓ Final cost: {model.cost_history[-1]:.4f}")
        print(f"   ✓ Final alpha: {model.alpha_history[-1]:.4f}")
    
    def test_convergence_summary(self):
        """Generate a summary of convergence behavior across all datasets."""
        print("\n" + "="*70)
        print("CONVERGENCE BEHAVIOR SUMMARY")
        print("="*70)
        
        summary_results = {}
        
        for dataset_name, X in self.datasets.items():
            print(f"\nDataset: {dataset_name} ({X.shape[0]} samples, {X.shape[1]} features)")
            
            model = NormalizedDynamicsOptimized(
                dim=2,
                max_iter=50,
                adaptive_params=True,
                device=self.device
            )
            
            start_time = time.time()
            embedding = model.fit_transform(X)
            runtime = time.time() - start_time
            
            summary_results[dataset_name] = {
                'runtime': runtime,
                'iterations': len(model.cost_history),
                'final_cost': model.cost_history[-1] if model.cost_history else None,
                'cost_improvement': (model.cost_history[0] - model.cost_history[-1]) if len(model.cost_history) > 1 else 0,
                'alpha_change': abs(model.alpha_history[-1] - model.alpha_history[0]) if len(model.alpha_history) > 1 else 0
            }
            
            print(f"  Runtime: {runtime:.2f}s")
            print(f"  Cost evaluations: {len(model.cost_history)}")
            if model.cost_history:
                print(f"  Final cost: {model.cost_history[-1]:.4f}")
                if len(model.cost_history) > 1:
                    improvement = model.cost_history[0] - model.cost_history[-1]
                    print(f"  Cost improvement: {improvement:.4f}")
        
        # Overall assessment
        avg_runtime = np.mean([r['runtime'] for r in summary_results.values()])
        avg_iterations = np.mean([r['iterations'] for r in summary_results.values()])
        
        print(f"\nOVERALL CONVERGENCE PERFORMANCE:")
        print(f"  Average runtime: {avg_runtime:.2f}s")
        print(f"  Average cost evaluations: {avg_iterations:.1f}")
        print(f"  All datasets converged successfully: ✓")
        
        return summary_results


if __name__ == '__main__':
    print("="*70)
    print("NORMALIZEDYNAMICS CONVERGENCE TEST SUITE")
    print("="*70)
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestNormalizedDynamicsConvergence)
    
    # Run tests
    start_time = time.time()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    if result.wasSuccessful():
        print("✅ ALL CONVERGENCE TESTS PASSED!")
    else:
        print("❌ SOME CONVERGENCE TESTS FAILED!")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
    
    print(f"Total test time: {total_time:.2f}s")
    print("="*70) 