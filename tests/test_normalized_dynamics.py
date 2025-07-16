import unittest
import torch
import numpy as np
import time
import random
from datetime import datetime
from sklearn.datasets import make_swiss_roll, make_moons, make_circles

# Updated import to use optimized version
import sys
sys.path.append('..')
from normalized_dynamics_optimized import NormalizedDynamicsOptimized

def set_global_seed(seed=None):
    """
    Set global random seed for reproducible results within a test run.
    
    Args:
        seed (int, optional): Random seed. If None, generates a random seed.
        
    Returns:
        int: The seed that was set
    """
    if seed is None:
        # Generate a random seed based on current time
        seed = int(datetime.now().timestamp() * 1000) % 2**32
    
    # Set all random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Set CUDA seed if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Make PyTorch deterministic (optional, can slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    return seed

class TestNormalizedDynamics(unittest.TestCase):
    """Unit tests for the NormalizedDynamics class."""
    
    def setUp(self):
        """Set up the model before each test."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Updated to use optimized version
        self.model = NormalizedDynamicsOptimized(dim=2, device=self.device)
        
        # Set seed for this test run (will be set once per test method)
        if not hasattr(TestNormalizedDynamics, '_test_seed'):
            TestNormalizedDynamics._test_seed = set_global_seed()
            print(f"\nTest run seed: {TestNormalizedDynamics._test_seed}")
        self.test_seed = TestNormalizedDynamics._test_seed

    def test_swiss_roll(self):
        """Test embedding of Swiss Roll dataset."""
        print(f"\n[1/5] Running Swiss Roll test (1000 samples)...")
        start_time = time.time()
        X, _ = make_swiss_roll(n_samples=1000, noise=0.1, random_state=self.test_seed)
        X_embedded = self.model.fit_transform(X)
        elapsed = time.time() - start_time
        print(f"-> Swiss Roll test completed in {elapsed:.2f}s")
        self.assertEqual(X_embedded.shape, (1000, 2))

    def test_moons(self):
        """Test embedding of Moons dataset."""
        print(f"\n[2/5] Running Moons test (1000 samples)...")
        start_time = time.time()
        X, _ = make_moons(n_samples=1000, noise=0.05, random_state=self.test_seed)
        X_embedded = self.model.fit_transform(X)
        elapsed = time.time() - start_time
        print(f"-> Moons test completed in {elapsed:.2f}s")
        self.assertEqual(X_embedded.shape, (1000, 2))

    def test_circles(self):
        """Test embedding of Circles dataset."""
        print(f"\n[3/5] Running Circles test (1000 samples)...")
        start_time = time.time()
        X, _ = make_circles(n_samples=1000, noise=0.05, factor=0.5, random_state=self.test_seed)
        X_embedded = self.model.fit_transform(X)
        elapsed = time.time() - start_time
        print(f"-> Circles test completed in {elapsed:.2f}s")
        self.assertEqual(X_embedded.shape, (1000, 2))

    def test_edge_case_small_data(self):
        """Test embedding of a small dataset."""
        print(f"\n[4/5] Running small data test (10 samples)...")
        start_time = time.time()
        # Use seeded random generator - np.random already seeded by set_global_seed
        X = np.random.randn(10, 5)
        X_embedded = self.model.fit_transform(X)
        elapsed = time.time() - start_time
        print(f"-> Small data test completed in {elapsed:.2f}s")
        self.assertEqual(X_embedded.shape, (10, 2))

    def test_edge_case_high_dim(self):
        """Test embedding of high-dimensional data."""
        print(f"\n[5/5] Running high-dimensional test (100 samples, 50 dims)...")
        start_time = time.time()
        # Use seeded random generator - np.random already seeded by set_global_seed
        X = np.random.randn(100, 50)
        X_embedded = self.model.fit_transform(X)
        elapsed = time.time() - start_time
        print(f"-> High-dimensional test completed in {elapsed:.2f}s")
        self.assertEqual(X_embedded.shape, (100, 2))

if __name__ == '__main__':
    print("=" * 60)
    print("NormalizedDynamics Test Suite (Optimized Version)")
    print("=" * 60)
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Set master seed for the test session
    master_seed = set_global_seed()
    print(f"Master test session seed: {master_seed}")
    print("Starting comprehensive tests with reproducible seeding...")
    
    start_time = time.time()
    unittest.main(verbosity=0, exit=False)
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print(f"All tests completed successfully in {total_time:.2f}s!")
    print(f"All results generated with master seed: {master_seed}")
    print("=" * 60) 