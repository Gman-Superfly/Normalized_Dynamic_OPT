#!/usr/bin/env python3
"""
Test Runner for NormalizedDynamics

This script runs all tests and generates comprehensive visualizations.
Results are saved in the static/results/ directory with organized subfolders.
"""

import os
import sys
import subprocess
import time
import torch
import numpy as np
import random
from datetime import datetime

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
    
    return seed

def main():
    """Run all tests and generate visualizations."""
    print("="*80)
    print("NormalizedDynamics Test Runner")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Set global seed for the entire test session
    session_seed = set_global_seed()
    print(f"Test session seed: {session_seed}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("All tests will use reproducible seeding within this session.")
    print()
    
    # Ensure we're in the right directory (go to parent directory since we're in src/)
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(script_dir)
    
    # Create results directory if it doesn't exist
    os.makedirs("static/results", exist_ok=True)
    os.makedirs("static/results/individual", exist_ok=True)
    os.makedirs("static/results/comprehensive", exist_ok=True)
    
    total_start = time.time()
    
    try:
        # Run basic unit tests
        print("1. Running Basic Unit Tests")
        print("-" * 50)
        result = subprocess.run([
            sys.executable, "-m", "pytest", "tests/test_normalized_dynamics.py", "-v"
        ], capture_output=True, text=True, cwd=script_dir)
        
        if result.returncode == 0:
            print("   Basic unit tests passed")
        else:
            print("   Basic unit tests encountered issues:")
            print(result.stdout)
            print(result.stderr)
        
        print()
        
        # Run comprehensive visualization tests
        print("2. Running Comprehensive Visualization Tests")
        print("-" * 50)
        
        # Change to tests directory and run comprehensive tests
        os.chdir("tests")
        result = subprocess.run([
            sys.executable, "test_comprehensive_visualizations.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("   Comprehensive tests completed successfully")
            print(result.stdout)
        else:
            print("   Comprehensive tests encountered issues:")
            print(result.stdout)
            print(result.stderr)
        
        # Change back to script directory
        os.chdir(script_dir)
        
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1
    
    total_time = time.time() - total_start
    
    print()
    print("="*80)
    print("Test Summary")
    print("="*80)
    print(f"Total execution time: {total_time:.2f} seconds")
    print()
    print("Generated files:")
    
    # List generated files
    results_dir = "static/results"
    if os.path.exists(results_dir):
        for root, dirs, files in os.walk(results_dir):
            for file in files:
                if file.endswith('.png'):
                    rel_path = os.path.relpath(os.path.join(root, file), script_dir)
                    print(f"  {rel_path}")
    
    print()
    print("All tests completed successfully.")
    print(f"Session seed used: {session_seed}")
    print(f"Check the 'static/results/' directory for visualization outputs.")
    print("="*80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 