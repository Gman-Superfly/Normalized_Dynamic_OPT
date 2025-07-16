#!/usr/bin/env python3
"""Quick test of smart sampling analysis to debug hanging issue."""

import sys
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Add project paths (we're now in src/, so go to parent directory)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, os.path.join(project_root, 'tests'))

def test_smart_sampling():
    """Test the smart sampling analysis function."""
    try:
        print("[ROCKET] Testing smart sampling analysis...")
        
        # Import the function
        from tests.smart_sampling_enhanced_analysis import run_smart_sampling_analysis
        
        print("[CHECK] Successfully imported smart sampling function")
        print("[CHART] Starting analysis...")
        
        # Run with timeout simulation
        import time
        start_time = time.time()
        
        # Run the analysis
        results, embeddings, sampling_info = run_smart_sampling_analysis()
        
        end_time = time.time()
        
        print(f"[CHECK] Analysis completed in {end_time - start_time:.1f}s")
        print(f"[CHART] Results keys: {list(results.keys()) if results else 'None'}")
        print(f"[CHART] Embeddings keys: {list(embeddings.keys()) if embeddings else 'None'}")
        print(f"[CHART] Sampling info keys: {list(sampling_info.keys()) if sampling_info else 'None'}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_smart_sampling()
    if success:
        print("[PARTY] Smart sampling test completed successfully!")
    else:
        print("[BOOM] Smart sampling test failed!") 