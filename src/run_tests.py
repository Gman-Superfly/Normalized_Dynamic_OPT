#!/usr/bin/env python3
"""Run the full test suite with deterministic seeding."""

from __future__ import annotations

import os
import random
import subprocess
import sys
import time
from datetime import datetime
from typing import Sequence

import numpy as np
import torch

DEFAULT_TEST_SEED = 42


def set_global_seed(seed: int = DEFAULT_TEST_SEED) -> int:
    """Set random seeds for deterministic test behavior.

    Args:
        seed: Random seed value used across Python, NumPy, and PyTorch.

    Returns:
        Seed value that was applied.

    Raises:
        AssertionError: If seed is not an integer.
    """
    assert isinstance(seed, int), f"seed must be int, got {type(seed)}"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    return seed


def run_command(command: Sequence[str], working_directory: str) -> int:
    """Run a subprocess command and print captured output.

    Args:
        command: Command and arguments to execute.
        working_directory: Directory where the command runs.

    Returns:
        Process exit code.
    """
    assert len(command) > 0, "command cannot be empty"
    assert os.path.isdir(working_directory), f"Invalid cwd: {working_directory}"

    result = subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        cwd=working_directory,
        check=False,
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return int(result.returncode)


def main() -> int:
    """Execute tests and return a nonzero code on failure."""
    print("=" * 80)
    print("NormalizedDynamics Test Runner")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    session_seed = set_global_seed(DEFAULT_TEST_SEED)
    print(f"Test session seed: {session_seed}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("Deterministic seed is active for this run.")
    print()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert os.path.isdir(repo_root), f"Repository root not found: {repo_root}"
    os.chdir(repo_root)

    os.makedirs("static/results", exist_ok=True)
    os.makedirs("static/results/individual", exist_ok=True)
    os.makedirs("static/results/comprehensive", exist_ok=True)

    total_start = time.time()
    failing_steps = 0

    print("1. Running repository tests with pytest")
    print("-" * 50)
    pytest_code = run_command(
        [sys.executable, "-m", "pytest", "tests", "-v"],
        working_directory=repo_root,
    )
    if pytest_code != 0:
        failing_steps += 1
        print(f"Pytest failed with exit code {pytest_code}")
    else:
        print("Pytest completed successfully")

    print()
    print("2. Running visualization script")
    print("-" * 50)
    visualization_code = run_command(
        [sys.executable, "tests/test_comprehensive_visualizations.py"],
        working_directory=repo_root,
    )
    if visualization_code != 0:
        failing_steps += 1
        print(f"Visualization script failed with exit code {visualization_code}")
    else:
        print("Visualization script completed successfully")

    total_time = time.time() - total_start

    print()
    print("=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Session seed used: {session_seed}")
    print("=" * 80)

    if failing_steps > 0:
        print(f"Failing steps: {failing_steps}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())