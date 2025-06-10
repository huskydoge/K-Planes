#!/usr/bin/env python3
"""Train K-Plane on DMlab data."""

import os
import subprocess
import sys

def main():
    """Start DMlab training."""
    config_path = "/data/hansen/projects/benhao/K-Planes/plenoxels/configs/final/dmlab/dmlab_explicit.py"
    
    # Check if config exists
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return 1
    
    # Create logs directory
    os.makedirs("logs/dmlab", exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable, "plenoxels/main.py",
        "--config-path", config_path,
        "--seed", "42",
        "--render-only",
        "--log-dir", "/data/hansen/projects/benhao/K-Planes/logs/dmlab/dmlab_debug_train_fr_50_test_fr_50"
    ]
    
    print("Starting DMlab training...")
    print(f"Command: {' '.join(cmd)}")
    print(f"Config: {config_path}")
    print("-" * 50)
    
    try:
        # Run training
        result = subprocess.run(cmd, check=True)
        print("Training completed successfully!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return 130

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 