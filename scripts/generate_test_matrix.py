#!/usr/bin/env python3
"""
Generate GitHub Actions test matrix from Python test configuration.

This script reads test configurations from DeeployTest config files and outputs
JSON arrays suitable for GitHub Actions matrix strategies.
"""

import json
import sys
from pathlib import Path

# Add DeeployTest to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent / "DeeployTest"))

from test_siracusa_tiled_config import (
    L2_SINGLEBUFFER_MODELS,
    L2_DOUBLEBUFFER_MODELS,
    L3_SINGLEBUFFER_MODELS,
    L3_DOUBLEBUFFER_MODELS,
)


def main():
    if len(sys.argv) != 2:
        print("Usage: generate_test_matrix.py <config-key>", file=sys.stderr)
        print("config-key must be one of:", file=sys.stderr)
        print("  l2-singlebuffer-models", file=sys.stderr)
        print("  l2-doublebuffer-models", file=sys.stderr)
        print("  l3-singlebuffer-models", file=sys.stderr)
        print("  l3-doublebuffer-models", file=sys.stderr)
        sys.exit(1)

    config_key = sys.argv[1]

    # Map config keys to Python dictionaries
    config_map = {
        "l2-singlebuffer-models": L2_SINGLEBUFFER_MODELS,
        "l2-doublebuffer-models": L2_DOUBLEBUFFER_MODELS,
        "l3-singlebuffer-models": L3_SINGLEBUFFER_MODELS,
        "l3-doublebuffer-models": L3_DOUBLEBUFFER_MODELS,
    }

    if config_key not in config_map:
        print(f"Error: Unknown config-key '{config_key}'", file=sys.stderr)
        sys.exit(1)

    # Extract test names from the dictionary keys
    test_dict = config_map[config_key]
    test_names = list(test_dict.keys())

    # Output as JSON array
    print(json.dumps(test_names))


if __name__ == "__main__":
    main()
