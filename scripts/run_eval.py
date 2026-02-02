#!/usr/bin/env python
"""
Evaluation script for running experiments.

This is the standardized entrypoint for all evaluations.

Usage:
    python -m scripts.run_eval --config configs/default.yaml --dataset mmhal
    python -m scripts.run_eval --config configs/default.yaml --dataset mmhal --debug
"""

import argparse
import sys
import os
from pathlib import Path

# Add paths
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.utils.config import load_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run evaluation")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=['mmhal', 'chair', 'pope'],
        help="Dataset to evaluate on"
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default=None,
        help="Path to pre-computed artifacts (skip fitting)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode"
    )

    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Override dataset
    config['data']['eval_dataset'] = args.dataset

    if args.debug:
        config['experiment']['debug'] = True

    print("="*80)
    print("EVALUATION SCRIPT")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Config: {args.config}")
    print(f"Debug: {args.debug}")
    print("")

    # Import and run main workflow
    # This would import from main.py and run the full pipeline
    # with evaluation enabled

    print("Evaluation would run here...")
    print("(Use main.py for now, this is a placeholder)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
