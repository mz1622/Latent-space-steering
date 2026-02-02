#!/usr/bin/env python
"""
Main entry point for hallucination mitigation experiments.

This script demonstrates the complete workflow:
1. Load configuration
2. Load model and data
3. Compute steering directions (fit)
4. Apply steering to model
5. Run inference
6. Evaluate results
7. Save outputs

Usage:
    python main.py --config configs/default.yaml --debug
    python main.py --config configs/default.yaml --max-samples 10
    python main.py --config configs/default.yaml --method mean_difference --alpha-text 0.4 --alpha-image 0.2
"""

import argparse
import sys
import os
from pathlib import Path
import torch
from transformers import set_seed

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
sys.path.append(os.path.dirname(__file__))

from src.utils.config import load_config, save_config
from src.utils.logging_utils import (
    setup_logging,
    create_experiment_dir,
    save_metrics,
    save_environment_info,
    save_git_info
)
from src.models.loader import load_model
from src.data.loader import load_demo_data, load_eval_dataset
from src.steering.mean_difference import MeanDifferenceSteeringMethod
from src.detectors.base import PassthroughDetector


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Hallucination mitigation via latent space steering"
    )

    # Core arguments
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode (skip expensive computations)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of training samples (for quick testing)"
    )

    # Override arguments
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Steering method to use (overrides config)"
    )
    parser.add_argument(
        "--alpha-image",
        type=float,
        default=None,
        help="Visual steering strength (overrides config)"
    )
    parser.add_argument(
        "--alpha-text",
        type=float,
        default=None,
        help="Textual steering strength (overrides config)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (overrides config)"
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Apply command-line overrides
    if args.debug:
        config['experiment']['debug'] = True
    if args.max_samples is not None:
        config['experiment']['max_samples'] = args.max_samples
    if args.method is not None:
        config['steering']['method'] = args.method
    if args.alpha_image is not None:
        config['steering']['config']['alpha_image'] = args.alpha_image
    if args.alpha_text is not None:
        config['steering']['config']['alpha_text'] = args.alpha_text
    if args.output_dir is not None:
        config['output']['base_dir'] = args.output_dir

    # Extract settings
    debug = config['experiment']['debug']
    max_samples = config['experiment']['max_samples']
    seed = config['experiment']['seed']
    experiment_name = config['experiment']['name']

    # Set random seed
    set_seed(seed)

    # Create experiment directory
    exp_dir = create_experiment_dir(
        base_dir=config['output']['base_dir'],
        experiment_name=experiment_name
    )

    # Setup logging
    logger = setup_logging(
        log_dir=exp_dir,
        experiment_name=experiment_name
    )

    logger.info("="*80)
    logger.info("HALLUCINATION MITIGATION VIA LATENT SPACE STEERING")
    logger.info("="*80)
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Output directory: {exp_dir}")
    logger.info(f"Debug mode: {debug}")
    logger.info(f"Max samples: {max_samples}")
    logger.info("")

    # Save configuration
    if config['output']['save_config']:
        save_config(config, exp_dir / "config.yaml")
        logger.info(f"✓ Configuration saved to {exp_dir / 'config.yaml'}")

    # Save environment info
    if config['output']['save_environment']:
        save_environment_info(exp_dir / "environment.txt")
        save_git_info(exp_dir / "git_commit.txt")
        logger.info("✓ Environment info saved")

    logger.info("")
    logger.info("-"*80)
    logger.info("PHASE 1: Model Loading")
    logger.info("-"*80)

    # Load model
    tokenizer, model, image_processor, context_len = load_model(
        model_name=config['model']['name'],
        model_base=config['model']['model_base'],
        device=config['model']['device'],
        debug=debug
    )

    device = next(model.parameters()).device
    logger.info(f"✓ Model loaded on {device}")

    logger.info("")
    logger.info("-"*80)
    logger.info("PHASE 2: Data Loading")
    logger.info("-"*80)

    # Load demonstration data for computing directions
    input_images, input_ids = load_demo_data(
        config=config,
        image_processor=image_processor,
        model=model,
        tokenizer=tokenizer,
        device=str(device),
        max_samples=max_samples,
        debug=debug
    )

    logger.info(f"✓ Loaded {len(input_images)} demonstration samples")

    logger.info("")
    logger.info("-"*80)
    logger.info("PHASE 3: Hallucination Detection (Placeholder)")
    logger.info("-"*80)

    # Initialize detector (placeholder for now)
    detector_type = config['detector']['type']
    logger.info(f"Detector type: {detector_type}")

    if detector_type == "passthrough":
        detector = PassthroughDetector(
            config=config['detector']['config'],
            debug=debug
        )
        logger.info("✓ PassthroughDetector initialized (no detection, always steer)")
    else:
        raise NotImplementedError(f"Detector {detector_type} not implemented")

    logger.info("")
    logger.info("-"*80)
    logger.info("PHASE 4: Steering Direction Computation")
    logger.info("-"*80)

    # Initialize steering method
    method_name = config['steering']['method']
    logger.info(f"Steering method: {method_name}")

    if method_name == "mean_difference":
        steering_method = MeanDifferenceSteeringMethod(
            config=config['steering']['config'],
            debug=debug
        )
    else:
        raise NotImplementedError(f"Method {method_name} not implemented")

    # Fit steering directions
    logger.info("Computing steering directions...")
    artifacts = steering_method.fit(
        model=model,
        train_data=(input_images, input_ids),
        max_samples=max_samples
    )

    logger.info(f"✓ Steering directions computed")

    # Save artifacts
    if config['output']['save_artifacts']:
        artifacts_dir = exp_dir / "artifacts"
        steering_method.save_artifacts(artifacts_dir)
        logger.info(f"✓ Artifacts saved to {artifacts_dir}")

    logger.info("")
    logger.info("-"*80)
    logger.info("PHASE 5: Apply Steering")
    logger.info("-"*80)

    # Apply steering to model
    alpha_image = config['steering']['config']['alpha_image']
    alpha_text = config['steering']['config']['alpha_text']
    logger.info(f"Applying steering (alpha_image={alpha_image}, alpha_text={alpha_text})")

    steering_method.apply(model)

    logger.info("✓ Steering applied to model")

    logger.info("")
    logger.info("-"*80)
    logger.info("PHASE 6: Inference & Evaluation")
    logger.info("-"*80)

    if debug:
        logger.info("[DEBUG MODE] Skipping full evaluation")
        logger.info("In non-debug mode, this would:")
        logger.info("  1. Load evaluation dataset (MMHal, CHAIR, etc.)")
        logger.info("  2. Run inference with steering")
        logger.info("  3. Compute evaluation metrics")
        logger.info("  4. Save predictions and results")
    else:
        logger.info("Full evaluation would run here")
        logger.info("(Not implemented in this initial version)")

    # Mock metrics for demonstration
    metrics = {
        'experiment': experiment_name,
        'method': method_name,
        'alpha_image': alpha_image,
        'alpha_text': alpha_text,
        'num_training_samples': len(input_images),
        'debug': debug
    }

    if debug:
        metrics['note'] = 'Debug mode - no real evaluation performed'

    # Save metrics
    save_metrics(metrics, exp_dir / "metrics.json")
    logger.info(f"✓ Metrics saved to {exp_dir / 'metrics.json'}")

    logger.info("")
    logger.info("-"*80)
    logger.info("PHASE 7: Cleanup")
    logger.info("-"*80)

    # Remove steering from model
    steering_method.remove(model)
    logger.info("✓ Steering removed from model")

    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("✓ GPU cache cleared")

    logger.info("")
    logger.info("="*80)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("="*80)
    logger.info(f"Results saved to: {exp_dir}")
    logger.info("")

    # Print summary
    logger.info("Summary:")
    logger.info(f"  - Method: {method_name}")
    logger.info(f"  - Training samples: {len(input_images)}")
    logger.info(f"  - Alpha (image): {alpha_image}")
    logger.info(f"  - Alpha (text): {alpha_text}")
    logger.info(f"  - Debug mode: {debug}")
    logger.info("")

    return exp_dir


if __name__ == "__main__":
    main()
