"""
Data loading utilities.
"""

from typing import Any, Tuple, Optional
import sys
import os

# Add paths (we're already in Latent-space-steering)
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))


def load_demo_data(
    config: dict,
    image_processor: Any,
    model: Any,
    tokenizer: Any,
    device: str = "cuda",
    max_samples: Optional[int] = None,
    debug: bool = False
) -> Tuple[Any, Any]:
    """
    Load demonstration data for computing steering directions.

    Args:
        config: Configuration dict with data parameters
        image_processor: Image processor
        model: LVLM model
        tokenizer: Tokenizer
        device: Device
        max_samples: Maximum number of samples to load
        debug: If True, load minimal data

    Returns:
        input_images: List of image tensors
        input_ids: List of tokenized prompts
    """
    if debug:
        print("[DEBUG MODE] Creating mock demo data...")
        # Return minimal mock data
        import torch
        mock_images = [(
            [torch.randn(3, 224, 224) for _ in range(2)],  # corrupted images
            torch.randn(3, 224, 224)  # original image
        )]
        mock_ids = [
            (torch.randint(0, 1000, (1, 10)), torch.randint(0, 1000, (1, 10)))
        ]
        return mock_images, mock_ids

    from vti_utils.utils import get_demos

    # Create args namespace from config
    class Args:
        def __init__(self, cfg):
            self.data_file = cfg['data']['data_file']
            self.num_demos = cfg['data']['num_demos']
            self.mask_ratio = cfg['steering']['config']['mask_ratio']
            self.num_trials = cfg['steering']['config']['num_trials']
            self.conv_mode = cfg['model']['conv_mode']

    args = Args(config)

    # Override num_demos if max_samples specified
    if max_samples is not None:
        args.num_demos = min(args.num_demos, max_samples)

    # Load demos
    demo_file = config['data']['demo_file']
    input_images, input_ids = get_demos(
        args,
        image_processor,
        model,
        tokenizer,
        file_path=demo_file,
        device=device
    )

    return input_images, input_ids


def load_eval_dataset(
    dataset_name: str,
    config: dict,
    debug: bool = False
):
    """
    Load evaluation dataset.

    Args:
        dataset_name: Name of dataset (mmhal, chair, pope)
        config: Configuration dict
        debug: If True, load minimal data

    Returns:
        dataset: Loaded dataset
    """
    if debug:
        print(f"[DEBUG MODE] Loading mock {dataset_name} dataset...")
        # Return minimal mock dataset
        return [
            {
                'image_path': 'mock_image.jpg',
                'question': 'What is in the image?',
                'gt_answer': 'Mock answer',
                'image_id': '0'
            }
        ]

    if dataset_name == "mmhal":
        from datasets import load_dataset
        dataset = load_dataset("Shengcao1006/MMHal-Bench")['test']
        return dataset
    elif dataset_name == "chair":
        # Placeholder for CHAIR
        raise NotImplementedError("CHAIR dataset not yet implemented")
    elif dataset_name == "pope":
        # Placeholder for POPE
        raise NotImplementedError("POPE dataset not yet implemented")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
