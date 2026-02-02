#!/usr/bin/env python
"""
Baseline evaluation script for MMHal-Bench.

Tests the original performance of LLaVA without any steering.

Usage:
    python scripts/evaluate_mmhal_baseline.py
    python scripts/evaluate_mmhal_baseline.py --debug
    python scripts/evaluate_mmhal_baseline.py --output results/llava_baseline.json
"""

import argparse
import sys
import os
from pathlib import Path
import json
import torch
from tqdm import tqdm
import requests

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.models.loader import load_model
from src.utils.logging_utils import setup_logging, create_experiment_dir, save_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="MMHal-Bench Baseline Evaluation")

    parser.add_argument("--model-name", type=str, default="liuhaotian/llava-v1.5-7b",
                       help="Model to evaluate")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file path (default: auto-generated)")
    parser.add_argument("--debug", action="store_true",
                       help="Debug mode with minimal samples")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                       help="Maximum tokens to generate")
    parser.add_argument("--num-beams", type=int, default=1,
                       help="Number of beams for generation")

    return parser.parse_args()


def load_mmhal_dataset(debug=False):
    """Load MMHal-Bench dataset."""
    print("Loading MMHal-Bench dataset...")

    if debug:
        # Mock dataset for debug
        return [{
            'question': 'What is in this image?',
            'image_src': 'https://llava-vl.github.io/static/images/view.jpg',
            'image_content': ['dog', 'cat'],
            'gt_answer': 'A dog and a cat.',
            'question_type': 'object recognition'
        }]

    # Download MMHal-Bench template if not exists
    template_path = Path("data/mmhal_bench_response_template.json")

    if not template_path.exists():
        print("Downloading MMHal-Bench template...")
        template_path.parent.mkdir(parents=True, exist_ok=True)

        # Download from the official repo
        url = "https://raw.githubusercontent.com/Shengcao1006/MMHal-Bench/main/response_template.json"
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(template_path, 'w') as f:
                f.write(response.text)
            print(f"✓ Downloaded to {template_path}")
        except Exception as e:
            print(f"Error downloading template: {e}")
            print("Please manually download from:")
            print("https://github.com/Shengcao1006/MMHal-Bench/blob/main/response_template.json")
            print(f"Save to: {template_path}")
            raise

    # Load the template
    with open(template_path, 'r') as f:
        dataset = json.load(f)

    print(f"✓ Loaded {len(dataset)} samples from MMHal-Bench")
    return dataset


def main():
    args = parse_args()

    # Create experiment directory
    exp_name = "mmhal_baseline"
    exp_dir = create_experiment_dir(
        base_dir="./outputs",
        experiment_name=exp_name
    )

    # Setup logging
    logger = setup_logging(
        log_dir=exp_dir,
        log_level=20,  # INFO level
        experiment_name=exp_name
    )

    logger.info("="*80)
    logger.info("MMHal-Bench Baseline Evaluation")
    logger.info("="*80)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Output: {exp_dir}")
    logger.info(f"Debug: {args.debug}")
    logger.info("")

    # Load model
    logger.info("Loading model...")
    tokenizer, model, image_processor, _ = load_model(
        model_name=args.model_name,
        device="auto",
        debug=args.debug
    )
    device = next(model.parameters()).device
    logger.info(f"✓ Model loaded on {device}")

    # Load dataset
    dataset = load_mmhal_dataset(debug=args.debug)
    logger.info(f"✓ Loaded {len(dataset)} samples")

    # Run inference
    logger.info("\nRunning inference...")
    results = []

    if not args.debug:
        from llava.mm_utils import tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX
        from PIL import Image
        import requests
        from io import BytesIO

        def load_image(image_url):
            """Load image from URL or path."""
            if image_url.startswith('http'):
                response = requests.get(image_url)
                image = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                image = Image.open(image_url).convert('RGB')
            return image

        for idx, item in enumerate(tqdm(dataset, desc="Evaluating")):
            try:
                # Load image
                image = load_image(item['image_src'])

                # Prepare prompt
                question = item['question']
                prompt = f"USER: <image>\n{question} ASSISTANT:"

                # Tokenize
                input_ids = tokenizer_image_token(
                    prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
                ).unsqueeze(0).to(device)

                # Process image
                image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                image_tensor = image_tensor.unsqueeze(0).to(device)

                # Match model dtype
                if device != torch.device("cpu"):
                    # Ensure both input_ids and image_tensor match model dtype
                    if model.dtype == torch.float16:
                        image_tensor = image_tensor.half()

                # Generate
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensor,
                        max_new_tokens=args.max_new_tokens,
                        num_beams=args.num_beams,
                        use_cache=False,
                        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                    )

                output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                output = output.strip()

                # Save result
                result = {
                    'question_type': item.get('question_type', ''),
                    'question_topic': item.get('question_topic', ''),
                    'image_id': item.get('image_id', idx),
                    'image_src': item['image_src'],
                    'image_content': item.get('image_content', []),
                    'question': question,
                    'gt_answer': item.get('gt_answer', ''),
                    'model_answer': output
                }
                results.append(result)

            except Exception as e:
                logger.error(f"Error processing sample {idx}: {e}")
                continue

    else:
        # Debug mode - mock results
        logger.info("[DEBUG MODE] Using mock responses")
        for item in dataset:
            result = {
                'question': item['question'],
                'gt_answer': item.get('gt_answer', ''),
                'model_answer': '[DEBUG] Mock model response',
                'image_content': item.get('image_content', [])
            }
            results.append(result)

    # Save results
    output_file = args.output if args.output else exp_dir / "mmhal_responses.json"
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✓ Results saved to: {output_file}")
    logger.info(f"  Total samples: {len(results)}")

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("Evaluation Complete")
    logger.info("="*80)
    logger.info(f"Results file: {output_file}")
    logger.info(f"Total samples evaluated: {len(results)}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Manually review the responses, OR")
    logger.info("2. Use GPT-based evaluation:")
    logger.info(f"   python scripts/evaluate_mmhal_gpt.py --response {output_file}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
