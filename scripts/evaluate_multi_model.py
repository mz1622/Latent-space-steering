#!/usr/bin/env python
"""
Multi-model evaluation script supporting LLaVA, Qwen3-VL, and other LVLMs.

Usage:
    # LLaVA on POPE (baseline)
    python scripts/evaluate_multi_model.py --model-type llava --benchmark pope --pope-type adversarial

    # LLaVA on POPE (with steering)
    python scripts/evaluate_multi_model.py --model-type llava --benchmark pope --pope-type adversarial \\
        --use-steering --steering-path outputs/20260202_002953/vti_baseline/artifacts

    # Qwen3-VL on POPE
    python scripts/evaluate_multi_model.py --model-type qwen3-vl --benchmark pope --pope-type random

    # LLaVA on CHAIR (with custom steering strength)
    python scripts/evaluate_multi_model.py --model-type llava --benchmark chair --max-samples 500 \\
        --use-steering --steering-path outputs/.../artifacts --alpha-image 0.5 --alpha-text 0.8

    # Qwen3-VL on CHAIR
    python scripts/evaluate_multi_model.py --model-type qwen3-vl --benchmark chair --max-samples 500
"""

import argparse
import sys
import os
from pathlib import Path
import json
import torch
from tqdm import tqdm

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.models.unified_loader import load_model, get_device
from src.data.pope_loader import load_pope_dataset, evaluate_pope_predictions
from src.data.chair_loader import load_chair_dataset, evaluate_chair_predictions
from src.utils.config import load_config
from src.utils.logging_utils import setup_logging, create_experiment_dir, save_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Model Benchmark Evaluation")

    # Model selection
    parser.add_argument("--model-type", type=str, default="llava",
                       choices=['llava', 'qwen3-vl', 'qwen2-vl', 'qwen'],
                       help="Model architecture type")
    parser.add_argument("--model-name", type=str, default=None,
                       help="Model name or path (overrides config)")
    parser.add_argument("--config", type=str, default=None,
                       help="Config file (auto-selected based on model-type if not provided)")

    # Benchmark selection
    parser.add_argument("--benchmark", type=str, default="pope",
                       choices=['pope', 'chair', 'mmhal'],
                       help="Benchmark to evaluate on")

    # Benchmark-specific options
    parser.add_argument("--pope-type", type=str, default="random",
                       choices=['random', 'popular', 'adversarial'],
                       help="POPE evaluation type")
    parser.add_argument("--data-dir", type=str, default="./data/MSCOCO",
                       help="Path to COCO data directory")

    # Evaluation options
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Limit number of samples")
    parser.add_argument("--debug", action="store_true",
                       help="Debug mode")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size (currently only 1 supported)")

    # Steering options
    parser.add_argument("--use-steering", action="store_true",
                       help="Apply VTI steering to model")
    parser.add_argument("--steering-path", type=str, default=None,
                       help="Path to steering artifacts directory")
    parser.add_argument("--alpha-image", type=float, default=0.9,
                       help="Visual steering strength")
    parser.add_argument("--alpha-text", type=float, default=0.9,
                       help="Textual steering strength")

    return parser.parse_args()


def load_and_apply_steering(model, steering_path, alpha_image, alpha_text, device, logger):
    """Load steering artifacts and apply to model."""
    import torch
    from pathlib import Path

    artifacts_dir = Path(steering_path)
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Steering path not found: {steering_path}")

    logger.info(f"Loading steering from: {steering_path}")

    # Load steering directions
    visual_dir_path = artifacts_dir / "visual_direction.pt"
    textual_dir_path = artifacts_dir / "textual_direction.pt"

    visual_direction = None
    if visual_dir_path.exists() and alpha_image != 0:
        visual_direction = torch.load(visual_dir_path, map_location=device)
        if visual_direction.dim() == 3:
            visual_direction = torch.stack([visual_direction], dim=1)
        logger.info(f"Loaded visual steering direction (alpha={alpha_image})")

    textual_direction = None
    if textual_dir_path.exists() and alpha_text != 0:
        textual_direction = torch.load(textual_dir_path, map_location=device)
        if textual_direction.dim() == 2:
            textual_direction = torch.stack([textual_direction], dim=1)
        logger.info(f"Loaded textual steering direction (alpha={alpha_text})")

    # Apply steering using VTI layers
    from vti_utils.llm_layers import add_vti_layers

    base_model = model.model if hasattr(model, "model") else model

    # Apply visual steering
    if visual_direction is not None:
        try:
            if hasattr(base_model, "get_vision_tower"):
                vision_model = base_model.get_vision_tower()
                if isinstance(vision_model, (list, tuple)):
                    vision_model = vision_model[0] if vision_model else None
                if vision_model is not None and hasattr(vision_model, "vision_tower"):
                    vision_model = vision_model.vision_tower
                if vision_model is not None and hasattr(vision_model, "vision_model"):
                    vision_model = vision_model.vision_model
            else:
                vision_model = base_model.vision_tower.vision_tower.vision_model
        except AttributeError:
            try:
                vision_model = base_model.vision_model
            except AttributeError:
                logger.warning("Could not find vision model for visual steering")
                vision_model = None

        if vision_model is not None:
            add_vti_layers(vision_model, visual_direction.to(device), alpha=[alpha_image])
            logger.info("[OK] Visual steering applied")

    # Apply textual steering
    if textual_direction is not None:
        add_vti_layers(base_model, textual_direction.to(device), alpha=[alpha_text])
        logger.info("[OK] Textual steering applied")

    return model


def prepare_llava_inputs(processor, tokenizer, image, question, device):
    """Prepare inputs for LLaVA model."""
    from llava.mm_utils import tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX

    prompt = f"USER: <image>\n{question} ASSISTANT:"

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).to(device)

    if hasattr(image, 'unsqueeze'):
        image_tensor = image.unsqueeze(0).to(device)
    else:
        image_tensor = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        image_tensor = image_tensor.unsqueeze(0).to(device)

    if device != torch.device("cpu"):
        image_tensor = image_tensor.half()

    return input_ids, image_tensor


def prepare_qwen_inputs(processor, tokenizer, image, question, device):
    """Prepare inputs for Qwen3-VL model."""
    # Qwen uses a unified processor for both image and text
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]

    # Apply chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Process inputs
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True
    )

    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    return inputs


def prepare_llava_inputs_batched(processor, tokenizer, images, questions, device):
    """Prepare batched inputs for LLaVA model."""
    from llava.mm_utils import tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX

    batch_size = len(images)
    assert len(questions) == batch_size, "Number of images and questions must match"

    # Prepare prompts
    prompts = [f"USER: <image>\n{q} ASSISTANT:" for q in questions]

    # Tokenize all prompts
    input_ids_list = []
    for prompt in prompts:
        ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        )
        input_ids_list.append(ids)

    # Pad to same length
    max_len = max(ids.shape[0] for ids in input_ids_list)
    padded_input_ids = []
    attention_mask = []

    for ids in input_ids_list:
        padding_len = max_len - ids.shape[0]
        # Pad on the left (important for decoder-only models)
        padded_ids = torch.cat([
            torch.full((padding_len,), tokenizer.pad_token_id, dtype=ids.dtype),
            ids
        ])
        # Create attention mask (0 for padding, 1 for real tokens)
        mask = torch.cat([
            torch.zeros(padding_len, dtype=torch.long),
            torch.ones(ids.shape[0], dtype=torch.long)
        ])
        padded_input_ids.append(padded_ids)
        attention_mask.append(mask)

    # Stack into batch
    input_ids = torch.stack(padded_input_ids).to(device)
    attention_mask = torch.stack(attention_mask).to(device)

    # Process images
    image_tensors = []
    for image in images:
        if hasattr(image, 'unsqueeze'):
            img_tensor = image
        else:
            img_tensor = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        image_tensors.append(img_tensor)

    # Stack images into batch
    image_tensor = torch.stack(image_tensors).to(device)
    if device != torch.device("cpu"):
        image_tensor = image_tensor.half()

    return input_ids, image_tensor, attention_mask


def prepare_qwen_inputs_batched(processor, tokenizer, images, questions, device):
    """Prepare batched inputs for Qwen3-VL model."""
    batch_size = len(images)
    assert len(questions) == batch_size, "Number of images and questions must match"

    # Prepare messages for all samples
    all_messages = []
    for question in questions:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},  # Placeholder, actual image passed separately
                    {"type": "text", "text": question},
                ],
            }
        ]
        all_messages.append(messages)

    # Apply chat template to all
    texts = [
        processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in all_messages
    ]

    # Process inputs with batching
    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True
    )

    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    return inputs


def prepare_caption_prompt_llava(processor, tokenizer, image, device):
    """Prepare caption generation inputs for LLaVA."""
    from llava.mm_utils import tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX

    prompt = "USER: <image>\nPlease describe this image in detail. ASSISTANT:"

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).to(device)

    if hasattr(image, 'unsqueeze'):
        image_tensor = image.unsqueeze(0).to(device)
    else:
        image_tensor = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        image_tensor = image_tensor.unsqueeze(0).to(device)

    if device != torch.device("cpu"):
        image_tensor = image_tensor.half()

    return input_ids, image_tensor


def prepare_caption_prompt_qwen(processor, tokenizer, image, device):
    """Prepare caption generation inputs for Qwen3-VL."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Please describe this image in detail."},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs


def prepare_caption_prompt_llava_batched(processor, tokenizer, images, device):
    """Prepare batched caption generation inputs for LLaVA."""
    from llava.mm_utils import tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX

    batch_size = len(images)
    prompt = "USER: <image>\nPlease describe this image in detail. ASSISTANT:"

    # Tokenize the same prompt for all images
    input_ids_single = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    )

    # Since prompt is the same, just repeat for batch
    input_ids = input_ids_single.unsqueeze(0).repeat(batch_size, 1).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    # Process images
    image_tensors = []
    for image in images:
        if hasattr(image, 'unsqueeze'):
            img_tensor = image
        else:
            img_tensor = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        image_tensors.append(img_tensor)

    # Stack images into batch
    image_tensor = torch.stack(image_tensors).to(device)
    if device != torch.device("cpu"):
        image_tensor = image_tensor.half()

    return input_ids, image_tensor, attention_mask


def prepare_caption_prompt_qwen_batched(processor, tokenizer, images, device):
    """Prepare batched caption generation inputs for Qwen3-VL."""
    batch_size = len(images)

    # Prepare messages for all samples (same prompt for all)
    all_messages = []
    for _ in range(batch_size):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},  # Placeholder
                    {"type": "text", "text": "Please describe this image in detail."},
                ],
            }
        ]
        all_messages.append(messages)

    # Apply chat template
    texts = [
        processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in all_messages
    ]

    # Process inputs with batching
    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True
    )

    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    return inputs


def run_pope_evaluation(args, config, logger):
    """Run POPE evaluation."""

    # Load model
    logger.info("Loading model...")
    model_name = args.model_name or config['model']['name']
    model_type = config['model'].get('model_type', args.model_type)

    tokenizer, model, processor, _ = load_model(
        model_name=model_name,
        model_type=model_type,
        device=config['model']['device'],
        debug=args.debug
    )
    device = get_device(model)
    logger.info(f"[OK] Model loaded on {device}")

    # Ensure tokenizer has pad_token for batching
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token for batching")

    # Apply steering if requested
    if args.use_steering:
        if args.steering_path is None:
            raise ValueError("--steering-path required when --use-steering is set")
        model = load_and_apply_steering(
            model, args.steering_path, args.alpha_image, args.alpha_text, device, logger
        )

    # Load POPE dataset
    logger.info(f"Loading POPE {args.pope_type} dataset...")
    pope_dataset = load_pope_dataset(
        data_dir=args.data_dir,
        pope_type=args.pope_type,
        image_processor=None  # We'll process manually
    )

    if args.max_samples:
        pope_dataset.data = pope_dataset.data[:args.max_samples]

    logger.info(f"[OK] Loaded {len(pope_dataset)} samples")

    # Run inference with batching
    logger.info(f"Running inference with batch_size={args.batch_size}...")
    predictions = []

    # Process in batches for efficiency
    for batch_start in tqdm(range(0, len(pope_dataset), args.batch_size), desc="Evaluating"):
        batch_end = min(batch_start + args.batch_size, len(pope_dataset))

        if args.debug:
            # Mock predictions
            for idx in range(batch_start, batch_end):
                item = pope_dataset.data[idx]
                prediction = "Yes" if idx % 2 == 0 else "No"
                predictions.append({
                    'question': item['text'],
                    'label': 1 if item['label'] == 'yes' else 0,
                    'prediction': prediction
                })
        else:
            # Process batch
            batch_items = [pope_dataset[i] for i in range(batch_start, batch_end)]

            # Extract batch data
            images = [item['image'] for item in batch_items]
            questions = [item['question'] for item in batch_items]
            labels = [item['label'] for item in batch_items]

            # Prepare batched inputs based on model type
            if model.model_type == 'llava':
                input_ids, image_tensor, attention_mask = prepare_llava_inputs_batched(
                    processor, tokenizer, images, questions, device
                )

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        images=image_tensor,
                        max_new_tokens=32,
                        use_cache=True,  # Enable KV cache for speedup
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id
                    )

                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            elif model.model_type in ['qwen', 'qwen3-vl', 'qwen2-vl']:
                inputs = prepare_qwen_inputs_batched(processor, tokenizer, images, questions, device)

                with torch.inference_mode():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=32,
                        use_cache=True,  # Enable KV cache
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id
                    )

                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                # Remove the input prompt from outputs
                outputs = [
                    output.split(question)[-1] if question in output else output
                    for output, question in zip(outputs, questions)
                ]

            # Process outputs
            for question, label, output in zip(questions, labels, outputs):
                output = output.strip()
                predictions.append({
                    'question': question,
                    'label': label,
                    'prediction': output
                })

    # Evaluate
    logger.info("Computing metrics...")
    metrics = evaluate_pope_predictions(predictions)

    return metrics, predictions


def run_chair_evaluation(args, config, logger):
    """Run CHAIR evaluation."""

    # Load model
    logger.info("Loading model...")
    model_name = args.model_name or config['model']['name']
    model_type = config['model'].get('model_type', args.model_type)

    tokenizer, model, processor, _ = load_model(
        model_name=model_name,
        model_type=model_type,
        device=config['model']['device'],
        debug=args.debug
    )
    device = get_device(model)
    logger.info(f"[OK] Model loaded on {device}")

    # Ensure tokenizer has pad_token for batching
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token for batching")

    # Apply steering if requested
    if args.use_steering:
        if args.steering_path is None:
            raise ValueError("--steering-path required when --use-steering is set")
        model = load_and_apply_steering(
            model, args.steering_path, args.alpha_image, args.alpha_text, device, logger
        )

    # Load CHAIR dataset
    logger.info("Loading CHAIR dataset...")
    chair_dataset = load_chair_dataset(
        data_dir=args.data_dir,
        image_processor=None,  # We'll process manually
        split="val2014",
        max_samples=args.max_samples
    )

    logger.info(f"[OK] Loaded {len(chair_dataset)} samples")

    # Run inference
    logger.info(f"Generating captions with batch_size={args.batch_size}...")
    predictions = []

    # Process in batches
    for batch_start in tqdm(range(0, len(chair_dataset), args.batch_size), desc="Generating"):
        batch_end = min(batch_start + args.batch_size, len(chair_dataset))

        if args.debug:
            # Mock predictions
            for idx in range(batch_start, batch_end):
                item = chair_dataset[idx]
                predictions.append({
                    'image_id': item['image_id'],
                    'caption': "A dog and a cat sitting together",
                    'gt_objects': item['gt_objects']
                })
        else:
            # Process batch
            batch_items = [chair_dataset[i] for i in range(batch_start, batch_end)]

            # Extract batch data
            images = [item['image'] for item in batch_items]
            image_ids = [item['image_id'] for item in batch_items]
            gt_objects_list = [item['gt_objects'] for item in batch_items]

            # Prepare batched inputs based on model type
            if model.model_type == 'llava':
                input_ids, image_tensor, attention_mask = prepare_caption_prompt_llava_batched(
                    processor, tokenizer, images, device
                )

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        images=image_tensor,
                        max_new_tokens=512,
                        use_cache=True,  # Enable KV cache
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id
                    )

                captions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            elif model.model_type in ['qwen', 'qwen3-vl', 'qwen2-vl']:
                inputs = prepare_caption_prompt_qwen_batched(processor, tokenizer, images, device)

                with torch.inference_mode():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        use_cache=True,  # Enable KV cache
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id
                    )

                captions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                # Remove the input prompt from captions
                prompt_text = "Please describe this image in detail."
                captions = [
                    caption.split(prompt_text)[-1] if prompt_text in caption else caption
                    for caption in captions
                ]

            # Process captions
            for image_id, caption, gt_objects in zip(image_ids, captions, gt_objects_list):
                caption = caption.strip()
                predictions.append({
                    'image_id': image_id,
                    'caption': caption,
                    'gt_objects': gt_objects
                })

    # Evaluate
    logger.info("Computing CHAIR metrics...")
    ann_file = os.path.join(args.data_dir, "annotations", "instances_val2014.json")
    metrics = evaluate_chair_predictions(predictions, ann_file)

    return metrics, predictions


def main():
    args = parse_args()

    # Auto-select config if not provided
    if args.config is None:
        if args.model_type in ['qwen', 'qwen3-vl', 'qwen2-vl']:
            args.config = "configs/qwen3_vl.yaml"
        else:
            args.config = "configs/default.yaml"

    # Load config
    config = load_config(args.config)

    # Apply overrides
    if args.debug:
        config['experiment']['debug'] = True
    if args.max_samples:
        config['experiment']['max_samples'] = args.max_samples

    # Create experiment directory
    if args.benchmark == 'pope':
        exp_name = f"{args.model_type}_{args.benchmark}_{args.pope_type}"
    else:
        exp_name = f"{args.model_type}_{args.benchmark}"

    exp_dir = create_experiment_dir(
        base_dir="./outputs",
        experiment_name=exp_name
    )

    # Setup logging
    import logging
    logger = setup_logging(
        log_dir=exp_dir,
        log_level=logging.INFO,
        experiment_name=exp_name
    )

    logger.info("="*80)
    logger.info(f"Multi-Model Evaluation: {args.model_type.upper()}")
    logger.info("="*80)
    logger.info(f"Benchmark: {args.benchmark}")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Output: {exp_dir}")
    logger.info("")

    # Run evaluation
    if args.benchmark == "pope":
        metrics, predictions = run_pope_evaluation(args, config, logger)

        logger.info("\n" + "="*50)
        logger.info("POPE Results")
        logger.info("="*50)
        logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall:    {metrics['recall']:.4f}")
        logger.info(f"F1:        {metrics['f1']:.4f}")
        logger.info(f"Yes Ratio: {metrics['yes_ratio']:.4f}")
        logger.info("="*50)

        # Save results
        save_metrics(metrics, exp_dir / "metrics.json")

        with open(exp_dir / "predictions.jsonl", 'w') as f:
            for pred in predictions:
                f.write(json.dumps(pred) + '\n')

    elif args.benchmark == "chair":
        metrics, predictions = run_chair_evaluation(args, config, logger)

        logger.info("\n" + "="*50)
        logger.info("CHAIR Results")
        logger.info("="*50)
        logger.info(f"CHAIRs (Sentence): {metrics['CHAIRs']:.4f}")
        logger.info(f"CHAIRi (Instance): {metrics['CHAIRi']:.4f}")
        logger.info(f"Total Sentences: {metrics['total_sentences']}")
        logger.info(f"Sentences with Hallucination: {metrics['sentences_with_hallucination']}")
        logger.info(f"Total Objects Mentioned: {metrics['total_objects_mentioned']}")
        logger.info(f"Total Hallucinated Objects: {metrics['total_hallucinated_objects']}")
        logger.info("="*50)

        # Save results
        save_metrics(metrics, exp_dir / "metrics.json")

        with open(exp_dir / "captions.jsonl", 'w') as f:
            for pred in predictions:
                f.write(json.dumps(pred) + '\n')

    else:
        logger.error(f"Benchmark {args.benchmark} not yet implemented")
        return

    logger.info(f"\n[OK] Results saved to: {exp_dir}")


if __name__ == "__main__":
    main()
