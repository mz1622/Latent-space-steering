#!/usr/bin/env python
"""
Simple test script to verify LLaVA works correctly.
Tests basic inference to diagnose dtype issues.
"""

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.models.loader import load_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from PIL import Image
import requests
from io import BytesIO


def main():
    print("="*80)
    print("LLaVA Simple Test")
    print("="*80)

    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Device: {device}")

    # Load model
    model_name = "liuhaotian/llava-v1.5-7b"
    print(f"Loading {model_name}...")

    tokenizer, model, image_processor, _ = load_model(
        model_name=model_name,
        model_base=None,
        device=device,
        debug=False
    )

    # Print model info
    model_dtype = next(model.parameters()).dtype
    print(f"Model dtype: {model_dtype}")
    print(f"Model device: {next(model.parameters()).device}")

    # Load test image
    print("\nLoading test image...")
    image_url = "https://llava-vl.github.io/static/images/view.jpg"
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert('RGB')

    # Prepare prompt
    question = "What is in this image?"

    # Use conversation template
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], f"<image>\n{question}")
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    print(f"Prompt: {prompt}")

    # Tokenize
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).to(device)

    print(f"Input IDs shape: {input_ids.shape}, dtype: {input_ids.dtype}")

    # Process image
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    image_tensor = image_tensor.unsqueeze(0).to(device)

    print(f"Image tensor shape: {image_tensor.shape}, dtype: {image_tensor.dtype}")

    # Convert to model dtype if needed
    if device != "cpu" and image_tensor.dtype != model_dtype:
        print(f"Converting image to {model_dtype}...")
        image_tensor = image_tensor.to(dtype=model_dtype)
        print(f"Image tensor dtype after conversion: {image_tensor.dtype}")

    # Generate
    print("\nGenerating response...")
    try:
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                max_new_tokens=100,
                use_cache=False
            )

        output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        print("\n" + "="*80)
        print("SUCCESS!")
        print("="*80)
        print(f"Question: {question}")
        print(f"Answer: {output}")
        print("="*80)

    except Exception as e:
        print("\n" + "="*80)
        print("ERROR!")
        print("="*80)
        print(f"Error: {e}")
        print("="*80)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
