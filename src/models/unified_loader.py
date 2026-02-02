"""
Unified model loader supporting multiple LVLM architectures.

Supported models:
- LLaVA (liuhaotian/llava-v1.5-7b, etc.)
- Qwen3-VL (Qwen/Qwen3-VL-*, etc.)
"""

import torch
from typing import Tuple, Any, Optional, Dict
import sys
import os
import importlib.util


class ModelWrapper:
    """Wrapper to provide unified interface for different model architectures."""

    def __init__(self, model, processor, tokenizer, model_type: str):
        self.model = model
        self.processor = processor  # Can be image_processor or unified processor
        self.tokenizer = tokenizer
        self.model_type = model_type  # 'llava' or 'qwen'

    def generate(self, *args, **kwargs):
        """Unified generate interface."""
        return self.model.generate(*args, **kwargs)

    def parameters(self):
        """Access model parameters."""
        return self.model.parameters()

    def eval(self):
        """Set model to eval mode."""
        return self.model.eval()

    def to(self, *args, **kwargs):
        """Move model to device."""
        return self.model.to(*args, **kwargs)

    def __getattr__(self, name):
        """Delegate attribute access to underlying model."""
        return getattr(self.model, name)


def load_llava_model(
    model_name: str,
    model_base: Optional[str] = None,
    device: str = "auto",
    debug: bool = False
) -> Tuple[Any, Any, Any, int]:
    """
    Load LLaVA model.

    Returns:
        tokenizer, model, image_processor, context_len
    """
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path

    if debug:
        print("[DEBUG MODE] Loading LLaVA in debug mode...")

    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Loading LLaVA model {model_name} on {device}...")

    model_path = os.path.expanduser(model_name)
    model_name_short = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name_short, device=device
    )

    # Ensure model is in eval mode
    model.eval()

    # Fix dtype inconsistency
    if device == "cpu":
        model = model.to(dtype=torch.float32)
        model_dtype = torch.float32
    else:
        model_dtype = next(model.parameters()).dtype

    # Align vision tower and projector dtype
    if device != "cpu":
        vision_tower = None
        if hasattr(model, "get_vision_tower"):
            vision_tower = model.get_vision_tower()
            if isinstance(vision_tower, (list, tuple)):
                vision_tower = vision_tower[0] if vision_tower else None
        elif hasattr(model, "model") and hasattr(model.model, "vision_tower"):
            vision_tower = model.model.vision_tower
            if hasattr(vision_tower, "vision_tower"):
                vision_tower = vision_tower.vision_tower

        if vision_tower is not None:
            vision_tower.to(dtype=model_dtype)
            print(f"Converted vision tower to {model_dtype}")

        projector = None
        if hasattr(model, "get_model"):
            core_model = model.get_model()
            if hasattr(core_model, "mm_projector"):
                projector = core_model.mm_projector
        elif hasattr(model, "mm_projector"):
            projector = model.mm_projector

        if projector is not None:
            projector.to(dtype=model_dtype)

            # Guard against stray float32 features
            if not getattr(projector, "_dtype_guarded", False):
                orig_forward = projector.forward

                def _forward_with_cast(x, *args, **kwargs):
                    if isinstance(x, torch.Tensor):
                        target_dtype = next(projector.parameters()).dtype
                        if x.dtype != target_dtype:
                            x = x.to(dtype=target_dtype)
                    return orig_forward(x, *args, **kwargs)

                projector.forward = _forward_with_cast
                projector._dtype_guarded = True

    print(f"[OK] LLaVA model loaded successfully (dtype: {model_dtype})")

    return tokenizer, model, image_processor, context_len


def load_qwen_model(
    model_name: str,
    device: str = "auto",
    debug: bool = False
) -> Tuple[Any, Any, Any, int]:
    """
    Load Qwen3-VL model.

    Returns:
        tokenizer, model, processor, context_len
    """
    from transformers import AutoModelForImageTextToText, AutoTokenizer, AutoProcessor

    if debug:
        print("[DEBUG MODE] Loading Qwen3-VL in debug mode...")

    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Loading Qwen3-VL model {model_name} on {device}...")

    # Load model with appropriate dtype
    # Qwen3-VL uses AutoModelForImageTextToText (formerly AutoModelForVision2Seq)
    if device == "cuda":
        has_accelerate = importlib.util.find_spec("accelerate") is not None
        if has_accelerate:
            try:
                model = AutoModelForImageTextToText.from_pretrained(
                    model_name,
                    dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            except (ImportError, ValueError) as exc:
                # Fallback if accelerate is not actually usable
                print(f"Warning: accelerate/device_map unavailable ({exc}), loading without device_map")
                model = AutoModelForImageTextToText.from_pretrained(
                    model_name,
                    dtype=torch.float16,
                    trust_remote_code=True
                )
                model = model.to(device)
        else:
            # Fallback if accelerate is not installed
            print("Warning: accelerate not installed, loading without device_map")
            model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                dtype=torch.float16,
                trust_remote_code=True
            )
            model = model.to(device)
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            dtype=torch.float32,
            trust_remote_code=True
        )
        model = model.to(device)

    # Load processor (handles both images and text)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model.eval()

    # Qwen3-VL typically has 32K context
    context_len = getattr(model.config, "max_position_embeddings", 32768)

    print(f"[OK] Qwen3-VL model loaded successfully (dtype: {model.dtype})")

    return tokenizer, model, processor, context_len


def load_model(
    model_name: str = "liuhaotian/llava-v1.5-7b",
    model_base: Optional[str] = None,
    model_type: Optional[str] = None,
    device: str = "auto",
    debug: bool = False
) -> Tuple[Any, ModelWrapper, Any, int]:
    """
    Unified model loader supporting multiple architectures.

    Args:
        model_name: Model name or path
        model_base: Base model name (for LLaVA LoRA)
        model_type: Model type ('llava', 'qwen3-vl', or None for auto-detect)
        device: Device to load on (auto, cuda, mps, cpu)
        debug: If True, may use smaller/mock model

    Returns:
        tokenizer: Model tokenizer
        model: Loaded model (wrapped)
        processor: Image processor or unified processor
        context_len: Context length
    """
    # Auto-detect model type from name if not specified
    if model_type is None:
        model_name_lower = model_name.lower()
        if 'qwen' in model_name_lower:
            model_type = 'qwen3-vl'
        elif 'llava' in model_name_lower:
            model_type = 'llava'
        else:
            # Default to LLaVA for backward compatibility
            print(f"Warning: Could not auto-detect model type from '{model_name}', defaulting to LLaVA")
            model_type = 'llava'

    print(f"Detected model type: {model_type}")

    # Load appropriate model
    if model_type == 'llava':
        tokenizer, model, processor, context_len = load_llava_model(
            model_name, model_base, device, debug
        )
    elif model_type in ['qwen', 'qwen3-vl', 'qwen2-vl']:
        tokenizer, model, processor, context_len = load_qwen_model(
            model_name, device, debug
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Wrap model for unified interface
    wrapped_model = ModelWrapper(model, processor, tokenizer, model_type)

    return tokenizer, wrapped_model, processor, context_len


def get_device(model: Any) -> torch.device:
    """Get device of model."""
    if isinstance(model, ModelWrapper):
        return next(model.model.parameters()).device
    return next(model.parameters()).device
