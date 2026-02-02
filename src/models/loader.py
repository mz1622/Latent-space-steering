"""
Model loading utilities.
"""

import torch
from typing import Callable
from typing import Tuple, Any, Optional
import sys
import os

# Add LLaVA to path (we're already in Latent-space-steering)
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))


def load_model(
    model_name: str = "liuhaotian/llava-v1.5-7b",
    model_base: Optional[str] = None,
    device: str = "auto",
    debug: bool = False
) -> Tuple[Any, Any, Any, int]:
    """
    Load LVLM model.

    Args:
        model_name: Model name or path
        model_base: Base model name (if using LoRA)
        device: Device to load on (auto, cuda, mps, cpu)
        debug: If True, may use smaller/mock model

    Returns:
        tokenizer: Model tokenizer
        model: Loaded model
        image_processor: Image processor
        context_len: Context length
    """
    if debug:
        print("[DEBUG MODE] Loading model in debug mode...")

    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Loading model {model_name} on {device}...")

    # Load LLaVA model
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path

    model_path = os.path.expanduser(model_name)
    model_name_short = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name_short, device=device
    )

    # Ensure model is in eval mode
    model.eval()

    # Fix dtype inconsistency in LLaVA
    # The vision tower may be in float32 while the rest uses the model dtype.
    # On CPU, force float32 to avoid matmul dtype mismatches.
    if device == "cpu":
        model = model.to(dtype=torch.float32)
        model_dtype = torch.float32
    else:
        model_dtype = next(model.parameters()).dtype

    if device != "cpu":

        # Prefer LLaVA's accessor if available.
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

        # Align projector dtype as well.
        projector = None
        if hasattr(model, "get_model"):
            core_model = model.get_model()
            if hasattr(core_model, "mm_projector"):
                projector = core_model.mm_projector
        elif hasattr(model, "mm_projector"):
            projector = model.mm_projector

        if projector is not None:
            projector.to(dtype=model_dtype)

            # Guard against stray float32 features by casting inputs to projector dtype.
            if not getattr(projector, "_dtype_guarded", False):
                orig_forward: Callable = projector.forward

                def _forward_with_cast(x, *args, **kwargs):
                    if isinstance(x, torch.Tensor):
                        target_dtype = next(projector.parameters()).dtype
                        if x.dtype != target_dtype:
                            x = x.to(dtype=target_dtype)
                    return orig_forward(x, *args, **kwargs)

                projector.forward = _forward_with_cast
                projector._dtype_guarded = True

        print(f"Model loaded successfully (dtype: {model_dtype})")
    else:
        # Align vision tower/projector dtype on CPU too.
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

        projector = None
        if hasattr(model, "get_model"):
            core_model = model.get_model()
            if hasattr(core_model, "mm_projector"):
                projector = core_model.mm_projector
        elif hasattr(model, "mm_projector"):
            projector = model.mm_projector

        if projector is not None:
            projector.to(dtype=model_dtype)

        print(f"Model loaded successfully (dtype: {model_dtype})")

    return tokenizer, model, image_processor, context_len


def get_device(model: Any) -> torch.device:
    """
    Get device of model.

    Args:
        model: PyTorch model

    Returns:
        device: Device
    """
    return next(model.parameters()).device
