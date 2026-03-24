"""LLaVA model loading utilities for the SAE package."""

from __future__ import annotations

import os
import sys
from typing import Any
from typing import Optional
from typing import Tuple

import torch


def _setup_import_paths() -> None:
    """Add local project roots so the bundled LLaVA package can be imported."""
    this_dir = os.path.dirname(__file__)
    sae_root = os.path.abspath(os.path.join(this_dir, "..", ".."))
    repo_root = os.path.abspath(os.path.join(this_dir, "..", "..", ".."))
    hulluedit_root = os.path.join(repo_root, "SAE", "HulluEdit")

    for path in (repo_root, sae_root, hulluedit_root):
        if path not in sys.path:
            sys.path.insert(0, path)


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_vision_tower(model: Any) -> Any | None:
    if hasattr(model, "get_vision_tower"):
        vision_tower = model.get_vision_tower()
        if isinstance(vision_tower, (list, tuple)):
            return vision_tower[0] if vision_tower else None
        return vision_tower

    if hasattr(model, "model") and hasattr(model.model, "vision_tower"):
        vision_tower = model.model.vision_tower
        if hasattr(vision_tower, "vision_tower"):
            return vision_tower.vision_tower
        return vision_tower

    return None


def _get_mm_projector(model: Any) -> Any | None:
    if hasattr(model, "get_model"):
        core_model = model.get_model()
        if hasattr(core_model, "mm_projector"):
            return core_model.mm_projector

    if hasattr(model, "mm_projector"):
        return model.mm_projector

    return None


def _align_dtype(model: Any, *, device: str) -> torch.dtype:
    if device == "cpu":
        model.to(dtype=torch.float32)
        model_dtype = torch.float32
    else:
        model_dtype = next(model.parameters()).dtype

    vision_tower = _get_vision_tower(model)
    if vision_tower is not None:
        vision_tower.to(dtype=model_dtype)

    projector = _get_mm_projector(model)
    if projector is None:
        return model_dtype

    projector.to(dtype=model_dtype)

    if getattr(projector, "_dtype_guarded", False):
        return model_dtype

    orig_forward = projector.forward

    def _forward_with_cast(x, *args, **kwargs):
        if isinstance(x, torch.Tensor):
            target_dtype = next(projector.parameters()).dtype
            if x.dtype != target_dtype:
                x = x.to(dtype=target_dtype)
        return orig_forward(x, *args, **kwargs)

    projector.forward = _forward_with_cast
    projector._dtype_guarded = True
    return model_dtype


def load_llava_model(
    model_name: str = "liuhaotian/llava-v1.5-7b",
    model_base: Optional[str] = None,
    device: str = "auto",
    debug: bool = False,
) -> Tuple[Any, Any, Any, int]:
    """Load a LLaVA model and return tokenizer, model, image processor, context length."""
    _setup_import_paths()

    if debug:
        print("[DEBUG MODE] Loading LLaVA model...")

    resolved_device = _resolve_device(device)
    print(f"Loading LLaVA model {model_name} on {resolved_device}...")

    from llava.mm_utils import get_model_name_from_path
    from llava.model.builder import load_pretrained_model

    model_path = os.path.expanduser(model_name)
    model_name_short = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        model_base,
        model_name_short,
        device=resolved_device,
    )
    model.eval()

    model_dtype = _align_dtype(model, device=resolved_device)
    print(f"LLaVA model loaded successfully (dtype: {model_dtype})")

    return tokenizer, model, image_processor, context_len


def get_device(model: Any) -> torch.device:
    """Get the device of a loaded PyTorch model."""
    return next(model.parameters()).device
