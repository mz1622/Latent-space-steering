"""
Mean Difference Steering Method (VTI-like approach).

This implements the Visual and Textual Intervention (VTI) method from:
"Reducing Hallucinations in Vision-Language Models via Latent Space Steering"
Liu et al., 2024

The method computes steering directions based on mean differences between
hallucinated and non-hallucinated hidden states, then applies PCA to extract
the principal direction of variation.
"""

from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
from pathlib import Path
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from .base import SteeringMethod


class MeanDifferenceSteeringMethod(SteeringMethod):
    """
    Steering method based on mean differences + PCA.

    This is the VTI approach: compute mean difference between hidden states
    of hallucinated vs. non-hallucinated outputs, apply PCA to find principal
    direction, then steer along that direction during inference.
    """

    def __init__(self, config: Dict[str, Any], debug: bool = False):
        """
        Initialize mean difference steering method.

        Args:
            config: Configuration dictionary with keys:
                - alpha_image: Steering strength for vision encoder
                - alpha_text: Steering strength for text decoder
                - rank: Number of PCA components (default: 1)
                - mask_ratio: Ratio of image patches to mask (default: 0.99)
                - num_trials: Number of masking trials per image (default: 50)
            debug: If True, use mock computations for testing
        """
        super().__init__(config, debug)
        self.alpha_image = config.get('alpha_image', 0.0)
        self.alpha_text = config.get('alpha_text', 0.8)
        self.rank = config.get('rank', 1)
        self.mask_ratio = config.get('mask_ratio', 0.99)
        self.num_trials = config.get('num_trials', 50)

    def fit(
        self,
        model: Any,
        train_data: Any,
        max_samples: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compute VTI visual and textual steering directions.

        Args:
            model: LVLM model (e.g., LLaVA)
            train_data: Tuple of (input_images, input_ids) from get_demos()
            max_samples: Limit number of training samples (for debugging)
            **kwargs: Additional arguments

        Returns:
            artifacts: Dict with 'visual_direction' and 'textual_direction' tensors
        """
        if self.debug:
            # Mock directions for debugging
            print("[DEBUG MODE] Creating mock steering directions")
            # Get model architecture info
            try:
                from vti_utils.llm_layers import get_layers
                layers = get_layers(model)
                num_layers = len(layers)
                hidden_dim = layers[0].mlp[0].out_features if hasattr(layers[0], 'mlp') else 4096
            except:
                num_layers = 32
                hidden_dim = 4096

            # Create mock directions
            visual_direction = torch.randn(num_layers, 576, hidden_dim) * 0.01
            textual_direction = torch.randn(num_layers, hidden_dim) * 0.01

            self.artifacts = {
                'visual_direction': visual_direction,
                'textual_direction': textual_direction,
                'num_layers': num_layers,
                'hidden_dim': hidden_dim,
                'debug': True
            }
            self.fitted = True
            return self.artifacts

        # Real computation
        input_images, input_ids = train_data

        # Limit samples if specified
        if max_samples is not None:
            input_images = input_images[:max_samples]
            input_ids = input_ids[:max_samples]

        print(f"Computing steering directions from {len(input_images)} samples...")

        # Compute visual direction
        visual_direction = None
        if self.alpha_image != 0:
            from vti_utils.utils import obtain_visual_vti
            device = next(model.parameters()).device
            vti_vision, _ = obtain_visual_vti(
                model, input_images, rank=self.rank, device=str(device)
            )
            visual_direction = vti_vision[1:]  # Skip first layer

        # Compute textual direction
        textual_direction = None
        if self.alpha_text != 0:
            from vti_utils.utils import obtain_textual_vti
            device = next(model.parameters()).device
            vti_text, _ = obtain_textual_vti(
                model, input_ids, input_images, rank=self.rank, device=str(device)
            )
            textual_direction = vti_text[1:]  # Skip first layer

        self.artifacts = {
            'visual_direction': visual_direction,
            'textual_direction': textual_direction,
        }
        self.fitted = True

        return self.artifacts

    def apply(self, model: Any, **kwargs) -> Any:
        """
        Apply VTI steering to model by adding VTI layers.

        Args:
            model: LVLM model to modify
            **kwargs: Optional alpha overrides

        Returns:
            model: Modified model with VTI layers added
        """
        if not self.fitted:
            raise ValueError("Method must be fitted before applying steering")

        from vti_utils.llm_layers import add_vti_layers

        device = next(model.parameters()).device
        alpha_image = kwargs.get('alpha_image', self.alpha_image)
        alpha_text = kwargs.get('alpha_text', self.alpha_text)

        # Add visual steering
        if self.artifacts['visual_direction'] is not None and alpha_image != 0:
            visual_direction = self.artifacts['visual_direction']
            if visual_direction.dim() == 3:
                visual_direction = torch.stack([visual_direction], dim=1)
            try:
                vision_model = model.model.vision_tower.vision_tower.vision_model
            except AttributeError:
                vision_model = model.vision_model

            add_vti_layers(
                vision_model,
                visual_direction.to(device),
                alpha=[alpha_image]
            )
            print(f"[OK] Visual steering applied (alpha={alpha_image})")

        # Add textual steering
        if self.artifacts['textual_direction'] is not None and alpha_text != 0:
            textual_direction = self.artifacts['textual_direction']
            if textual_direction.dim() == 2:
                textual_direction = torch.stack([textual_direction], dim=1)
            add_vti_layers(
                model,
                textual_direction.to(device),
                alpha=[alpha_text]
            )
            print(f"[OK] Textual steering applied (alpha={alpha_text})")

        return model

    def remove(self, model: Any) -> Any:
        """
        Remove VTI steering from model.

        Args:
            model: Model with VTI layers

        Returns:
            model: Model with VTI layers removed
        """
        from vti_utils.llm_layers import remove_vti_layers

        # Remove from vision model
        try:
            vision_model = model.model.vision_tower.vision_tower.vision_model
            remove_vti_layers(vision_model)
        except AttributeError:
            pass

        # Remove from text model
        remove_vti_layers(model)

        print("[OK] Steering removed")
        return model

    def infer(
        self,
        model: Any,
        inputs: Any,
        **kwargs
    ) -> List[str]:
        """
        Run generation with VTI steering applied.

        Args:
            model: Model with steering already applied via apply()
            inputs: List of (input_ids, image_tensor, tokenizer) tuples
            **kwargs: Generation parameters (num_beams, max_new_tokens, etc.)

        Returns:
            outputs: List of generated text strings
        """
        if self.debug:
            print("[DEBUG MODE] Skipping actual generation")
            return ["[DEBUG] Mock generated output"] * len(inputs)

        outputs = []
        num_beams = kwargs.get('num_beams', 5)
        max_new_tokens = kwargs.get('max_new_tokens', 256)
        do_sample = kwargs.get('do_sample', False)

        device = next(model.parameters()).device

        for input_ids, image_tensor, tokenizer in inputs:
            with torch.inference_mode():
                if image_tensor is not None:
                    image_input = image_tensor.unsqueeze(0).to(device)
                    if device != torch.device("cpu"):
                        image_input = image_input.half()

                    output_ids = model.generate(
                        input_ids,
                        images=image_input,
                        num_beams=num_beams,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        use_cache=False
                    )
                else:
                    output_ids = model.generate(
                        input_ids,
                        num_beams=num_beams,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        use_cache=False
                    )

            output_text = tokenizer.batch_decode(output_ids[:, :], skip_special_tokens=True)[0]
            output_text = output_text.strip()
            outputs.append(output_text)

        return outputs

    def evaluate(
        self,
        predictions: List[Dict[str, Any]],
        metrics: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics.

        Args:
            predictions: List of dicts with 'prediction', 'ground_truth', etc.
            metrics: List of metric names (e.g., ['accuracy', 'f1'])
            **kwargs: Metric-specific args (e.g., API key for GPT eval)

        Returns:
            results: Dict of metric_name -> score
        """
        if self.debug:
            print("[DEBUG MODE] Returning mock metrics")
            return {
                'accuracy': 0.85,
                'hallucination_rate': 0.15,
                'num_samples': len(predictions)
            }

        # Placeholder for actual metrics computation
        # Will be implemented in evaluation harness
        results = {
            'num_samples': len(predictions)
        }

        # Add custom metrics based on what's requested
        if metrics:
            for metric in metrics:
                results[metric] = 0.0  # Placeholder

        return results
