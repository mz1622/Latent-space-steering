"""
Base abstraction for all steering methods.

All steering methods must implement this interface to ensure consistency
across different approaches (VTI, TruthPrInt, AutoSteer, SEA, etc.).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import torch
from pathlib import Path
import json


class SteeringMethod(ABC):
    """
    Abstract base class for all latent space steering methods.

    Methods must implement:
    - fit(): Compute steering directions from training data
    - apply(): Apply steering to a model via hooks/modifications
    - infer(): Run generation with steering applied
    - evaluate(): Compute metrics on results
    """

    def __init__(self, config: Dict[str, Any], debug: bool = False):
        """
        Initialize steering method.

        Args:
            config: Configuration dictionary for the method
            debug: If True, skip expensive computations for testing
        """
        self.config = config
        self.debug = debug
        self.fitted = False
        self.artifacts = {}

    @abstractmethod
    def fit(
        self,
        model: Any,
        train_data: Any,
        max_samples: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compute steering directions from training data.

        Args:
            model: The LVLM model to compute directions for
            train_data: Training data (format depends on method)
            max_samples: Maximum number of samples to use (for debugging)
            **kwargs: Method-specific arguments

        Returns:
            artifacts: Dictionary containing computed directions and metadata
                      (e.g., {'visual_direction': tensor, 'textual_direction': tensor, 'layer_info': dict})
        """
        pass

    @abstractmethod
    def apply(self, model: Any, **kwargs) -> Any:
        """
        Apply steering to model via hooks or direct modification.

        This should modify the model in-place or return a context manager
        that applies steering during inference.

        Args:
            model: The LVLM model to apply steering to
            **kwargs: Method-specific arguments (e.g., alpha, layers)

        Returns:
            model or context manager with steering applied
        """
        pass

    @abstractmethod
    def infer(
        self,
        model: Any,
        inputs: Any,
        **kwargs
    ) -> List[str]:
        """
        Run generation with steering applied.

        Args:
            model: The LVLM model (with steering applied)
            inputs: Input data (images, prompts, etc.)
            **kwargs: Generation parameters

        Returns:
            generations: List of generated text outputs
        """
        pass

    @abstractmethod
    def evaluate(
        self,
        predictions: List[Dict[str, Any]],
        metrics: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics on predictions.

        Args:
            predictions: List of prediction dicts with keys like:
                        {'question', 'prediction', 'ground_truth', etc.}
            metrics: List of metric names to compute
            **kwargs: Metric-specific arguments

        Returns:
            results: Dictionary of metric_name -> score
        """
        pass

    def save_artifacts(self, save_dir: Path) -> None:
        """
        Save computed artifacts (directions, metadata) to disk.

        Args:
            save_dir: Directory to save artifacts to
        """
        if not self.fitted:
            raise ValueError("Method must be fitted before saving artifacts")

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save tensors
        for key, value in self.artifacts.items():
            if isinstance(value, torch.Tensor):
                torch.save(value, save_dir / f"{key}.pt")
            elif isinstance(value, (dict, list)):
                with open(save_dir / f"{key}.json", "w") as f:
                    json.dump(value, f, indent=2)

        # Save config
        with open(save_dir / "config.json", "w") as f:
            json.dump(self.config, f, indent=2)

    def load_artifacts(self, load_dir: Path) -> None:
        """
        Load previously computed artifacts from disk.

        Args:
            load_dir: Directory to load artifacts from
        """
        load_dir = Path(load_dir)

        # Load tensors
        for pt_file in load_dir.glob("*.pt"):
            key = pt_file.stem
            self.artifacts[key] = torch.load(pt_file)

        # Load JSON artifacts
        for json_file in load_dir.glob("*.json"):
            if json_file.name != "config.json":
                key = json_file.stem
                with open(json_file) as f:
                    self.artifacts[key] = json.load(f)

        self.fitted = True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(fitted={self.fitted}, debug={self.debug})"
