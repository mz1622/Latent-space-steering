"""
Base abstraction for hallucination detection methods.

This is a placeholder for future hallucination detection work.
VTI always steers, so detection is not part of the current implementation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import torch


class HallucinationDetector(ABC):
    """
    Abstract base class for hallucination detection methods.

    Future implementations might include:
    - Uncertainty-based detection
    - Attention-based detection
    - Contrastive detection
    - Probe-based detection
    """

    def __init__(self, config: Dict[str, Any], debug: bool = False):
        """
        Initialize detector.

        Args:
            config: Configuration dictionary
            debug: If True, skip expensive computations
        """
        self.config = config
        self.debug = debug
        self.fitted = False

    @abstractmethod
    def fit(
        self,
        model: Any,
        train_data: Any,
        max_samples: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train/calibrate the detector.

        Args:
            model: LVLM model
            train_data: Training data
            max_samples: Maximum samples to use
            **kwargs: Method-specific arguments

        Returns:
            artifacts: Trained detector artifacts
        """
        pass

    @abstractmethod
    def detect(
        self,
        model: Any,
        inputs: Any,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Detect hallucinations in model outputs.

        Args:
            model: LVLM model
            inputs: Input data (images, prompts)
            **kwargs: Detection parameters

        Returns:
            detections: List of detection results with confidence scores
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(fitted={self.fitted}, debug={self.debug})"


class PassthroughDetector(HallucinationDetector):
    """
    Placeholder detector that always indicates "no hallucination detected".

    This is used as a default when no real detection is needed (e.g., VTI
    which always applies steering regardless of detection).
    """

    def fit(
        self,
        model: Any,
        train_data: Any,
        max_samples: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        No-op fit for passthrough detector.
        """
        self.fitted = True
        return {}

    def detect(
        self,
        model: Any,
        inputs: Any,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Always return "no hallucination" for all inputs.
        """
        num_inputs = len(inputs) if isinstance(inputs, list) else 1
        return [
            {
                'hallucination_detected': False,
                'confidence': 1.0,
                'method': 'passthrough'
            }
            for _ in range(num_inputs)
        ]
