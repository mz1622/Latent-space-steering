"""
CHAIR (Caption Hallucination Assessment with Image Relevance) dataset loader and evaluation.

CHAIR evaluates object hallucination in image captioning by checking if mentioned
objects are actually present in the image.

Reference: Rohrbach et al., "Object Hallucination in Image Captioning", EMNLP 2018
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from PIL import Image
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO


class CHAIRDataset(Dataset):
    """
    CHAIR dataset for caption hallucination evaluation.

    Uses COCO validation images and generates captions to evaluate hallucination.
    """

    def __init__(
        self,
        annotation_file: str,
        image_dir: str,
        transform: Optional[Any] = None,
        max_samples: Optional[int] = None
    ):
        """
        Initialize CHAIR dataset.

        Args:
            annotation_file: Path to COCO annotations JSON
            image_dir: Directory containing COCO images
            transform: Optional image transformation
            max_samples: Limit number of samples (for testing)
        """
        self.image_dir = Path(image_dir)
        self.transform = transform

        # Load COCO annotations
        self.coco = COCO(annotation_file)

        # Get image IDs
        self.image_ids = list(self.coco.imgs.keys())
        if max_samples:
            self.image_ids = self.image_ids[:max_samples]

        print(f"Loaded {len(self.image_ids)} images for CHAIR evaluation")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.imgs[image_id]

        # Load image
        image_path = self.image_dir / image_info['file_name']
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Get ground truth objects
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        gt_objects = set([self.coco.cats[ann['category_id']]['name'] for ann in anns])

        return {
            'image': image,
            'image_id': image_id,
            'gt_objects': list(gt_objects),
            'file_name': image_info['file_name']
        }


def load_chair_dataset(
    data_dir: str,
    image_processor: Optional[Any] = None,
    split: str = "val2014",
    max_samples: Optional[int] = None
) -> CHAIRDataset:
    """
    Load CHAIR dataset.

    Args:
        data_dir: Base directory containing COCO data
        image_processor: Image processor/transform
        split: COCO split (val2014 or test2014)
        max_samples: Limit number of samples

    Returns:
        CHAIRDataset instance
    """
    # Annotation file path
    if split == "val2014":
        ann_file = os.path.join(data_dir, "annotations", "instances_val2014.json")
    else:
        ann_file = os.path.join(data_dir, "annotations", f"instances_{split}.json")

    # Image directory
    image_dir = os.path.join(data_dir, split)

    if not os.path.exists(ann_file):
        raise FileNotFoundError(
            f"COCO annotation file not found: {ann_file}\n"
            f"Please download COCO annotations"
        )

    if not os.path.exists(image_dir):
        raise FileNotFoundError(
            f"Image directory not found: {image_dir}\n"
            f"Please download COCO {split} images"
        )

    return CHAIRDataset(
        annotation_file=ann_file,
        image_dir=image_dir,
        transform=image_processor,
        max_samples=max_samples
    )


class CHAIREvaluator:
    """
    CHAIR metric evaluator.

    Computes CHAIRs (sentence-level) and CHAIRi (instance-level) metrics.
    """

    def __init__(self, coco_annotation_file: str):
        """
        Initialize CHAIR evaluator.

        Args:
            coco_annotation_file: Path to COCO instances JSON
        """
        self.coco = COCO(coco_annotation_file)

        # Build synonyms dict (simplified - full version would use external synonyms)
        self.synonyms = self._build_synonyms()

    def _build_synonyms(self) -> Dict[str, Set[str]]:
        """Build synonym mapping for COCO objects."""
        # Simplified synonym mapping
        # In practice, use WordNet or external synonym database
        synonyms = {}
        for cat_id, cat_info in self.coco.cats.items():
            name = cat_info['name']
            synonyms[name] = {name}

        return synonyms

    def _extract_objects(self, caption: str) -> List[str]:
        """
        Extract object mentions from caption.

        Simplified version - in practice, use POS tagging and noun extraction.
        """
        # Simplified: split caption and filter common words
        common_words = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'of', 'and', 'or', 'is', 'are'}
        words = caption.lower().replace('.', '').replace(',', '').split()
        objects = [w for w in words if w not in common_words and len(w) > 2]
        return objects

    def evaluate(
        self,
        predictions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Evaluate CHAIR metrics.

        Args:
            predictions: List of dicts with 'caption', 'image_id', 'gt_objects'

        Returns:
            metrics: Dict with CHAIRs, CHAIRi, and other statistics
        """
        total_sentences = len(predictions)
        sentences_with_hallucination = 0
        total_objects_mentioned = 0
        total_hallucinated_objects = 0

        for item in predictions:
            caption = item['caption']
            gt_objects = set(item.get('gt_objects', []))

            # Extract mentioned objects
            mentioned_objects = self._extract_objects(caption)

            # Check for hallucinations
            hallucinated = []
            for obj in mentioned_objects:
                # Check if object is in ground truth (or synonym)
                is_present = False
                for gt_obj in gt_objects:
                    if obj in self.synonyms.get(gt_obj, {gt_obj}):
                        is_present = True
                        break
                    if gt_obj in obj or obj in gt_obj:  # Partial match
                        is_present = True
                        break

                if not is_present:
                    hallucinated.append(obj)

            # Update counters
            total_objects_mentioned += len(mentioned_objects)
            total_hallucinated_objects += len(hallucinated)

            if len(hallucinated) > 0:
                sentences_with_hallucination += 1

        # Compute CHAIR metrics
        chairs = sentences_with_hallucination / total_sentences if total_sentences > 0 else 0
        chairi = total_hallucinated_objects / total_objects_mentioned if total_objects_mentioned > 0 else 0

        return {
            'CHAIRs': chairs,  # Sentence-level
            'CHAIRi': chairi,  # Instance-level
            'total_sentences': total_sentences,
            'sentences_with_hallucination': sentences_with_hallucination,
            'total_objects_mentioned': total_objects_mentioned,
            'total_hallucinated_objects': total_hallucinated_objects,
            'avg_objects_per_caption': total_objects_mentioned / total_sentences if total_sentences > 0 else 0
        }


def evaluate_chair_predictions(
    predictions: List[Dict[str, Any]],
    coco_annotation_file: str
) -> Dict[str, float]:
    """
    Evaluate CHAIR metrics on predictions.

    Args:
        predictions: List of prediction dicts
        coco_annotation_file: Path to COCO annotations

    Returns:
        metrics: CHAIR evaluation results
    """
    evaluator = CHAIREvaluator(coco_annotation_file)
    return evaluator.evaluate(predictions)
