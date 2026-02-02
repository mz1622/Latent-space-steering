"""
POPE (Polling-based Object Probing Evaluation) dataset loader.

POPE evaluates object hallucination by asking yes/no questions about objects in images.
Three variants: Random, Popular, and Adversarial sampling.

Reference: Li et al., "Evaluating Object Hallucination in Large Vision-Language Models", 2023
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset


class POPEDataset(Dataset):
    """
    POPE dataset for object hallucination evaluation.

    Each sample contains:
    - image: PIL Image
    - question: Yes/no question about object presence
    - label: 0 (No) or 1 (Yes)
    - image_id: COCO image ID
    """

    def __init__(
        self,
        pope_file: str,
        image_dir: str,
        pope_type: str = "random",  # random, popular, adversarial
        transform: Optional[Any] = None
    ):
        """
        Initialize POPE dataset.

        Args:
            pope_file: Path to POPE JSON file
            image_dir: Directory containing COCO images
            pope_type: Type of POPE evaluation (random/popular/adversarial)
            transform: Optional image transformation
        """
        self.pope_file = pope_file
        self.image_dir = Path(image_dir)
        self.pope_type = pope_type
        self.transform = transform

        # Load POPE data (supports JSON list or JSONL)
        with open(pope_file, "r", encoding="utf-8") as f:
            try:
                self.data = json.load(f)
            except json.JSONDecodeError:
                f.seek(0)
                self.data = [json.loads(line) for line in f if line.strip()]

        print(f"Loaded {len(self.data)} POPE {pope_type} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load image
        image_id = item['image']
        image_path = self.image_dir / image_id
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Get question and label
        question = item['text']
        label = 1 if item['label'] == 'yes' else 0

        return {
            'image': image,
            'question': question,
            'label': label,
            'image_id': image_id
        }


def load_pope_dataset(
    data_dir: str,
    pope_type: str = "random",
    image_processor: Optional[Any] = None,
    split: str = "val2014"
) -> POPEDataset:
    """
    Load POPE dataset.

    Args:
        data_dir: Base directory containing POPE and COCO data
        pope_type: Type of POPE (random, popular, adversarial)
        image_processor: Image processor/transform
        split: COCO split (val2014 or test2014)

    Returns:
        POPEDataset instance
    """
    # POPE file path
    pope_file = os.path.join(data_dir, f"pope_coco/coco_pope_{pope_type}.json")

    # Image directory
    image_dir = os.path.join(data_dir, split)

    if not os.path.exists(pope_file):
        raise FileNotFoundError(
            f"POPE file not found: {pope_file}\n"
            f"Please download POPE data from: https://github.com/RUCAIBox/POPE"
        )

    if not os.path.exists(image_dir):
        raise FileNotFoundError(
            f"Image directory not found: {image_dir}\n"
            f"Please download COCO {split} images"
        )

    return POPEDataset(
        pope_file=pope_file,
        image_dir=image_dir,
        pope_type=pope_type,
        transform=image_processor
    )


def evaluate_pope_predictions(predictions: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Evaluate POPE predictions.

    Args:
        predictions: List of dicts with 'prediction', 'label', 'question'

    Returns:
        metrics: Dict with accuracy, precision, recall, F1, yes_ratio
    """
    NEG_WORDS = ["No", "not", "no", "NO", "n't"]

    pred_list = []
    label_list = []

    for item in predictions:
        answer = item['prediction'].strip()
        label = item['label']

        # Parse answer (check for negative words)
        answer_clean = answer.replace(".", "").replace(",", "")
        words = answer_clean.split()

        if any(word in NEG_WORDS for word in words) or any(word.endswith("n't") for word in words):
            pred = 0  # No
        else:
            pred = 1  # Yes

        pred_list.append(pred)
        label_list.append(label)

    # Compute metrics
    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == 1 and label == 1:
            TP += 1
        elif pred == 1 and label == 0:
            FP += 1
        elif pred == 0 and label == 0:
            TN += 1
        elif pred == 0 and label == 1:
            FN += 1

    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    yes_ratio = pred_list.count(1) / len(pred_list) if len(pred_list) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'yes_ratio': yes_ratio,
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'num_samples': len(predictions)
    }
