"""Image-only MSCOCO data loading for SAE vision training.

This file provides a minimal dataset and dataloader for images stored under
`SAE/data/MSCOCO/train2014`. The main entry point is `make_mscoco_dataloader`,
which returns batches of preprocessed image tensors suitable for the LLaVA
vision tower.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


@dataclasses.dataclass(frozen=True)
class MSCOCOImageLoaderConfig:
    """Configuration for image-only MSCOCO loading."""

    root: pathlib.Path = pathlib.Path("SAE/data/MSCOCO/train2014")
    batch_size: int = 8
    shuffle: bool = True
    num_workers: int = 0
    max_images: int | None = None


def _process_image(image_processor: Any, image: Image.Image) -> Tensor:
    """Run the LLaVA image processor and return one `[3, H, W]` tensor."""
    if hasattr(image_processor, "preprocess"):
        processed = image_processor.preprocess(image, return_tensors="pt")
    else:
        processed = image_processor(images=image, return_tensors="pt")

    if isinstance(processed, dict):
        pixel_values = processed["pixel_values"]
    else:
        pixel_values = processed.pixel_values

    msg = f"Expected pixel_values shape [1, 3, H, W], got {tuple(pixel_values.shape)}."
    assert pixel_values.ndim == 4 and pixel_values.shape[0] == 1, msg
    return pixel_values[0]


class MSCOCOImageDataset(Dataset):
    """Minimal image-only dataset over `train2014` JPEG files."""

    def __init__(
        self,
        image_root: pathlib.Path,
        *,
        image_processor: Any,
        max_images: int | None = None,
    ):
        self.image_root = image_root
        self.image_processor = image_processor

        msg = f"Image root does not exist: {image_root}"
        assert image_root.exists(), msg

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_paths = sorted(
            path for path in image_root.iterdir() if path.suffix.lower() in exts
        )
        if max_images is not None:
            image_paths = image_paths[:max_images]

        msg = f"No images found in {image_root}."
        assert image_paths, msg
        self.image_paths = image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        image_fpath = self.image_paths[index]
        with Image.open(image_fpath) as img:
            image = img.convert("RGB")

        pixel_values_chw = _process_image(self.image_processor, image)
        return {
            "pixel_values": pixel_values_chw,
            "image_path": str(image_fpath),
        }


def _collate_batch(
    batch: list[dict[str, Tensor | str]],
) -> dict[str, Tensor | list[str]]:
    pixel_values_bchw = torch.stack([sample["pixel_values"] for sample in batch])
    image_paths = [sample["image_path"] for sample in batch]
    return {
        "pixel_values": pixel_values_bchw,
        "image_paths": image_paths,
    }


def make_mscoco_dataset(
    image_processor: Any,
    *,
    root: str | pathlib.Path = "SAE/data/MSCOCO/train2014",
    max_images: int | None = None,
) -> MSCOCOImageDataset:
    """Create the image-only MSCOCO dataset."""
    return MSCOCOImageDataset(
        pathlib.Path(root),
        image_processor=image_processor,
        max_images=max_images,
    )


def make_mscoco_dataloader(
    image_processor: Any,
    *,
    root: str | pathlib.Path = "SAE/data/MSCOCO/train2014",
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    max_images: int | None = None,
) -> DataLoader:
    """Create a dataloader over preprocessed MSCOCO training images."""
    dataset = make_mscoco_dataset(
        image_processor,
        root=root,
        max_images=max_images,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate_batch,
    )
