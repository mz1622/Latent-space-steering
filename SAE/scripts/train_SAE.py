"""Train SAEs on LLaVA vision tower activations from MSCOCO images.

This script loads a LLaVA model, builds an image-only dataloader over
`SAE/data/MSCOCO/train2014`, extracts selected vision-layer patch activations,
trains one SAE per requested layer, and saves the checkpoints under
`SAE/outputs/sae/`.

SAE/.venv311/bin/python SAE/scripts/train_SAE.py --max-images 64 --layers -16
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from typing import Any

import torch


def _setup_imports() -> None:
    this_dir = os.path.dirname(__file__)
    src_root = os.path.abspath(os.path.join(this_dir, "..", "src"))
    if src_root not in sys.path:
        sys.path.insert(0, src_root)


_setup_imports()

from data_loader import make_mscoco_dataloader
from method.SAE import LlavaVisionActivationExtractor
from method.SAE import train_saes_on_layers
from method.SAE import TrainerConfig
from model_loader import load_llava_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAE on LLaVA vision activations.")
    parser.add_argument("--model-name", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--data-root", default="SAE/data/MSCOCO/train2014")
    parser.add_argument("--output-dir", default="SAE/outputs/sae")
    parser.add_argument("--layers", nargs="+", type=int, default=[-2])
    parser.add_argument("--latent-dim", type=int, default=16384)
    parser.add_argument("--l1-coeff", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--include-cls", action="store_true")
    return parser.parse_args()


def collect_layer_activations(
    dataloader,
    extractor: LlavaVisionActivationExtractor,
    *,
    layers: list[int],
    include_cls: bool,
) -> dict[int, torch.Tensor]:
    acts_by_layer: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}

    for batch in dataloader:
        pixel_values_bchw = batch["pixel_values"]
        layer_outs = extractor.extract(
            pixel_values_bchw,
            layers=layers,
            include_cls=include_cls,
        )
        for layer, out in layer_outs.items():
            msg = f"Collected non-finite activations for layer {layer}."
            assert torch.isfinite(out.flat_nd).all(), msg
            acts_by_layer[layer].append(out.flat_nd.to(dtype=torch.float32).cpu())

    stacked = {}
    for layer, blocks in acts_by_layer.items():
        msg = f"No activations collected for layer {layer}."
        assert blocks, msg
        stacked[layer] = torch.cat(blocks, dim=0)
        msg = f"Stacked activations for layer {layer} contain non-finite values."
        assert torch.isfinite(stacked[layer]).all(), msg
    return stacked


def save_saes(
    saes: dict[int, Any],
    *,
    output_dir: pathlib.Path,
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "train_config.json", "w", encoding="utf-8") as fd:
        json.dump(metadata, fd, indent=2)

    for layer, sae in saes.items():
        ckpt = {
            "layer": layer,
            "state_dict": sae.state_dict(),
            "sae_config": {
                "input_dim": sae.cfg.input_dim,
                "latent_dim": sae.cfg.latent_dim,
                "l1_coeff": sae.cfg.l1_coeff,
            },
        }
        torch.save(ckpt, output_dir / f"sae_layer_{layer}.pt")


def main() -> None:
    args = parse_args()

    tokenizer, model, image_processor, context_len = load_llava_model(
        model_name=args.model_name,
        model_base=args.model_base,
        device=args.device,
    )
    del tokenizer, context_len

    dataloader = make_mscoco_dataloader(
        image_processor,
        root=args.data_root,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        max_images=args.max_images,
    )

    extractor = LlavaVisionActivationExtractor(model)
    activations = collect_layer_activations(
        dataloader,
        extractor,
        layers=args.layers,
        include_cls=args.include_cls,
    )

    trainer_cfg = TrainerConfig(
        batch_size=4096,
        lr=args.lr,
        epochs=args.epochs,
        device=next(model.parameters()).device,
    )
    saes = train_saes_on_layers(
        activations,
        latent_dim=args.latent_dim,
        l1_coeff=args.l1_coeff,
        trainer_cfg=trainer_cfg,
        dtype=torch.float32,
        device=next(model.parameters()).device,
    )

    output_dir = pathlib.Path(args.output_dir)
    save_saes(
        saes,
        output_dir=output_dir,
        metadata={
            "model_name": args.model_name,
            "data_root": args.data_root,
            "layers": args.layers,
            "latent_dim": args.latent_dim,
            "l1_coeff": args.l1_coeff,
            "epochs": args.epochs,
            "lr": args.lr,
            "max_images": args.max_images,
            "include_cls": args.include_cls,
        },
    )
    print(f"Saved SAE checkpoints to {output_dir}")


if __name__ == "__main__":
    main()
