"""Run full step-2 SAE alignment analysis on COCO positive/negative object lists.

This script loads a trained SAE checkpoint, loads the local LLaVA vision tower,
builds pooled image-level and mention-level alignment samples, computes direct
mean-difference rankings for all object classes and for
hallucinated-vs-supported mentions, and saves a structured summary to disk.

Example:

SAE/.venv311/bin/python SAE/scripts/alignment.py --checkpoint SAE/outputs/sae/sae_layer_-16.pt --max-records 32
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

from method.SAE import SAEConfig
from method.SAE import SparseAutoencoder
from method.SAE import compute_object_presence_stats
from method.SAE import compute_supported_vs_hallucinated_stats
from method.SAE import load_coco_object_records
from method.SAE import make_image_alignment_samples
from method.SAE import make_mention_alignment_samples
from method.SAE import rank_top_latents
from method.SAE import top_hallucination_latents
from model_loader import load_llava_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run step-2 SAE alignment analysis.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-name", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", default=None)
    parser.add_argument(
        "--annotations",
        default="SAE/data/MSCOCO/annotations/edit_coco2014_train_combine.jsonl",
    )
    parser.add_argument("--image-root", default="SAE/data/MSCOCO/train2014")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--max-records", type=int, default=64)
    parser.add_argument("--object-name", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="SAE/outputs/alignment")
    parser.add_argument("--save-json", default=None)
    parser.add_argument("--save-samples", action="store_true")
    return parser.parse_args()


def load_sae_checkpoint(
    ckpt_fpath: str | pathlib.Path,
    *,
    device: str | torch.device,
) -> tuple[SparseAutoencoder, dict[str, Any]]:
    """Load a trained SAE checkpoint saved by `train_SAE.py`."""
    ckpt = torch.load(ckpt_fpath, map_location="cpu")
    sae_cfg = ckpt["sae_config"]
    sae = SparseAutoencoder(
        SAEConfig(
            input_dim=int(sae_cfg["input_dim"]),
            latent_dim=int(sae_cfg["latent_dim"]),
            l1_coeff=float(sae_cfg["l1_coeff"]),
            dtype=torch.float32,
            device=device,
        )
    )
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()
    return sae, ckpt


def summarize_object_presence(
    image_samples,
    *,
    top_n: int,
) -> tuple[dict[str, dict[str, Any]], int]:
    """Compute object-presence summaries for all classes with both pos/neg support."""
    all_objects = sorted(
        {
            obj
            for sample in image_samples
            for obj in [*sample.pos_object_list, *sample.neg_object_list]
        }
    )
    summaries: dict[str, dict[str, Any]] = {}
    n_skipped = 0

    for object_name in all_objects:
        n_present = sum(object_name in s.pos_object_list for s in image_samples)
        n_absent = sum(object_name in s.neg_object_list for s in image_samples)
        if n_present == 0 or n_absent == 0:
            n_skipped += 1
            continue

        stats = compute_object_presence_stats(image_samples, object_name)
        summaries[object_name] = {
            "n_present": stats.n_present,
            "n_absent": stats.n_absent,
            "top_latents": rank_top_latents(stats.difference, top_n=top_n),
        }

    return summaries, n_skipped


def main() -> None:
    args = parse_args()

    _, model, image_processor, _ = load_llava_model(
        model_name=args.model_name,
        model_base=args.model_base,
        device=args.device,
    )
    sae, ckpt = load_sae_checkpoint(
        args.checkpoint,
        device=next(model.parameters()).device,
    )
    layer = ckpt["layer"] if args.layer is None else args.layer

    records = load_coco_object_records(args.annotations, max_records=args.max_records)
    image_samples = make_image_alignment_samples(
        records,
        sae=sae,
        model=model,
        image_processor=image_processor,
        layer=layer,
        image_root=args.image_root,
        top_k=args.top_k,
    )
    mention_samples = make_mention_alignment_samples(image_samples)
    mention_stats = compute_supported_vs_hallucinated_stats(mention_samples)
    hallucination_top = top_hallucination_latents(mention_samples, top_n=args.top_n)
    object_presence, n_skipped_objects = summarize_object_presence(
        image_samples,
        top_n=args.top_n,
    )

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_fpath = (
        pathlib.Path(args.save_json)
        if args.save_json is not None
        else output_dir / "alignment_summary.json"
    )
    summary_fpath.parent.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "layer": int(layer),
        "n_image_samples": len(image_samples),
        "n_mention_samples": len(mention_samples),
        "top_k": int(args.top_k),
        "annotation_path": str(args.annotations),
        "image_root": str(args.image_root),
        "supported_vs_hallucinated": {
            "n_supported": mention_stats.n_supported,
            "n_hallucinated": mention_stats.n_hallucinated,
            "top_latents": hallucination_top,
        },
        "object_presence": object_presence,
    }

    if args.object_name is not None and args.object_name in object_presence:
        summary["selected_object"] = {
            "object_name": args.object_name,
            **object_presence[args.object_name],
        }

    with open(summary_fpath, "w", encoding="utf-8") as fd:
        json.dump(summary, fd, indent=2)

    if args.save_samples:
        torch.save(image_samples, output_dir / "image_alignment_samples.pt")
        torch.save(mention_samples, output_dir / "mention_alignment_samples.pt")

    print(f"Processed {len(image_samples)} images from {args.annotations}")
    print(
        f"Layer={layer}, mentions={len(mention_samples)}, "
        f"supported={mention_stats.n_supported}, hallucinated={mention_stats.n_hallucinated}"
    )
    print("Top hallucination latents:")
    for idx, value in hallucination_top:
        print(f"  latent={idx} diff={value:.6f}")
    print(
        f"Summarized {len(object_presence)} object classes "
        f"(skipped {n_skipped_objects} without both present/absent support)"
    )

    if args.object_name is not None:
        if args.object_name in object_presence:
            print(f"Top latents for object presence '{args.object_name}':")
            for idx, value in object_presence[args.object_name]["top_latents"]:
                print(f"  latent={idx} diff={value:.6f}")
        else:
            print(
                f"Requested object '{args.object_name}' not summarized "
                f"(missing present or absent support in loaded subset)."
            )

    preview_objects = list(object_presence.items())[:2]
    for object_name, info in preview_objects:
        print(
            f"Preview {object_name}: present={info['n_present']} "
            f"absent={info['n_absent']} top={info['top_latents'][:3]}"
        )
    print(f"Saved alignment summary to {summary_fpath}")

    if args.save_samples:
        print(f"Saved pooled samples to {output_dir}")


if __name__ == "__main__":
    main()
