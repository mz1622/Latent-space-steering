# SAE Workstream

This directory contains the minimal LLaVA vision-tower SAE workstream.

## Main paths

- `src/model_loader/`: model loading utilities
- `src/data_loader/`: image-only dataloaders
- `src/method/SAE.py`: SAE model, vision activation extractor, trainer, and step-2 alignment helpers
- `scripts/train_SAE.py`: end-to-end SAE training script
- `scripts/alignment.py`: runnable step-2 alignment analysis script

## Run

Use the local environment in `SAE/.venv311` and run:

```bash
SAE/.venv311/bin/python SAE/scripts/train_SAE.py --max-images 32 --layers -2
```

The current default SAE sparsity strength is `--l1-coeff 1e-3`.

Outputs are written under `SAE/outputs/sae/`.

## Alignment

Step-2 alignment utilities now live in `SAE/src/method/SAE.py`. They support:

- loading COCO positive/negative object supervision from `SAE/data/MSCOCO/annotations/edit_coco2014_train_combine.jsonl`
- top-k token pooling over token-wise SAE codes
- image-level pooled SAE samples for object presence analysis
- mention-level pooled SAE samples for supported vs hallucinated analysis
- direct mean-difference ranking helpers for aligned latents

Runnable example:

```bash
SAE/.venv311/bin/python SAE/scripts/alignment.py \
  --checkpoint SAE/outputs/sae/sae_layer_-16.pt \
  --max-records 32
```

The script now automatically:

- computes supported-vs-hallucinated statistics
- computes per-object presence statistics for every class with both present and absent support
- saves a summary JSON under `SAE/outputs/alignment/` by default
- can optionally save pooled samples with `--save-samples`
