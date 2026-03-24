# DEVLOG

## 2026-03-23

Expanded `SAE/scripts/alignment.py` from a single-object helper into the full step-2 analysis script. It now computes global hallucination stats and per-object presence stats for all object classes in the loaded subset, and saves a machine-readable summary by default.

Added `SAE/scripts/alignment.py` so the new step-2 alignment analysis is directly runnable with configurable SAE checkpoint, annotation file, image root, target object, and JSON export.

Implemented the step-2 alignment setup for the vision-tower SAE directly in `SAE/src/method/SAE.py`. The new code builds pooled image-level and mention-level alignment samples from the COCO positive/negative object JSONL and exposes simple mean-difference ranking helpers for aligned latents.

Improved the SAE objective in `SAE/src/method/SAE.py` so the sparse part is no longer effectively negligible by default. The code now uses a stronger default `l1_coeff=1e-3`, keeps SAE optimization in float32 end-to-end, and prints per-epoch sparsity diagnostics to show whether the latent is actually sparse.

Hardened the LLaVA vision extractor in `SAE/src/method/SAE.py` so it matches the local object structure and no longer assumes token 0 is always CLS. The extractor now expects preprocessed `pixel_values`, keeps `flat_nd` consistently flattened, and includes a lightweight object-structure smoke test.

Fixed the first real training bug in the vision-tower SAE path: SAE optimization was inheriting `float16` from LLaVA and producing NaNs. SAE training now runs in `float32` and raises immediately on non-finite activations, losses, gradients, or weights.

Added the first runnable image-only SAE training path for the LLaVA vision tower.

Paths:
- `SAE/src/data_loader/data_loader.py`
- `SAE/scripts/train_SAE.py`

Run:

```bash
SAE/.venv311/bin/python SAE/scripts/train_SAE.py --max-images 32 --layers -2
```
