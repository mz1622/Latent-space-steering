# MEMORY

- User issue: needed an image-only MSCOCO loader and a script that trains SAEs on LLaVA vision activations.
- Key edits: added `SAE/src/data_loader/` and `SAE/scripts/train_SAE.py`.
- Solution summary: LLaVA is loaded once, MSCOCO images are preprocessed with its image processor, activations are collected layer-wise from the real CLIP vision model under the local LLaVA wrapper, and one SAE is trained per selected vision layer.
- Caveat: real end-to-end training still depends on the local LLaVA environment and model weights being available. SAE training must stay in `float32`; using the model's default `float16` can produce NaNs. Current default sparsity strength is `l1_coeff=1e-3`, and the epoch logs now expose density/active-latent diagnostics so weak sparsity is visible immediately.
- Step-2 alignment now exists in `SAE/src/method/SAE.py`: COCO JSONL records can be turned into pooled image-level or mention-level SAE samples, and the current alignment analysis is direct mean-difference statistics only, with no steering or probes yet.
- `SAE/scripts/alignment.py` is the runnable front end for step-2. It now automatically summarizes all object classes in the loaded subset, saves a JSON summary by default, and can optionally save pooled `.pt` samples.
