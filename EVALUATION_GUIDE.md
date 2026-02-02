# Evaluation Guide

Quick reference for running evaluations with the unified multi-model script.

## Main Evaluation Script

**Use:** `scripts/evaluate_multi_model.py` for all evaluations

This unified script supports:
- ✅ Multiple models (LLaVA, Qwen3-VL)
- ✅ Multiple benchmarks (POPE, CHAIR*, MMHal*)
- ✅ Optimized batching with KV cache
- ✅ Debug mode for testing

*CHAIR and MMHal support coming soon

---

## Quick Commands

### LLaVA on POPE

```bash
# Debug test (fast)
python scripts/evaluate_multi_model.py \
  --model-type llava \
  --benchmark pope \
  --debug

# Full evaluation
python scripts/evaluate_multi_model.py \
  --model-type llava \
  --benchmark pope \
  --pope-type adversarial \
  --max-samples 3000 \
  --batch-size 1
```

### Qwen3-VL on POPE

```bash
# Debug test
python scripts/evaluate_multi_model.py \
  --model-type qwen3-vl \
  --benchmark pope \
  --debug

# Full evaluation
python scripts/evaluate_multi_model.py \
  --model-type qwen3-vl \
  --benchmark pope \
  --pope-type adversarial \
  --max-samples 3000 \
  --batch-size 1
```

### Custom Model Path

```bash
python scripts/evaluate_multi_model.py \
  --model-name "D:/Hallucination/models/Qwen3-VL-8B-Instruct" \
  --model-type qwen3-vl \
  --benchmark pope \
  --pope-type random
```

---

## POPE Variants

Run all three POPE variants for comprehensive evaluation:

```bash
# Random
python scripts/evaluate_multi_model.py --model-type llava --pope-type random

# Popular
python scripts/evaluate_multi_model.py --model-type llava --pope-type popular

# Adversarial (hardest)
python scripts/evaluate_multi_model.py --model-type llava --pope-type adversarial
```

---

## Performance Optimizations

The script now includes:

1. **KV Caching** (`use_cache=True`)
   - 3-5x faster generation
   - Enabled by default

2. **Batching** (via `--batch-size`)
   - Currently processes samples sequentially within batch
   - Future: True parallel batching

3. **Efficient Loading**
   - Models loaded once per run
   - Images preprocessed on-the-fly

---

## Output Structure

Results saved to timestamped directories:

```
outputs/
└── 20260131_230145_llava_pope_adversarial/
    ├── metrics.json              # Accuracy, Precision, Recall, F1
    ├── predictions.jsonl         # All predictions with labels
    └── llava_pope_adversarial.log  # Full execution log
```

---

## Comparing Models

Run evaluations for both models:

```bash
# LLaVA
python scripts/evaluate_multi_model.py --model-type llava --pope-type adversarial --max-samples 3000

# Qwen3-VL
python scripts/evaluate_multi_model.py --model-type qwen3-vl --pope-type adversarial --max-samples 3000
```

Compare results:
```python
import json

# Load metrics
llava = json.load(open('outputs/.../metrics.json'))
qwen = json.load(open('outputs/.../metrics.json'))

# Compare
print(f"LLaVA  - Acc: {llava['accuracy']:.4f}, F1: {llava['f1']:.4f}")
print(f"Qwen3V - Acc: {qwen['accuracy']:.4f}, F1: {qwen['f1']:.4f}")
```

---

## Other Evaluation Scripts

While `evaluate_multi_model.py` is the main script, these remain for specific use cases:

### CHAIR Evaluation
```bash
python scripts/evaluate_chair.py --max-samples 500
```

### MMHal Baseline
```bash
python scripts/evaluate_mmhal_baseline.py
```

### MMHal GPT Evaluation
```bash
python scripts/evaluate_mmhal_gpt.py --response outputs/.../mmhal_responses.json
```

---

## Troubleshooting

### Slow Performance
Check if KV cache is enabled (should be by default):
```python
# In the script, verify:
output_ids = model.generate(
    ...,
    use_cache=True  # Must be True
)
```

### GPU Not Used
Run diagnostic:
```bash
python check_cuda.py
```

Should show `CUDA available: True`

### Out of Memory
- Reduce `--max-samples`
- Use smaller model (Qwen3-VL-2B instead of 8B)
- Lower batch size (currently 1 by default)

---

## Expected Performance

On RTX 5070 Ti (16GB VRAM):

| Model | Samples/sec | Time for 3000 samples |
|-------|-------------|----------------------|
| LLaVA-7B | 0.5-1.0 | 1-2 hours |
| Qwen3-VL-8B | 0.4-0.8 | 1.5-2.5 hours |

**Note**: First run may be slower due to model downloads and caching.

---

## Next Steps

1. **Run baseline evaluations** for both models
2. **Compare results** across POPE variants
3. **Extend to CHAIR** (image captioning)
4. **Extend to MMHal** (multi-modal hallucination)
5. **Apply steering methods** and re-evaluate
