# Evaluation Benchmarks

This document describes the hallucination evaluation benchmarks supported in this framework.

## Overview

We support three major benchmarks for evaluating hallucination in LVLMs:

1. **POPE** - Polling-based object probing with yes/no questions
2. **CHAIR** - Caption hallucination assessment
3. **MMHal-Bench** - GPT-based evaluation of detailed descriptions

## POPE (Polling-based Object Probing Evaluation)

### Description

POPE evaluates object hallucination by asking yes/no questions about whether specific objects are present in images.

### Variants

1. **Random**: Negative samples are randomly selected objects
2. **Popular**: Negative samples are popular COCO objects (harder)
3. **Adversarial**: Negative samples are co-occurring objects (hardest)

### Metrics

- **Accuracy**: Overall correctness
- **Precision**: Of all "yes" answers, how many are correct
- **Recall**: Of all present objects, how many are identified
- **F1 Score**: Harmonic mean of precision and recall
- **Yes Ratio**: Tendency to answer "yes" (measures conservativeness)

### Usage

```bash
# Evaluate on random variant
python scripts/evaluate_pope.py --pope-type random

# Evaluate on adversarial variant
python scripts/evaluate_pope.py --pope-type adversarial

# With steering
python scripts/evaluate_pope.py --pope-type adversarial --use-steering --alpha-text 0.9

# Debug mode
python scripts/evaluate_pope.py --pope-type random --debug
```

### Reference

Li et al., "Evaluating Object Hallucination in Large Vision-Language Models", EMNLP 2023

- [Paper](https://arxiv.org/abs/2305.10355)
- [GitHub](https://github.com/RUCAIBox/POPE)

## CHAIR (Caption Hallucination Assessment with Image Relevance)

### Description

CHAIR evaluates hallucination in free-form image captions by comparing mentioned objects against ground truth annotations.

### Metrics

- **CHAIRs** (Sentence-level): Percentage of sentences containing at least one hallucinated object
- **CHAIRi** (Instance-level): Percentage of mentioned objects that are hallucinated

### Usage

```bash
# Baseline evaluation
python scripts/evaluate_chair.py --max-samples 500

# With steering
python scripts/evaluate_chair.py --use-steering --alpha-text 0.9 --alpha-image 0.9

# Debug mode
python scripts/evaluate_chair.py --debug
```

### Reference

Rohrbach et al., "Object Hallucination in Image Captioning", EMNLP 2018

- [Paper](https://arxiv.org/abs/1809.02156)

## Data Requirements

### POPE
- COCO val2014 images
- POPE JSON files (included in `data/pope_coco/`)

### CHAIR
- COCO val2014 images
- COCO annotations (`annotations/instances_val2014.json`)

## Full Documentation

See [docs/benchmarks.md](benchmarks.md) for comprehensive documentation including:
- Detailed metrics explanations
- Expected performance ranges
- Comparison of benchmarks
- Troubleshooting guide
- Tips for good results
