# Architecture Documentation

## Overview

This codebase implements a modular framework for mitigating hallucinations in Large Vision-Language Models (LVLMs) via latent space steering.

## Design Principles

1. **Separation of Concerns**: Detection and steering are separate modules
2. **Extensibility**: Easy to add new methods, datasets, models
3. **Reproducibility**: Config-driven, versioned outputs
4. **Clean Interfaces**: Standard method abstraction (fit/apply/infer/evaluate)
5. **Development-Friendly**: Debug mode, sample limiting, extensive logging

## Core Abstractions

### SteeringMethod (src/steering/base.py)

All steering methods implement this interface:

```python
class SteeringMethod(ABC):
    def fit(model, train_data, max_samples, **kwargs) -> artifacts
    def apply(model, **kwargs) -> model
    def infer(model, inputs, **kwargs) -> outputs
    def evaluate(predictions, metrics, **kwargs) -> results
```

**Methods:**

- `fit()`: Compute steering directions from training data
  - Input: Model, training data
  - Output: Artifacts (directions, metadata)
  - Persists: Saves artifacts to disk

- `apply()`: Apply steering to model
  - Input: Model, steering parameters
  - Output: Modified model (with hooks/layers)
  - Side effect: Modifies model in-place

- `infer()`: Run generation with steering
  - Input: Model (with steering), test inputs
  - Output: Generated text
  - Use case: Inference pipeline

- `evaluate()`: Compute metrics
  - Input: Predictions, ground truth
  - Output: Metrics dict
  - Use case: Evaluation pipeline

### HallucinationDetector (src/detectors/base.py)

Interface for detection methods (placeholder):

```python
class HallucinationDetector(ABC):
    def fit(model, train_data, **kwargs) -> artifacts
    def detect(model, inputs, **kwargs) -> detections
```

Current implementation: `PassthroughDetector` (always returns "no hallucination")

## Implemented Methods

### Mean Difference Steering (VTI)

**File:** `src/steering/mean_difference.py`

**Algorithm:**
1. Collect paired examples: (image, truthful_caption, hallucinated_caption)
2. Run model on both captions, collect hidden states
3. Compute mean difference: `h_truthful - h_hallucinated`
4. Apply PCA to extract principal direction
5. Steer by adding direction to hidden states during inference

**Key Parameters:**
- `alpha_image`: Visual steering strength
- `alpha_text`: Text steering strength
- `rank`: Number of PCA components
- `mask_ratio`: Ratio of image patches to mask
- `num_trials`: Number of masking trials

**References:**
- Liu et al., "Reducing Hallucinations in Vision-Language Models via Latent Space Steering", 2024

## Data Flow

```
Config File (YAML)
    ↓
Load Config + CLI Overrides
    ↓
Load Model (LLaVA)
    ↓
Load Demo Data (COCO + annotations)
    ↓
Initialize Detector (PassthroughDetector)
    ↓
Initialize Steering Method (MeanDifferenceSteeringMethod)
    ↓
Fit: Compute Directions
    ↓
Save Artifacts
    ↓
Apply: Add Steering Layers
    ↓
Inference: Generate with Steering
    ↓
Evaluate: Compute Metrics
    ↓
Save Results
    ↓
Cleanup: Remove Steering
```

## Module Responsibilities

### src/steering/
Implements steering methods.
- `base.py`: Abstract interface
- `mean_difference.py`: VTI implementation
- Future: `truthful.py`, `autosteer.py`, `sea.py`

### src/detectors/
Implements hallucination detection.
- `base.py`: Abstract interface, PassthroughDetector
- Future: Probe-based, uncertainty-based, attention-based

### src/models/
Model loading and utilities.
- `loader.py`: Load LLaVA and other LVLMs

### src/data/
Dataset loading and preprocessing.
- `loader.py`: Load demo data, evaluation datasets

### src/evaluation/
Evaluation metrics.
- `metrics.py`: CHAIR, POPE, GPT-based eval

### src/utils/
Common utilities.
- `config.py`: Config loading, merging, saving
- `logging_utils.py`: Logging, experiment dirs, versioning

### configs/
YAML configuration files.
- `default.yaml`: Default settings
- `methods/`: Method-specific configs

### scripts/
Standalone scripts.
- `run_eval.py`: Evaluation entrypoint

## Configuration System

Configs are hierarchical YAML files with override support:

```yaml
# Base config: configs/default.yaml
steering:
  method: "mean_difference"
  config:
    alpha_text: 0.8

# Override: configs/methods/custom.yaml
steering:
  config:
    alpha_text: 1.0  # Overrides base

# CLI override:
# --alpha-text 1.5  # Overrides both
```

Merge order: Base config → Method config → CLI args

## Experiment Outputs

Each run creates:

```
outputs/YYYYMMDD_HHMMSS/experiment_name/
├── config.yaml          # Full merged config
├── metrics.json         # Evaluation results
├── environment.txt      # Python/CUDA/GPU info
├── git_commit.txt       # Git commit + dirty status
├── experiment_name.log  # Execution log
└── artifacts/           # Steering directions
    ├── visual_direction.pt
    ├── textual_direction.pt
    └── config.json
```

This ensures full reproducibility.

## Debug Mode

Debug mode (`--debug`) enables:
- Mock steering directions (random tensors)
- Skipped expensive computations
- Minimal data loading
- Fast iteration (~seconds vs hours)

Useful for:
- Testing code changes
- Verifying pipeline
- CI/CD integration

## Extension Points

### Adding a New Steering Method

1. Create `src/steering/new_method.py`
2. Inherit from `SteeringMethod`
3. Implement `fit()`, `apply()`, `infer()`, `evaluate()`
4. Add to `main.py` method registry
5. Create `configs/methods/new_method.yaml`

### Adding a New Dataset

1. Add loader in `src/data/loader.py`
2. Implement dataset-specific preprocessing
3. Update config schema
4. Add evaluation metrics if needed

### Adding a New Detector

1. Create `src/detectors/new_detector.py`
2. Inherit from `HallucinationDetector`
3. Implement `fit()`, `detect()`
4. Add to `main.py` detector registry

## Testing Strategy

- **Unit tests**: Test individual components in isolation
- **Integration tests**: Test full pipeline with mock data
- **Smoke tests**: Quick runs to verify nothing is broken
- **Benchmarks**: Full evaluation on standard datasets

Planned structure:
```
tests/
├── test_steering.py
├── test_detectors.py
├── test_data.py
└── test_integration.py
```

## Dependencies

### Core
- PyTorch
- transformers
- LLaVA (included in Latent-space-steering/)

### Data
- datasets (HuggingFace)
- PIL
- torchvision

### Utilities
- PyYAML (config)
- tqdm (progress bars)

### Evaluation
- OpenAI API (for GPT-based evaluation)
- pycocotools (for CHAIR)

## Future Directions

### Phase 2: Framework Consolidation
- Unified evaluation harness
- Comprehensive metrics
- Automated benchmarking
- Test coverage

### Phase 3: Research Extensions
- Non-linear steering (SEA-inspired)
- Conditional steering (AutoSteer-like)
- Probe-based detection
- Transfer learning across models
- Benchmark integration (AXBENCH, RePS)

## References

- **VTI**: Liu et al., "Reducing Hallucinations in Vision-Language Models via Latent Space Steering", 2024
- **SEA**: Concept geometry in latent spaces
- **AutoSteer**: Probe-triggered conditional steering
- **TruthPrInt**: Truthful-guided interventions
- **CHAIR**: Rohrbach et al., "Object Hallucination in Image Captioning", 2018
- **POPE**: Li et al., "Evaluating Object Hallucination in Large Vision-Language Models", 2023
