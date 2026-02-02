# Hallucination Mitigation via Latent Space Steering

A modular research framework for mitigating hallucinations in Large Vision-Language Models (LVLMs) through latent space steering methods.

## Overview

This codebase provides:

1. **Extensible Framework**: Clean abstractions for steering methods and hallucination detection
2. **VTI Baseline**: Implementation of Visual and Textual Intervention (Liu et al., 2024)
3. **Config-Driven**: All experiments controlled via YAML configs
4. **Reproducible**: Automatic versioning of outputs, configs, and environment
5. **Development-Friendly**: Debug mode, sample limiting, extensive logging

## Quick Start

### Installation

```bash
# Clone repository
git clone <repo-url>
cd Hallucination/Latent-space-steering

# Install dependencies
pip install -r requirement.txt
pip install pyyaml

# Download COCO dataset (required for VTI)
# Run download_mscoco.py or manually download from https://cocodataset.org/
```

### Basic Usage

```bash
# Run with default settings
python main.py --config configs/default.yaml

# Debug mode (fast, uses mock computations)
python main.py --config configs/default.yaml --debug

# Limit training samples for quick testing
python main.py --config configs/default.yaml --max-samples 10

# Custom steering strength
python main.py --config configs/default.yaml --alpha-text 0.9 --alpha-image 0.9
```

## Project Structure

```
Latent-space-steering/
├── main.py                 # Main entry point ⭐
├── configs/                # Configuration files
│   ├── default.yaml       # Default config
│   └── methods/           # Method-specific configs
├── src/                   # Source code
│   ├── steering/          # Steering methods
│   │   ├── base.py       # Abstract interface
│   │   └── mean_difference.py  # VTI implementation
│   ├── detectors/         # Hallucination detection (placeholder)
│   ├── models/            # Model loading utilities
│   ├── data/              # Data loading utilities
│   ├── evaluation/        # Evaluation metrics
│   └── utils/             # Common utilities
├── scripts/               # Standalone scripts
├── docs/                  # Documentation
│   ├── usage.md          # Usage guide
│   ├── architecture.md   # Architecture details
│   └── assumptions.md    # Implementation assumptions
├── vti_utils/             # VTI utility functions
├── experiments/data/      # Demo data for direction computation
├── artifacts/             # Saved steering directions
└── outputs/               # Experiment outputs
```

## Features

### 1. Modular Steering Methods

All methods implement a standard interface:

```python
class SteeringMethod:
    def fit(model, train_data, ...) -> artifacts
    def apply(model, ...) -> model
    def infer(model, inputs, ...) -> outputs
    def evaluate(predictions, ...) -> metrics
```

**Currently Implemented:**
- Mean Difference (VTI): Liu et al., 2024

**Planned:**
- TruthPrInt-like: Truthful-guided interventions
- AutoSteer-like: Probe-triggered conditional steering
- SEA-inspired: Non-linear manifold steering

### 2. Hallucination Detection

Placeholder module for future detection methods:

```python
class HallucinationDetector:
    def fit(model, train_data, ...) -> artifacts
    def detect(model, inputs, ...) -> detections
```

**Currently:** PassthroughDetector (no detection, always steer)

**Planned:**
- Uncertainty-based detection
- Attention-based detection
- Probe-based detection

### 3. Config-Driven Experiments

All experiments controlled via YAML:

```yaml
steering:
  method: "mean_difference"
  config:
    alpha_image: 0.9     # Visual steering strength
    alpha_text: 0.9      # Text steering strength
    rank: 1              # PCA components
    mask_ratio: 0.99     # Image masking ratio
    num_trials: 50       # Masking trials
```

Override from command line:

```bash
python main.py --config configs/default.yaml --alpha-text 1.0
```

### 4. Reproducible Outputs

Each experiment creates a versioned directory:

```
outputs/20260131_143022/experiment_name/
├── config.yaml          # Full merged config
├── metrics.json         # Evaluation results
├── environment.txt      # Python/CUDA/GPU info
├── git_commit.txt       # Git commit + status
├── experiment_name.log  # Execution log
└── artifacts/           # Steering directions
    ├── visual_direction.pt
    ├── textual_direction.pt
    └── config.json
```

### 5. Evaluation Benchmarks

Support for multiple hallucination benchmarks:

**POPE (Polling-based Object Probing Evaluation)**
- Binary yes/no questions about object presence
- Three variants: Random, Popular, Adversarial
- Metrics: Accuracy, Precision, Recall, F1

```bash
python scripts/evaluate_pope.py --pope-type random
python scripts/evaluate_pope.py --pope-type adversarial --use-steering
```

**CHAIR (Caption Hallucination Assessment with Image Relevance)**
- Evaluates object hallucination in generated captions
- Metrics: CHAIRs (sentence-level), CHAIRi (instance-level)

```bash
python scripts/evaluate_chair.py --max-samples 500
python scripts/evaluate_chair.py --use-steering --alpha-text 0.9
```

**MMHal-Bench**
- GPT-based evaluation of detailed descriptions
- Currently supported via main pipeline

### 6. Debug Mode

Fast iteration without expensive computations:

```bash
python main.py --config configs/default.yaml --debug
```

Debug mode:
- Uses mock steering directions
- Skips LLM inference
- Completes in seconds vs hours
- Perfect for testing pipeline changes

## Workflow

The pipeline executes these phases:

1. **Model Loading** - Load LVLM (e.g., LLaVA-1.5)
2. **Data Loading** - Load demonstration data
3. **Detection** - Initialize hallucination detector
4. **Steering Computation** - Compute directions via `fit()`
5. **Apply Steering** - Modify model with steering
6. **Inference** - Generate with steering applied
7. **Evaluation** - Compute metrics
8. **Cleanup** - Remove steering, save results

## Documentation

- [Quick Start](docs/quick_start.md) - Get started in 3 steps ⭐
- [Usage Guide](docs/usage.md) - Detailed usage instructions
- [Architecture](docs/architecture.md) - Design and implementation details
- [Assumptions](docs/assumptions.md) - Implementation choices and rationale

## Examples

### Steering Methods

```bash
# VTI with high steering
python main.py --config configs/methods/vti_high_steering.yaml

# Text-only steering
python main.py --config configs/methods/vti_text_only.yaml

# Quick test
python main.py --debug --max-samples 5
```

### Benchmark Evaluations

```bash
# POPE evaluation (all three variants)
python scripts/evaluate_pope.py --pope-type random
python scripts/evaluate_pope.py --pope-type popular
python scripts/evaluate_pope.py --pope-type adversarial

# CHAIR evaluation
python scripts/evaluate_chair.py --max-samples 500

# With steering
python scripts/evaluate_pope.py --pope-type adversarial --use-steering --alpha-text 0.9
python scripts/evaluate_chair.py --use-steering --alpha-image 0.9 --alpha-text 0.9

# Debug mode for quick testing
python scripts/evaluate_pope.py --pope-type random --debug
python scripts/evaluate_chair.py --debug
```

## Adding New Methods

1. Create `src/steering/your_method.py`
2. Inherit from `SteeringMethod`
3. Implement `fit()`, `apply()`, `infer()`, `evaluate()`
4. Add to `main.py` method registry
5. Create config in `configs/methods/`

See [architecture.md](docs/architecture.md) for details.

## Requirements

- Python 3.9+
- PyTorch 2.0+
- transformers
- LLaVA (included in Latent-space-steering/)
- datasets (HuggingFace)
- PyYAML
- tqdm

See `Latent-space-steering/requirement.txt` for full dependencies.

## Development Roadmap

### Phase 1: VTI Replication (Current)
- [x] Clean architecture and abstractions
- [x] VTI method implementation
- [x] Config system
- [x] Debug mode and logging
- [ ] Full evaluation harness
- [ ] Unit tests

### Phase 2: Framework Consolidation
- [ ] Additional datasets (CHAIR, POPE)
- [ ] Comprehensive metrics
- [ ] Automated benchmarking
- [ ] Test coverage
- [ ] Documentation expansion

### Phase 3: Research Extensions
- [ ] Non-linear steering (SEA)
- [ ] Conditional steering (AutoSteer)
- [ ] Probe-based detection
- [ ] Transfer experiments
- [ ] Benchmark integration (AXBENCH, RePS)

## Citation

If you use this codebase, please cite:

```bibtex
@article{liu2024reducing,
  title={Reducing Hallucinations in Vision-Language Models via Latent Space Steering},
  author={Liu, Sheng and Ye, Haotian and Zou, James},
  journal={arXiv preprint arXiv:2410.15778},
  year={2024}
}
```

## License

See LICENSE file for details.

## Contributing

Contributions welcome! Please:
1. Follow existing code style
2. Add tests for new features
3. Update documentation
4. Follow the steering method interface

## Contact

For questions or issues, please open a GitHub issue.
