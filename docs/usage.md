# Usage Guide

## Quick Start

### 1. Basic Usage

Run the complete pipeline with default settings:

```bash
python main.py --config configs/default.yaml
```

### 2. Debug Mode

Run in debug mode to test the pipeline without loading the full model or computing real directions:

```bash
python main.py --config configs/default.yaml --debug
```

Debug mode:
- Uses mock steering directions
- Skips expensive LLM computations
- Useful for testing code changes quickly

### 3. Limited Samples

Test with a small number of training samples:

```bash
python main.py --config configs/default.yaml --max-samples 10
```

This is useful for:
- Quick iterations during development
- Verifying the pipeline works before full runs
- Testing on limited hardware

### 4. Custom Steering Strength

Override steering parameters from command line:

```bash
python main.py \
  --config configs/default.yaml \
  --alpha-image 0.9 \
  --alpha-text 0.9
```

### 5. Text-Only Steering

Disable visual steering, only steer text decoder:

```bash
python main.py \
  --config configs/methods/vti_text_only.yaml
```

## Configuration

### Config Files

Config files are in YAML format under `configs/`:

- `configs/default.yaml` - Default configuration
- `configs/methods/` - Method-specific configs

### Config Structure

```yaml
experiment:
  name: "experiment_name"
  seed: 42
  debug: false
  max_samples: null

model:
  name: "liuhaotian/llava-v1.5-7b"
  device: "auto"

data:
  data_file: "/path/to/MSCOCO/"
  num_demos: 70

detector:
  type: "passthrough"  # placeholder for future

steering:
  method: "mean_difference"  # VTI-like approach
  config:
    alpha_image: 0.9    # Visual steering strength
    alpha_text: 0.9     # Text steering strength
    rank: 1             # PCA components
    mask_ratio: 0.99    # Image masking ratio
    num_trials: 50      # Masking trials per image

generation:
  num_beams: 5
  max_new_tokens: 256

output:
  base_dir: "./outputs"
  save_artifacts: true
```

### Command-Line Overrides

Override config values from command line:

```bash
python main.py \
  --config configs/default.yaml \
  --method mean_difference \
  --alpha-text 1.0 \
  --max-samples 20 \
  --debug
```

## Project Structure

```
├── main.py                 # Main entry point
├── configs/                # Configuration files
│   ├── default.yaml
│   └── methods/
├── src/                    # Source code
│   ├── steering/           # Steering methods
│   │   ├── base.py        # Abstract base class
│   │   └── mean_difference.py  # VTI implementation
│   ├── detectors/          # Hallucination detection
│   ├── models/             # Model loading
│   ├── data/               # Data loading
│   ├── evaluation/         # Metrics
│   └── utils/              # Utilities
├── scripts/                # Scripts
│   └── run_eval.py        # Evaluation entrypoint
├── artifacts/              # Saved steering directions
├── outputs/                # Experiment outputs
└── docs/                   # Documentation
```

## Outputs

Each experiment creates a timestamped directory in `outputs/`:

```
outputs/
└── 20260131_143022/
    └── experiment_name/
        ├── config.yaml         # Saved configuration
        ├── metrics.json        # Evaluation metrics
        ├── environment.txt     # Python/CUDA versions
        ├── git_commit.txt      # Git commit hash
        ├── experiment_name.log # Execution log
        └── artifacts/          # Steering directions
            ├── visual_direction.pt
            ├── textual_direction.pt
            └── config.json
```

## Workflow Phases

The main.py script executes these phases:

1. **Model Loading** - Load LVLM (LLaVA)
2. **Data Loading** - Load demonstration data
3. **Hallucination Detection** - Initialize detector (placeholder)
4. **Steering Direction Computation** - Compute directions via fit()
5. **Apply Steering** - Modify model with steering layers
6. **Inference & Evaluation** - Run generation and evaluate
7. **Cleanup** - Remove steering, save results

## Adding New Steering Methods

To add a new steering method:

1. Create `src/steering/your_method.py`
2. Inherit from `SteeringMethod` base class
3. Implement: `fit()`, `apply()`, `infer()`, `evaluate()`
4. Register in main.py
5. Create config file in `configs/methods/`

Example:

```python
from src.steering.base import SteeringMethod

class YourMethod(SteeringMethod):
    def fit(self, model, train_data, max_samples=None, **kwargs):
        # Compute directions
        ...
        return artifacts

    def apply(self, model, **kwargs):
        # Apply to model
        ...
        return model

    def infer(self, model, inputs, **kwargs):
        # Generate outputs
        ...
        return outputs

    def evaluate(self, predictions, metrics=None, **kwargs):
        # Compute metrics
        ...
        return results
```

## Examples

### Run VTI with High Steering

```bash
python main.py --config configs/methods/vti_high_steering.yaml
```

### Run with Custom Output Directory

```bash
python main.py \
  --config configs/default.yaml \
  --output-dir ./my_experiments
```

### Quick Test Run

```bash
python main.py \
  --config configs/default.yaml \
  --debug \
  --max-samples 5
```

This will:
- Use mock computations (debug mode)
- Limit to 5 training samples
- Complete in seconds rather than hours
