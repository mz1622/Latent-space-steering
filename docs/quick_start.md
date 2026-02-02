# Quick Start Guide

## Installation

```bash
# Navigate to project directory
cd d:/Hallucination/Latent-space-steering

# Install dependencies
pip install -r requirement.txt
```

## Test the Pipeline

### 1. Debug Mode (Fastest - Takes seconds)

Test the complete pipeline without loading models:

```bash
python main.py --debug
```

This will:
- Use mock steering directions
- Skip expensive LLM computations
- Verify the pipeline works end-to-end

### 2. Limited Samples (Medium - Takes minutes)

Run with real model but limited data:

```bash
python main.py --max-samples 5
```

This will:
- Load the actual LLaVA model
- Use only 5 training samples
- Compute real steering directions

### 3. Full Run (Slowest - Takes hours)

**Before running**, update paths in `configs/default.yaml`:
```yaml
data:
  data_file: "D:/path/to/MSCOCO/"  # Your COCO path
  image_folder: "D:/path/to/MSCOCO/val2014"
```

Then run:
```bash
python main.py --config configs/default.yaml
```

## Common Commands

```bash
# Custom steering strength
python main.py --alpha-text 0.9 --alpha-image 0.9

# Text-only steering
python main.py --config configs/methods/vti_text_only.yaml

# High steering
python main.py --config configs/methods/vti_high_steering.yaml

# Debug with custom settings
python main.py --debug --alpha-text 1.0
```

## Project Structure

```
Latent-space-steering/
├── main.py              # Main entry point ⭐
├── configs/             # YAML configurations
├── src/                 # Source code
│   ├── steering/       # Steering methods
│   ├── detectors/      # Detection (placeholder)
│   ├── models/         # Model loading
│   ├── data/           # Data loading
│   ├── evaluation/     # Metrics
│   └── utils/          # Utilities
├── vti_utils/          # VTI utility functions
├── experiments/data/   # Demo data
├── docs/               # Documentation
├── outputs/            # Results (created on run)
└── artifacts/          # Saved directions (created on run)
```

## Output

Each run creates a timestamped directory:

```
outputs/20260131_143022/vti_baseline/
├── config.yaml              # Configuration used
├── metrics.json             # Results
├── vti_baseline.log         # Execution log
├── environment.txt          # Python/CUDA info
├── git_commit.txt           # Git commit
└── artifacts/               # Steering directions
    ├── visual_direction.pt
    └── textual_direction.pt
```

## Next Steps

- Read [usage.md](usage.md) for detailed usage
- Read [architecture.md](architecture.md) for design details
- Check [assumptions.md](assumptions.md) for implementation choices
