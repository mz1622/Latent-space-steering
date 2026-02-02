# Implementation Assumptions

This document records assumptions made during implementation where paper details were unclear or implementation choices were required.

## VTI (Mean Difference Method)

### Direction Computation

**Assumption 1: PCA Component Selection**
- **Choice**: Use rank-1 PCA (single principal component)
- **Rationale**: Paper uses rank=1 in experiments
- **Alternative**: Could explore higher ranks for multi-dimensional steering
- **Config**: Exposed as `rank` parameter

**Assumption 2: Direction Aggregation**
- **Choice**: Average PCA components across demonstrations
- **Formula**: `(pca.components_.sum(dim=1) + pca.mean_).mean(0)`
- **Rationale**: Based on original VTI implementation
- **Alternative**: Could use other aggregation methods

**Assumption 3: Layer Selection**
- **Choice**: Skip first layer (`direction[1:]`)
- **Rationale**: First layer often represents low-level features
- **Alternative**: Could steer all layers or select specific layers

### Image Masking

**Assumption 4: Mask Ratio**
- **Choice**: Default 0.99 (mask 99% of patches)
- **Rationale**: VTI paper uses high masking ratios
- **Impact**: Higher ratio = stronger degradation signal
- **Config**: Exposed as `mask_ratio` parameter

**Assumption 5: Masking Strategy**
- **Choice**: Random patch selection
- **Rationale**: Simplest approach, used in VTI
- **Alternative**: Could use attention-guided masking

**Assumption 6: Number of Trials**
- **Choice**: Default 50 trials per image
- **Rationale**: Balance between variance reduction and compute
- **Impact**: More trials = more stable directions but slower
- **Config**: Exposed as `num_trials` parameter

### Steering Application

**Assumption 7: Steering Formula**
- **Choice**: Normalize and blend: `normalize(normalize(x) + 0.1 * y)`
- **Rationale**: From VTI implementation
- **Magic Number**: 0.1 scaling factor (hardcoded in VTI)
- **Alternative**: Could expose as hyperparameter

**Assumption 8: Alpha Interpretation**
- **Choice**: Alpha scales the direction magnitude
- **Range**: Typically 0.0 - 2.0
- **Impact**: Higher alpha = stronger steering
- **Trade-off**: Too high can degrade generation quality

### Model Architecture

**Assumption 9: Vision Model Path**
- **Choice**: `model.model.vision_tower.vision_tower.vision_model`
- **Rationale**: LLaVA-specific path
- **Alternative**: Fallback to `model.vision_model`
- **Generalization**: May need adjustment for other LVLMs

**Assumption 10: Layer Identification**
- **Choice**: Find longest ModuleList in model
- **Rationale**: Heuristic for finding transformer layers
- **Risk**: May fail on non-standard architectures
- **Future**: Could use model-specific configs

## Data

### Demonstration Selection

**Assumption 11: Random Sampling**
- **Choice**: Random sample of demonstrations
- **Rationale**: Simple and effective
- **Alternative**: Could select diverse/hard examples
- **Seed**: Controlled via config for reproducibility

**Assumption 12: Hallucinated Caption Source**
- **Choice**: Use provided `h_value` field from demo data
- **Rationale**: Dataset comes with paired captions
- **Assumption**: These captions represent realistic hallucinations
- **Future**: Could generate synthetic hallucinations

### Preprocessing

**Assumption 13: Image Preprocessing**
- **Choice**: Use model's default image processor
- **Rationale**: Standard practice
- **Alternative**: Could apply custom augmentations

**Assumption 14: Patch Size**
- **Choice**: 14x14 patches (ViT default)
- **Rationale**: Matches ViT-L/14 used in LLaVA
- **Risk**: Hardcoded, may not generalize

## Evaluation

### Metrics

**Assumption 15: GPT Evaluation**
- **Choice**: Use GPT-4 for MMHal-Bench evaluation
- **Rationale**: Paper uses GPT-based evaluation
- **Alternative**: Could use other LLMs or human eval
- **Cost**: Expensive, may need caching

**Assumption 16: Metric Definitions**
- **Choice**: Defer to dataset-specific metrics
- **Rationale**: Each benchmark has established protocols
- **Status**: Placeholders in current implementation

## Debug Mode

**Assumption 17: Mock Direction Shape**
- **Choice**: Use typical LLaVA dimensions (32 layers, 4096 hidden)
- **Rationale**: Allows testing without model
- **Risk**: May not match actual model architecture

**Assumption 18: Mock Direction Distribution**
- **Choice**: Small random Gaussian (std=0.01)
- **Rationale**: Prevent extreme values, test numerics
- **Impact**: Won't produce meaningful steering

## Configuration

**Assumption 19: Default Hyperparameters**
- **Source**: VTI paper experiments
- **Values**:
  - alpha_image: 0.9
  - alpha_text: 0.9
  - num_demos: 70
  - rank: 1
- **Note**: These may need tuning per dataset/model

**Assumption 20: Device Selection**
- **Choice**: Auto-select CUDA > MPS > CPU
- **Rationale**: Use best available hardware
- **Override**: Exposed in config

## Future Considerations

### Open Questions

1. **Optimal number of demonstrations**: 70 is default, but is this optimal?
2. **Layer-specific alphas**: Should different layers use different steering strengths?
3. **Dynamic steering**: Should steering strength vary per example?
4. **Direction persistence**: Do directions transfer across datasets/tasks?

### Assumptions to Validate

1. Random demonstration sampling is sufficient
2. Rank-1 PCA captures the key direction
3. Same alpha for all layers is optimal
4. Direction computed on one dataset generalizes

### Documentation Updates

When modifying assumptions:
1. Update this file
2. Update config with new parameters
3. Add ablation experiments to validate
4. Document in paper/report

## References

See `architecture.md` for implementation details and `usage.md` for how to override these assumptions via configuration.
