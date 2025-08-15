# VAR Layer Skipping Implementation

This implementation adds stage-specific layer skipping functionality to the VAR (Visual AutoRegressive) model, allowing you to skip specific transformer layers during certain generation stages.

## 🎯 Key Features

- **Stage-specific layer skipping**: Skip layers only in specific generation stages
- **Flexible configuration**: Skip individual layers, ranges, or use threshold-based skipping
- **Performance monitoring**: Track skipped layers and generation speed
- **Command-line interface**: Easy-to-use scripts for different skipping strategies

## 📁 Files Added

- `models/var_layer_skip.py` - Enhanced VAR model with layer skipping
- `var_evaluate_layer_skip.py` - Evaluation script with layer skipping support
- `run_layer_skip_evaluation.sh` - Convenient bash script with presets
- `LAYER_SKIP_README.md` - This documentation

## 🚀 Quick Start

### Basic Usage

```bash
# Skip layers 0,1,2 in the last stage (stage 9)
./run_layer_skip_evaluation.sh \
    --model_path /path/to/var_d16_new.pth \
    --vae_path /path/to/vae_ch160v4096z32.pth \
    skip-last-stage-low

# Run without any layer skipping (baseline)
./run_layer_skip_evaluation.sh \
    --model_path /path/to/var_d16_new.pth \
    --vae_path /path/to/vae_ch160v4096z32.pth \
    no-skip
```

### Advanced Custom Configuration

```bash
# Skip specific layers in multiple stages
./run_layer_skip_evaluation.sh \
    --model_path /path/to/var_d16_new.pth \
    --vae_path /path/to/vae_ch160v4096z32.pth \
    --stage_layer_skip "8:0,1;9:0,1,2" \
    custom

# Skip all layers below threshold 3 in the last stage
./run_layer_skip_evaluation.sh \
    --model_path /path/to/var_d16_new.pth \
    --vae_path /path/to/vae_ch160v4096z32.pth \
    --skip_layers_below "9:3" \
    custom
```

## 🔧 Configuration Options

### LayerSkipConfig Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `stage_layer_skip` | Dict[int, Set[int]] | Skip specific layers in specific stages | `{9: {0, 1, 2}}` |
| `skip_all_layers_below` | Dict[int, int] | Skip all layers below threshold | `{9: 3}` (skip layers 0,1,2) |
| `skip_all_layers_above` | Dict[int, int] | Skip all layers above threshold | `{9: 13}` (skip layers 14,15) |
| `enable_skip` | bool | Global enable/disable | `True` |

### Command Line Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--stage_layer_skip` | Stage-specific layer skipping | `"9:0,1,2"` |
| `--skip_layers_below` | Skip layers below threshold | `"9:3"` |
| `--skip_layers_above` | Skip layers above threshold | `"9:13"` |
| `--disable_layer_skip` | Disable all layer skipping | Flag |

## 📊 VAR Stages and Layers

### Generation Stages
VAR uses 10 generation stages by default, corresponding to different patch sizes:

| Stage | Patch Size | Resolution | Tokens |
|-------|------------|------------|---------|
| 0 | 1×1 | 1×1 | 1 |
| 1 | 2×2 | 2×2 | 4 |
| 2 | 3×3 | 3×3 | 9 |
| 3 | 4×4 | 4×4 | 16 |
| 4 | 5×5 | 5×5 | 25 |
| 5 | 6×6 | 6×6 | 36 |
| 6 | 8×8 | 8×8 | 64 |
| 7 | 10×10 | 10×10 | 100 |
| 8 | 13×13 | 13×13 | 169 |
| 9 | 16×16 | 16×16 | 256 |

### Transformer Layers
For VAR-D16 (default), there are 16 transformer layers (indices 0-15).

## 💡 Example Use Cases

### 1. Skip Early Layers in Final Stage
Skip the first few layers in the final high-resolution stage:
```python
LayerSkipConfig(skip_all_layers_below={9: 3})  # Skip layers 0,1,2 in stage 9
```

### 2. Skip Late Layers in Multiple Stages
Skip the last few layers in later stages:
```python
LayerSkipConfig(stage_layer_skip={
    7: {14, 15},  # Skip layers 14,15 in stage 7
    8: {14, 15},  # Skip layers 14,15 in stage 8
    9: {14, 15}   # Skip layers 14,15 in stage 9
})
```

### 3. Progressive Layer Reduction
Gradually reduce layers in later stages:
```python
LayerSkipConfig(stage_layer_skip={
    6: {0},        # Skip layer 0 in stage 6
    7: {0, 1},     # Skip layers 0,1 in stage 7
    8: {0, 1, 2},  # Skip layers 0,1,2 in stage 8
    9: {0, 1, 2, 3} # Skip layers 0,1,2,3 in stage 9
})
```

## 🏃‍♂️ Performance Impact

Layer skipping can provide significant speedup with minimal quality loss:

- **Early layer skipping**: Affects feature extraction but may preserve high-level semantics
- **Late layer skipping**: Affects fine-grained details but preserves overall structure
- **Stage-specific skipping**: Allows targeted optimization for different resolutions

## 📈 Benchmarking

Run comprehensive benchmarks:
```bash
./run_layer_skip_evaluation.sh \
    --model_path /path/to/model.pth \
    --vae_path /path/to/vae.pth \
    benchmark
```

This will test multiple configurations and provide performance comparisons.

## 🔍 Monitoring and Logging

The implementation provides detailed logging:

```
Stage 6 (patch_size=8x8): skipped 1/16 layers
Stage 7 (patch_size=10x10): skipped 2/16 layers
Stage 8 (patch_size=13x13): skipped 3/16 layers
Stage 9 (patch_size=16x16): skipped 4/16 layers
Total layer skipping: 10/160 (6.2%)
```

## 🎛️ Python API Usage

```python
from models.var_layer_skip import VARLayerSkip, LayerSkipConfig

# Create layer skip configuration
config = LayerSkipConfig(
    skip_all_layers_below={9: 3},  # Skip layers 0,1,2 in stage 9
    enable_skip=True
)

# Build model with layer skipping
vae, var = build_vae_var_layer_skip(
    device='cuda',
    layer_skip_config=config
)

# Generate images
generated_images = var.autoregressive_infer_cfg(
    B=4,
    label_B=torch.tensor([281, 285, 287, 291]),  # ImageNet class IDs
    cfg=1.5
)
```

## 🔬 Research Applications

This implementation enables research into:

- **Computational efficiency**: Trade-off between speed and quality
- **Layer importance**: Understanding which layers contribute most at different resolutions
- **Progressive generation**: Studying how complexity requirements change across stages
- **Adaptive computation**: Dynamic layer selection based on content complexity

## ⚠️ Important Notes

1. **Compatibility**: Works with standard VAR checkpoints - no retraining required
2. **Memory**: Layer skipping reduces both computation and memory usage
3. **Quality**: Impact on generation quality depends on which layers and stages are skipped
4. **Validation**: Always validate results against baseline (no skipping) for your use case

## 🛠️ Troubleshooting

### Common Issues

1. **Import errors**: Ensure all dependencies are installed
2. **CUDA memory**: Reduce batch size if encountering OOM errors
3. **Configuration syntax**: Use proper format for stage:layer specifications

### Debug Mode

Add verbose logging:
```python
LayerSkipConfig(enable_skip=True)  # Will print detailed skip information
```

## 📝 Citation

If you use this layer skipping implementation in your research, please cite the original VAR paper and mention this enhancement.

## 🤝 Contributing

Feel free to extend this implementation with:
- Dynamic layer skipping based on content complexity
- Learning-based skip decisions
- Additional performance optimizations
- Quality preservation techniques