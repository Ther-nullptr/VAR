# Enhanced VAR Multi-Stage Caching System

This enhanced implementation extends the original VAR caching mechanism to support flexible multi-stage caching with separate control over attention and MLP layers.

## 🚀 Key Features

- **Flexible Stage Control**: Support for skipping any combination of 1-9 stages
- **Layer-Specific Caching**: Independent control over attention and MLP layer caching  
- **Adaptive Thresholding**: Dynamic threshold adjustment based on similarity patterns
- **Multiple Interpolation Modes**: Bilinear, nearest, and bicubic upsampling
- **Comprehensive Benchmarking**: Performance comparison across different configurations
- **Easy Configuration**: Command-line interface with preset configurations

## 📁 File Structure

```
VAR/
├── models/
│   ├── basic_var_enhanced.py      # Enhanced transformer blocks
│   └── var_enhanced.py            # Enhanced VAR model
├── var_evaluate_enhanced.py       # Main evaluation script
├── run_enhanced_evaluation.sh     # Convenient shell wrapper
├── cache_configs.json             # Configuration presets
└── ENHANCED_CACHING_README.md     # This file
```

## 🔧 Installation

1. Ensure you have the base VAR repository set up
2. The enhanced files are standalone and don't modify existing files
3. No additional dependencies beyond the original VAR requirements

## 🚀 Quick Start

### 1. Basic Usage

```bash
# Original VAR caching
./run_enhanced_evaluation.sh --model_path model.pth --vae_path vae.pth original-cache

# No caching (baseline)
./run_enhanced_evaluation.sh --model_path model.pth --vae_path vae.pth no-cache

# Aggressive caching for maximum speed
./run_enhanced_evaluation.sh --model_path model.pth --vae_path vae.pth aggressive-cache
```

### 2. Custom Configuration

```bash
# Custom skip and cache stages
./run_enhanced_evaluation.sh --model_path model.pth --vae_path vae.pth \
    --skip_stages "100,169,256" --cache_stages "64,100,169" custom

# MLP-only caching
./run_enhanced_evaluation.sh --model_path model.pth --vae_path vae.pth mlp-only-cache

# Attention-only caching  
./run_enhanced_evaluation.sh --model_path model.pth --vae_path vae.pth attn-only-cache
```

### 3. Benchmarking and Comparison

```bash
# Compare multiple configurations
./run_enhanced_evaluation.sh --model_path model.pth --vae_path vae.pth compare

# Run calibration to optimize thresholds
./run_enhanced_evaluation.sh --model_path model.pth --vae_path vae.pth calibrate

# Speed benchmark only
./run_enhanced_evaluation.sh --model_path model.pth --vae_path vae.pth benchmark
```

## ⚙️ Configuration Options

### Cache Configuration Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `skip_stages` | List[int] | Stages to skip computation | `[]` |
| `cache_stages` | List[int] | Stages to cache results | `[]` |
| `enable_attn_cache` | bool | Enable attention caching | `True` |
| `enable_mlp_cache` | bool | Enable MLP caching | `True` |
| `threshold` | float | Similarity threshold | `0.7` |
| `max_skip_stages` | int | Maximum stages to skip | `9` |
| `adaptive_threshold` | bool | Use adaptive thresholding | `False` |
| `interpolation_mode` | str | Upsampling mode | `'bilinear'` |

### Valid Stage Numbers

The system supports the following resolution stages:

| Stage | Resolution | Typical Use Case |
|-------|------------|------------------|
| 1 | 1×1 | Always compute (initial) |
| 4 | 2×2 | Rarely cached |
| 9 | 3×3 | Rarely cached |
| 16 | 4×4 | Sometimes cached |
| 25 | 5×5 | Sometimes cached |
| 36 | 6×6 | Good for caching |
| 64 | 8×8 | Excellent for caching |
| 100 | 10×10 | Excellent for caching |
| 169 | 13×13 | Prime for caching |
| 256 | 16×16 | Often skipped |

## 🎛️ Preset Configurations

### Performance-Oriented Presets

```bash
# Ultra-fast generation (maximum speedup)
./run_enhanced_evaluation.sh --model_path model.pth --vae_path vae.pth \
    --skip_stages "36,64,100,169,256" --cache_stages "25,36,64,100,169" \
    --threshold 0.5 --adaptive_threshold custom

# Aggressive caching (good speedup)  
./run_enhanced_evaluation.sh --model_path model.pth --vae_path vae.pth aggressive-cache
```

### Quality-Oriented Presets

```bash
# High quality (minimal caching)
./run_enhanced_evaluation.sh --model_path model.pth --vae_path vae.pth \
    --skip_stages "256" --cache_stages "169" --threshold 0.9 custom

# Conservative caching
./run_enhanced_evaluation.sh --model_path model.pth --vae_path vae.pth conservative-cache
```

### Memory-Oriented Presets

```bash
# Memory efficient (attention-only)
./run_enhanced_evaluation.sh --model_path model.pth --vae_path vae.pth attn-only-cache

# Minimal memory footprint
./run_enhanced_evaluation.sh --model_path model.pth --vae_path vae.pth \
    --skip_stages "169,256" --cache_stages "169" --disable_mlp_cache custom
```

## 📊 Performance Analysis

### Benchmark Results Format

The evaluation script generates comprehensive performance metrics:

```json
{
  "benchmark": {
    "mean_time": 2.35,
    "std_time": 0.12,  
    "mean_memory_mb": 8420.5,
    "throughput_img_per_sec": 6.81
  },
  "comparison": {
    "no_cache": {"performance": {...}},
    "original": {"performance": {...}},
    "aggressive": {"performance": {...}}
  }
}
```

### Expected Performance Improvements

| Configuration | Speedup | Memory | Quality |
|---------------|---------|---------|---------|
| No Cache | 1.0× | Baseline | Best |
| Conservative | 1.2-1.5× | +5% | Excellent |
| Original VAR | 1.5-2.0× | +10% | Very Good |
| Aggressive | 2.0-3.0× | +15% | Good |
| Ultra Fast | 3.0-4.0× | +20% | Acceptable |

## 🔬 Advanced Features

### 1. Adaptive Thresholding

```python
# Enable adaptive thresholding
cache_config = CacheConfig(
    skip_stages=[169, 256],
    cache_stages=[100, 169],
    adaptive_threshold=True,
    threshold=0.7  # Initial threshold
)
```

The system tracks similarity patterns and adjusts the threshold to the 80th percentile of recent similarities.

### 2. Calibration Mode

```bash
# Run calibration to optimize thresholds
./run_enhanced_evaluation.sh --model_path model.pth --vae_path vae.pth calibrate
```

Calibration mode:
- Generates samples while computing both cached and non-cached results
- Tracks cosine similarity between interpolated cache and computed features
- Saves similarity statistics for optimal threshold selection

### 3. Layer-Specific Control

```python
# Attention-only caching
cache_config = CacheConfig(
    skip_stages=[169, 256],
    cache_stages=[100, 169], 
    enable_attn_cache=True,
    enable_mlp_cache=False
)

# MLP-only caching
cache_config = CacheConfig(
    skip_stages=[169, 256],
    cache_stages=[100, 169],
    enable_attn_cache=False, 
    enable_mlp_cache=True
)
```

### 4. Multiple Interpolation Modes

```python
cache_config = CacheConfig(
    interpolation_mode='bicubic'  # 'bilinear', 'nearest', 'bicubic'
)
```

- **Bilinear**: Fastest, good quality (default)
- **Nearest**: Fastest, lower quality
- **Bicubic**: Slower, best quality

## 🛠️ Integration Guide

### Using Enhanced VAR in Your Code

```python
from models.var_enhanced import VAREnhanced, CacheConfig

# Create cache configuration
cache_config = CacheConfig(
    skip_stages=[169, 256],
    cache_stages=[100, 169],
    enable_attn_cache=True,
    enable_mlp_cache=True,
    threshold=0.7
)

# Load model
model = VAREnhanced.from_pretrained('path/to/model')
model.set_cache_config(cache_config)

# Generate with caching
images = model.autoregressive_infer_cfg(
    B=16, label_B=None, g_seed=42, cfg=1.5
)
```

### Configuration from JSON

```python
import json
from models.basic_var_enhanced import CacheConfig

# Load preset from JSON
with open('cache_configs.json', 'r') as f:
    configs = json.load(f)

preset = configs['presets']['aggressive']
cache_config = CacheConfig(**preset)
```

## 🎯 Tuning Guidelines

### For Maximum Speed
- Use aggressive skip stages: `[64, 100, 169, 256]`
- Lower threshold: `0.5-0.6`
- Enable both attention and MLP caching
- Consider adaptive thresholding

### For Best Quality
- Minimal skip stages: `[256]` only
- Higher threshold: `0.8-0.9` 
- Use bicubic interpolation
- Conservative cache stages

### For Memory Efficiency
- Attention-only caching
- Fewer cache stages
- Higher threshold to reduce cache usage

### For Balanced Performance
- Original VAR configuration: skip `[169, 256]`, cache `[100, 169]`
- Threshold: `0.7`
- Both attention and MLP caching

## 📈 Monitoring and Debugging

### Cache Statistics

```python
# Get cache usage statistics
stats = model.get_cache_statistics()
print("Attention similarities:", stats['attn_similarities'])
print("MLP similarities:", stats['mlp_similarities'])

if model.cache_config.adaptive_threshold:
    print("Effective thresholds:", stats['effective_thresholds'])
```

### Similarity Data Management

```python
# Save calibrated similarity data
model.save_similarity_data('similarity_cache.pth')

# Load pre-computed similarity data  
model.load_similarity_data('similarity_cache.pth')
```

## ❗ Limitations and Considerations

### Quality vs Speed Trade-off
- More aggressive caching increases speed but may reduce output quality
- Higher resolution stages (169, 256) are most beneficial to skip but also most impactful on quality

### Memory Usage
- Caching increases memory usage proportional to number of cached stages
- Consider memory constraints when selecting cache stages

### Interpolation Artifacts
- Feature interpolation may introduce slight artifacts
- Higher quality interpolation modes reduce artifacts but increase computation

### Adaptive Threshold Convergence
- Adaptive thresholding requires calibration period to converge
- Performance may be suboptimal during initial calibration

## 🤝 Contributing

To extend the caching system:

1. **Add New Interpolation Modes**: Extend `feature_interpolate()` function
2. **Custom Similarity Metrics**: Modify similarity computation in enhanced blocks  
3. **Advanced Scheduling**: Implement stage-specific threshold schedules
4. **Memory Optimization**: Add cache eviction strategies

## 📚 References

- Original VAR Paper: [Visual Autoregressive Modeling](https://arxiv.org/abs/2404.02905)
- FastVAR Implementation: [Post-training Speedup](https://github.com/csguoh/FastVAR)
- Feature Interpolation: Standard PyTorch interpolation methods

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size or number of cache stages
   - Use attention-only caching
   - Consider gradient checkpointing

2. **Quality Degradation**  
   - Increase similarity threshold
   - Use fewer skip stages
   - Try bicubic interpolation

3. **No Speedup Observed**
   - Ensure skip stages are computationally expensive (e.g., 169, 256)
   - Check that cache stages provide good features
   - Run calibration to optimize thresholds

4. **Memory Leaks**
   - Cache tensors are automatically cleared between stages
   - Check for retention of large intermediate tensors

### Performance Debugging

```bash
# Profile memory usage
./run_enhanced_evaluation.sh --model_path model.pth --vae_path vae.pth \
    --batch_size 4 benchmark

# Compare configurations systematically  
./run_enhanced_evaluation.sh --model_path model.pth --vae_path vae.pth compare
```

This enhanced caching system provides fine-grained control over the speed-quality trade-off in VAR models, enabling efficient deployment across different computational constraints and quality requirements.