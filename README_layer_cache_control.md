# VAR 细粒度层缓存控制 (Fine-Grained Layer Cache Control)

## 概述

VAR细粒度层缓存控制是一个新的功能，允许用户在生成阶段级别和Transformer层级别上精确控制缓存策略。相比于原有的阶段级缓存策略，这个新功能提供了更加精细的控制，可以指定特定阶段的特定层是否使用缓存-插值策略。

## 主要特性

### 🎯 细粒度控制
- **阶段+层级控制**: 同时指定生成阶段(stage)和Transformer层(layer)
- **多种控制模式**: `cache`, `skip`, `normal` 三种模式
- **独立的注意力/MLP控制**: 分别控制注意力层和MLP层的缓存
- **混合比例**: 支持缓存特征和计算特征的平滑插值 (0.0=纯计算, 1.0=纯缓存)
- **多种插值模式**: `linear`, `bilinear`, `bicubic` 特征上采样

### 🚀 兼容性
- 与原有阶段级缓存策略完全兼容
- 支持混合使用细粒度控制和传统控制
- 无需修改原有模型文件

## 文件结构

```
VAR/
├── models/
│   ├── var_layer_cache_control.py      # 主要实现文件
│   └── basic_var_layer_control.py      # 增强的Transformer块
├── var_evaluate_layer_cache_control.py # 评估脚本
├── run_layer_cache_control.sh          # Bash脚本
├── test_layer_cache_control.py         # 测试脚本
└── README_layer_cache_control.md       # 本文档
```

## 快速开始

### 1. 基本使用

```bash
./run_layer_cache_control.sh \
  --model_path /path/to/var.pth \
  --vae_path /path/to/vae.pth \
  --layer_cache_spec "9:0:cache,1:cache,2:cache" \
  --num_samples 1000
```

### 2. 高级控制

```bash
./run_layer_cache_control.sh \
  --model_path /path/to/var.pth \
  --vae_path /path/to/vae.pth \
  --layer_cache_spec "9:0:cache,1:cache,2:cache;8:0:cache,15:cache" \
  --layer_blend_ratios "9:0:0.8,1:0.6,2:0.9;8:0:0.7" \
  --interpolation_mode bicubic
```

## 详细参数说明

### 核心参数

#### `--layer_cache_spec`
细粒度层缓存规范，指定哪些阶段的哪些层使用何种缓存模式。

**格式**: `"stage1:layer1:mode1,layer2:mode2;stage2:layer3:mode3"`

**示例**: `"9:0:cache,1:cache,2:cache;8:0:cache,15:cache"`

**模式说明**:
- `cache`: 使用缓存-插值策略
- `skip`: 跳过该层计算(直接使用缓存)
- `normal`: 正常计算(不使用缓存)

#### `--cache_attn_layers`
指定哪些阶段的哪些层使用注意力缓存。

**格式**: `"stage1:layer1,layer2;stage2:layer3,layer4"`

**示例**: `"9:0,1,2;8:0,15"`

#### `--cache_mlp_layers`
指定哪些阶段的哪些层使用MLP缓存。

**格式**: `"stage1:layer1,layer2;stage2:layer3,layer4"`

**示例**: `"9:1,2;8:15"`

#### `--layer_blend_ratios`
层级混合比例，控制缓存特征和计算特征的混合程度。

**格式**: `"stage1:layer1:ratio1,layer2:ratio2;stage2:layer3:ratio3"`

**示例**: `"9:0:0.8,1:0.6,2:0.9;8:0:0.7"`

**比例说明**:
- `0.0`: 完全使用计算特征(无缓存)
- `1.0`: 完全使用缓存特征
- `0.5`: 50%缓存 + 50%计算的混合

#### `--interpolation_mode`
特征上采样的插值模式。

**选项**: `linear`, `bilinear`, `bicubic`

**默认**: `bilinear`

### 阶段和层信息

#### VAR阶段对应关系
```
阶段 0: 1x1   (L=1)      阶段 5: 6x6   (L=36)
阶段 1: 2x2   (L=4)      阶段 6: 8x8   (L=64)  
阶段 2: 3x3   (L=9)      阶段 7: 10x10 (L=100)
阶段 3: 4x4   (L=16)     阶段 8: 13x13 (L=169)
阶段 4: 5x5   (L=25)     阶段 9: 16x16 (L=256)
```

#### Transformer层信息
- VAR通常有16个Transformer层 (索引0-15)
- 每层包含注意力(attention)和MLP组件
- 细粒度控制允许分别指定注意力和MLP的缓存策略

## 使用场景和示例

### 场景1: 加速最高分辨率阶段
在最高分辨率阶段(stage 9)缓存前几层以提升速度：

```bash
./run_layer_cache_control.sh \
  --model_path /path/to/var.pth \
  --vae_path /path/to/vae.pth \
  --layer_cache_spec "9:0:cache,1:cache,2:cache" \
  --num_samples 5000
```

### 场景2: 多阶段优化
在多个关键阶段进行层级缓存：

```bash
./run_layer_cache_control.sh \
  --model_path /path/to/var.pth \
  --vae_path /path/to/vae.pth \
  --layer_cache_spec "9:0:cache,1:cache;8:0:cache,15:cache;7:10:cache" \
  --cache_attn_layers "9:0,1;8:0,15" \
  --cache_mlp_layers "9:1;8:15"
```

### 场景3: 质量-性能平衡
使用混合比例来平衡生成质量和速度：

```bash
./run_layer_cache_control.sh \
  --model_path /path/to/var.pth \
  --vae_path /path/to/vae.pth \
  --layer_cache_spec "9:0:cache,1:cache,2:cache" \
  --layer_blend_ratios "9:0:0.8,1:0.6,2:0.9" \
  --interpolation_mode bicubic
```

### 场景4: 与传统缓存混合使用
结合传统阶段级缓存和新的细粒度控制：

```bash
./run_layer_cache_control.sh \
  --model_path /path/to/var.pth \
  --vae_path /path/to/vae.pth \
  --layer_cache_spec "9:0:cache;8:15:cache" \
  --skip_stages "169,256" \
  --cache_stages "100,169"
```

### 场景5: FID评估
生成图像并计算FID分数：

```bash
./run_layer_cache_control.sh \
  --model_path /path/to/var.pth \
  --vae_path /path/to/vae.pth \
  --layer_cache_spec "9:0:cache,1:cache,2:cache" \
  --compute_fid true \
  --real_img_dir /path/to/real/images \
  --num_samples 10000
```

## Python API 使用

### 直接使用Python脚本

```python
python var_evaluate_layer_cache_control.py \
  --model_path /path/to/var.pth \
  --vae_path /path/to/vae.pth \
  --layer_cache_spec "9:0:cache,1:cache,2:cache" \
  --layer_blend_ratios "9:0:0.8,1:0.6" \
  --num_samples 1000
```

### 在代码中集成

```python
from models.var_layer_cache_control import VARLayerCacheControl, parse_layer_cache_spec
from models.basic_var_layer_control import LayerCacheConfig

# 解析缓存规范
stage_layer_cache_control = parse_layer_cache_spec("9:0:cache,1:cache,2:cache")

# 创建配置
layer_cache_config = LayerCacheConfig(
    stage_layer_cache_control=stage_layer_cache_control,
    interpolation_mode='bilinear'
)

# 创建模型
var_model = VARLayerCacheControl(
    vae_local=vae,
    layer_cache_config=layer_cache_config
)

# 生成图像
images = var_model.autoregressive_infer_cfg(
    B=batch_size,
    label_B=labels,
    cfg=4.0,
    top_k=900,
    top_p=0.96
)
```

## 性能优化建议

### 1. 选择合适的缓存层
- **前几层**: 通常包含底层特征，缓存影响较小
- **中间层**: 平衡质量和性能
- **后几层**: 包含高级语义，谨慎缓存

### 2. 使用混合比例
- 开始时使用较高比例 (0.8-0.9)
- 根据质量评估调整比例
- 关键层使用较低比例保持质量

### 3. 插值模式选择
- `bilinear`: 默认选择，平衡质量和速度
- `bicubic`: 更好的质量，略慢
- `linear`: 最快，适用于简单场景

### 4. 分阶段测试
```bash
# 测试单阶段效果
--layer_cache_spec "9:0:cache"

# 逐步增加缓存层
--layer_cache_spec "9:0:cache,1:cache"

# 添加混合比例优化
--layer_blend_ratios "9:0:0.8,1:0.7"
```

## 故障排除

### 常见错误

#### 1. 规范格式错误
```bash
# 错误
--layer_cache_spec "9-0-cache"

# 正确
--layer_cache_spec "9:0:cache"
```

#### 2. 阶段/层索引超出范围
```bash
# 检查模型的实际层数
python -c "
import torch
model = torch.load('var.pth')
print('Layers:', len(model['model']['backbone']))
"
```

#### 3. 内存不足
- 减少batch_size
- 减少缓存层数量
- 使用更低的混合比例

### 调试技巧

#### 1. 启用详细日志
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### 2. 检查缓存配置
```python
print(layer_cache_config.get_cache_summary(10, 16))
```

#### 3. 性能监控
```bash
# 使用时间测量
time ./run_layer_cache_control.sh [参数]

# 监控GPU使用
nvidia-smi -l 1
```

## API 参考

### LayerCacheConfig 类

```python
class LayerCacheConfig:
    def __init__(
        self,
        skip_stages: List[int] = None,
        cache_stages: List[int] = None,
        enable_attn_cache: bool = True,
        enable_mlp_cache: bool = True,
        threshold: float = 0.7,
        max_skip_stages: int = 9,
        adaptive_threshold: bool = False,
        interpolation_mode: str = 'bilinear',
        stage_layer_cache_control: Dict[int, Dict[int, str]] = None,
        cache_attn_layers: Dict[int, Set[int]] = None,
        cache_mlp_layers: Dict[int, Set[int]] = None,
        layer_blend_ratios: Dict[Tuple[int, int], float] = None
    )
```

### 主要方法

#### `should_cache_layer_attn(stage_idx, layer_idx, L)`
检查指定阶段和层是否应该缓存注意力。

#### `should_cache_layer_mlp(stage_idx, layer_idx, L)`
检查指定阶段和层是否应该缓存MLP。

#### `get_layer_blend_ratio(stage_idx, layer_idx)`
获取指定阶段和层的混合比例。

#### `get_cache_summary(num_layers, num_stages)`
获取缓存配置的摘要信息。

### 解析函数

#### `parse_layer_cache_spec(spec: str)`
解析层缓存规范字符串。

#### `parse_layer_set_spec(spec: str)`
解析层集合规范字符串。

#### `parse_layer_blend_spec(spec: str)`
解析层混合比例规范字符串。

## 更新日志

### v1.0.0 (当前版本)
- ✅ 实现细粒度层缓存控制
- ✅ 支持阶段+层级规范
- ✅ 独立的注意力/MLP控制
- ✅ 混合比例支持
- ✅ 多种插值模式
- ✅ 完整的命令行界面
- ✅ 全面的测试和文档

## 贡献

欢迎提交Issue和Pull Request来改进这个功能。

## 许可证

遵循原VAR项目的许可证。