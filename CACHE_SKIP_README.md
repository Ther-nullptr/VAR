# VAR Layer Cache-Interpolation Skipping Implementation

这个实现为VAR (Visual AutoRegressive) 模型添加了基于缓存-插值策略的阶段特定层跳过功能。与直接跳过层不同，这种方法使用之前阶段的缓存特征并进行插值来替代某些层的计算。

## 🎯 核心特性

- **缓存-插值策略**: 使用缓存特征替代层计算，而非直接跳过
- **特征插值**: 自动插值缓存特征到目标分辨率
- **可配置混合**: 支持缓存特征与当前计算结果的混合
- **阶段特定控制**: 在特定生成阶段使用缓存特定层
- **性能监控**: 跟踪缓存使用率和生成速度

## 📁 新增文件

- `models/var_layer_skip_cache.py` - 带缓存-插值跳层的增强VAR模型
- `var_evaluate_cache_skip.py` - 支持缓存-插值跳层的评估脚本
- `run_cache_skip_evaluation.sh` - 便捷的bash脚本
- `CACHE_SKIP_README.md` - 本文档

## 🚀 快速开始

### 基本使用

```bash
# 在最后阶段(stage 9)对层0,1,2使用缓存
./run_cache_skip_evaluation.sh \
    --model_path /path/to/var_d16_new.pth \
    --vae_path /path/to/vae_ch160v4096z32.pth \
    cache-last-low

# 不使用任何缓存-跳层（基线）
./run_cache_skip_evaluation.sh \
    --model_path /path/to/var_d16_new.pth \
    --vae_path /path/to/vae_ch160v4096z32.pth \
    no-cache-skip
```

### 高级自定义配置

```bash
# 在多个阶段对特定层使用缓存
./run_cache_skip_evaluation.sh \
    --model_path /path/to/var_d16_new.pth \
    --vae_path /path/to/vae_ch160v4096z32.pth \
    --stage_layer_skip "8:0,1;9:0,1,2" \
    custom

# 对最后阶段阈值以下的所有层使用缓存，60%混合比率
./run_cache_skip_evaluation.sh \
    --model_path /path/to/var_d16_new.pth \
    --vae_path /path/to/vae_ch160v4096z32.pth \
    --skip_layers_below "9:3" \
    --cache_blend_ratio 0.6 \
    custom
```

## 🔧 缓存-插值机制

### 工作原理

1. **特征缓存**: 每层的输出特征在每个阶段都被缓存
2. **缓存检索**: 当某层需要"跳过"时，从之前阶段检索缓存特征
3. **特征插值**: 将缓存特征插值到当前阶段的分辨率
4. **可选混合**: 将缓存特征与当前计算结果按比例混合

### 插值策略

```python
def feature_interpolate(x, target_length: int, mode='bilinear'):
    """
    将特征插值到目标序列长度
    Args:
        x: 输入张量 [B, L, C]
        target_length: 目标序列长度
        mode: 插值模式
    Returns:
        插值后的张量 [B, target_length, C]
    """
```

### 混合策略

```python
# 混合缓存特征和当前计算结果
blend_ratio = self.layer_cache_skip_config.cache_blend_ratio
x = blend_ratio * cached_features + (1 - blend_ratio) * current_output
```

## 🔧 配置选项

### LayerCacheSkipConfig 参数

| 参数 | 类型 | 描述 | 示例 |
|------|------|------|------|
| `stage_layer_skip` | Dict[int, Set[int]] | 特定阶段使用缓存的特定层 | `{9: {0, 1, 2}}` |
| `skip_all_layers_below` | Dict[int, int] | 阈值以下所有层使用缓存 | `{9: 3}` (层0,1,2使用缓存) |
| `skip_all_layers_above` | Dict[int, int] | 阈值以上所有层使用缓存 | `{9: 13}` (层14,15使用缓存) |
| `cache_blend_ratio` | float | 缓存特征混合比率 | `0.8` (80%缓存+20%当前) |
| `interpolation_mode` | str | 插值模式 | `'bilinear'`, `'nearest'`, `'bicubic'` |
| `adaptive_cache_selection` | bool | 自适应缓存选择 | `True` |

### 命令行参数

| 参数 | 描述 | 示例 |
|------|------|------|
| `--stage_layer_skip` | 阶段特定层缓存 | `"9:0,1,2"` |
| `--skip_layers_below` | 阈值以下层缓存 | `"9:3"` |
| `--skip_layers_above` | 阈值以上层缓存 | `"9:13"` |
| `--cache_blend_ratio` | 混合比率 | `0.6` |
| `--interpolation_mode` | 插值模式 | `bilinear` |
| `--adaptive_cache_selection` | 自适应缓存选择 | 标志 |

## 💡 使用场景示例

### 1. 早期层缓存策略
在最终高分辨率阶段对前几层使用缓存：
```python
LayerCacheSkipConfig(skip_all_layers_below={9: 3})  # 阶段9的层0,1,2使用缓存
```

### 2. 多阶段后期层缓存
在后期阶段对最后几层使用缓存：
```python
LayerCacheSkipConfig(stage_layer_skip={
    7: {14, 15},  # 阶段7的层14,15使用缓存
    8: {14, 15},  # 阶段8的层14,15使用缓存
    9: {14, 15}   # 阶段9的层14,15使用缓存
})
```

### 3. 渐进式层缓存
在后期阶段逐渐增加缓存层数：
```python
LayerCacheSkipConfig(stage_layer_skip={
    6: {0},        # 阶段6的层0使用缓存
    7: {0, 1},     # 阶段7的层0,1使用缓存
    8: {0, 1, 2},  # 阶段8的层0,1,2使用缓存
    9: {0, 1, 2, 3} # 阶段9的层0,1,2,3使用缓存
})
```

### 4. 混合策略
使用60%缓存特征和40%当前计算：
```python
LayerCacheSkipConfig(
    skip_all_layers_below={9: 3},
    cache_blend_ratio=0.6
)
```

## 🏃‍♂️ 性能影响

缓存-插值策略相比直接跳层提供更好的质量保持：

- **早期层缓存**: 保持高级语义，可能影响细节特征提取
- **后期层缓存**: 保持整体结构，可能影响精细细节
- **混合策略**: 在计算效率和质量之间找到平衡
- **插值质量**: 双线性插值提供良好的特征平滑过渡

## 📈 基准测试

运行综合基准测试：
```bash
./run_cache_skip_evaluation.sh \
    --model_path /path/to/model.pth \
    --vae_path /path/to/vae.pth \
    compare
```

这将测试多种配置并提供性能对比。

## 🔍 监控和日志

实现提供详细的日志记录：

```
Stage 8 (patch_size=13x13): used cache for 2/16 layers
Stage 9 (patch_size=16x16): used cache for 3/16 layers
Total layer cache usage: 5/160 (3.1%)
```

## 🎛️ Python API 使用

```python
from models.var_layer_skip_cache import VARLayerSkipCache, LayerCacheSkipConfig

# 创建缓存-跳层配置
config = LayerCacheSkipConfig(
    skip_all_layers_below={9: 3},  # 阶段9的层0,1,2使用缓存
    cache_blend_ratio=0.8,         # 80%缓存混合
    interpolation_mode='bilinear', # 双线性插值
    enable_skip=True
)

# 构建带缓存-跳层的模型
vae, var = build_vae_var_cache_skip(
    device='cuda',
    layer_cache_skip_config=config
)

# 生成图像
generated_images = var.autoregressive_infer_cfg(
    B=4,
    label_B=torch.tensor([281, 285, 287, 291]),  # ImageNet类别ID
    cfg=1.5
)
```

## 🔬 与直接跳层的对比

| 策略 | 计算成本 | 质量保持 | 内存使用 | 适用场景 |
|------|----------|----------|----------|----------|
| 直接跳层 | 最低 | 较低 | 最低 | 极端加速需求 |
| 缓存-插值 | 中等 | 较高 | 中等 | 平衡质量和速度 |
| 混合策略 | 较高 | 最高 | 较高 | 质量优先场景 |

## ⚠️ 重要注意事项

1. **兼容性**: 与标准VAR检查点兼容 - 无需重新训练
2. **内存**: 缓存-插值策略会增加内存使用
3. **质量**: 对生成质量的影响取决于缓存的层和阶段
4. **插值精度**: 双线性插值对大多数情况效果良好
5. **混合比率**: 较高的混合比率偏向缓存特征，较低的比率保持更多当前计算

## 🛠️ 故障排除

### 常见问题

1. **导入错误**: 确保所有依赖项已安装
2. **CUDA内存**: 如遇OOM错误，减少批大小
3. **插值质量**: 尝试不同的插值模式
4. **混合效果**: 调整cache_blend_ratio找到最佳平衡

### 调试模式

添加详细日志：
```python
LayerCacheSkipConfig(enable_skip=True)  # 将打印详细的缓存使用信息
```

## 🤝 贡献

欢迎扩展此实现：
- 基于内容复杂度的动态缓存选择
- 学习基于的缓存决策
- 额外的插值策略
- 质量保持技术

## 📝 引用

如果您在研究中使用此缓存-插值跳层实现，请引用原始VAR论文并提及此增强功能。