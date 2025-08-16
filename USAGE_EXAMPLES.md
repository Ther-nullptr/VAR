# VAR 细粒度层缓存控制使用示例

本文档提供了各种使用场景的详细示例和最佳实践。

## 📚 目录

1. [基础使用示例](#基础使用示例)
2. [高级控制示例](#高级控制示例)
3. [性能优化示例](#性能优化示例)
4. [质量评估示例](#质量评估示例)
5. [故障排除示例](#故障排除示例)
6. [批量实验示例](#批量实验示例)

## 基础使用示例

### 1. 简单的层缓存

最基本的使用方式，在最高分辨率阶段缓存前3层：

```bash
./run_layer_cache_control.sh \
  --model_path ./checkpoints/var_d16.pth \
  --vae_path ./checkpoints/vae_ch256v4096z32.pth \
  --layer_cache_spec "9:0:cache,1:cache,2:cache" \
  --num_samples 1000 \
  --batch_size 50
```

**预期效果**: 在stage 9(16x16分辨率)缓存前3层，应该有约10-15%的速度提升。

### 2. 多阶段缓存

在多个阶段进行缓存：

```bash
./run_layer_cache_control.sh \
  --model_path ./checkpoints/var_d16.pth \
  --vae_path ./checkpoints/vae_ch256v4096z32.pth \
  --layer_cache_spec "9:0:cache,1:cache;8:0:cache,15:cache" \
  --num_samples 5000 \
  --batch_size 100
```

**说明**: 
- Stage 9: 缓存层0和层1
- Stage 8: 缓存层0和层15
- 预期20-30%速度提升

### 3. 分离的注意力和MLP控制

```bash
./run_layer_cache_control.sh \
  --model_path ./checkpoints/var_d16.pth \
  --vae_path ./checkpoints/vae_ch256v4096z32.pth \
  --cache_attn_layers "9:0,1,2;8:0,15" \
  --cache_mlp_layers "9:1,2;8:15" \
  --num_samples 2000
```

**说明**: 
- 注意力缓存: Stage 9的层0,1,2 + Stage 8的层0,15
- MLP缓存: Stage 9的层1,2 + Stage 8的层15
- 更精细的控制，可能有更好的质量保持

## 高级控制示例

### 4. 使用混合比例

通过混合比例来平衡质量和性能：

```bash
./run_layer_cache_control.sh \
  --model_path ./checkpoints/var_d16.pth \
  --vae_path ./checkpoints/vae_ch256v4096z32.pth \
  --layer_cache_spec "9:0:cache,1:cache,2:cache,3:cache" \
  --layer_blend_ratios "9:0:0.9,1:0.8,2:0.7,3:0.6" \
  --interpolation_mode bicubic \
  --num_samples 1000
```

**说明**: 
- 层0: 90%缓存 + 10%计算
- 层1: 80%缓存 + 20%计算  
- 层2: 70%缓存 + 30%计算
- 层3: 60%缓存 + 40%计算
- 使用bicubic插值获得更好的质量

### 5. 复杂的混合策略

```bash
./run_layer_cache_control.sh \
  --model_path ./checkpoints/var_d16.pth \
  --vae_path ./checkpoints/vae_ch256v4096z32.pth \
  --layer_cache_spec "9:0:cache,1:cache,2:cache;8:0:cache,14:cache,15:cache;7:10:cache" \
  --layer_blend_ratios "9:0:0.85,1:0.75,2:0.65;8:0:0.8,14:0.7,15:0.6;7:10:0.9" \
  --cache_attn_layers "9:0,1,2;8:0,14,15;7:10" \
  --cache_mlp_layers "9:1,2;8:14,15;7:10" \
  --interpolation_mode bilinear \
  --num_samples 3000
```

**说明**: 跨越3个阶段的复杂缓存策略，适用于需要最大性能提升的场景。

### 6. 与传统缓存结合

```bash
./run_layer_cache_control.sh \
  --model_path ./checkpoints/var_d16.pth \
  --vae_path ./checkpoints/vae_ch256v4096z32.pth \
  --layer_cache_spec "9:0:cache,1:cache;8:15:cache" \
  --skip_stages "169,256" \
  --cache_stages "100,169" \
  --enable_attn_cache true \
  --enable_mlp_cache true \
  --threshold 0.8 \
  --num_samples 2000
```

**说明**: 结合新的细粒度控制和原有的阶段级缓存策略。

## 性能优化示例

### 7. 渐进式缓存测试

逐步增加缓存层数来找到最佳平衡点：

```bash
# 测试1: 只缓存1层
./run_layer_cache_control.sh \
  --model_path ./checkpoints/var_d16.pth \
  --vae_path ./checkpoints/vae_ch256v4096z32.pth \
  --layer_cache_spec "9:0:cache" \
  --num_samples 500 \
  --output_dir test_1layer

# 测试2: 缓存2层
./run_layer_cache_control.sh \
  --model_path ./checkpoints/var_d16.pth \
  --vae_path ./checkpoints/vae_ch256v4096z32.pth \
  --layer_cache_spec "9:0:cache,1:cache" \
  --num_samples 500 \
  --output_dir test_2layers

# 测试3: 缓存3层
./run_layer_cache_control.sh \
  --model_path ./checkpoints/var_d16.pth \
  --vae_path ./checkpoints/vae_ch256v4096z32.pth \
  --layer_cache_spec "9:0:cache,1:cache,2:cache" \
  --num_samples 500 \
  --output_dir test_3layers
```

### 8. 混合比例优化

测试不同混合比例的效果：

```bash
# 高缓存比例 (更快，可能质量略降)
./run_layer_cache_control.sh \
  --model_path ./checkpoints/var_d16.pth \
  --vae_path ./checkpoints/vae_ch256v4096z32.pth \
  --layer_cache_spec "9:0:cache,1:cache,2:cache" \
  --layer_blend_ratios "9:0:0.9,1:0.9,2:0.9" \
  --num_samples 1000 \
  --output_dir high_cache_ratio

# 中等缓存比例 (平衡)
./run_layer_cache_control.sh \
  --model_path ./checkpoints/var_d16.pth \
  --vae_path ./checkpoints/vae_ch256v4096z32.pth \
  --layer_cache_spec "9:0:cache,1:cache,2:cache" \
  --layer_blend_ratios "9:0:0.7,1:0.7,2:0.7" \
  --num_samples 1000 \
  --output_dir medium_cache_ratio

# 低缓存比例 (更慢，更好质量)
./run_layer_cache_control.sh \
  --model_path ./checkpoints/var_d16.pth \
  --vae_path ./checkpoints/vae_ch256v4096z32.pth \
  --layer_cache_spec "9:0:cache,1:cache,2:cache" \
  --layer_blend_ratios "9:0:0.5,1:0.5,2:0.5" \
  --num_samples 1000 \
  --output_dir low_cache_ratio
```

### 9. 插值模式比较

```bash
# Linear插值 (最快)
./run_layer_cache_control.sh \
  --model_path ./checkpoints/var_d16.pth \
  --vae_path ./checkpoints/vae_ch256v4096z32.pth \
  --layer_cache_spec "9:0:cache,1:cache,2:cache" \
  --interpolation_mode linear \
  --num_samples 1000 \
  --output_dir linear_interp

# Bilinear插值 (默认)
./run_layer_cache_control.sh \
  --model_path ./checkpoints/var_d16.pth \
  --vae_path ./checkpoints/vae_ch256v4096z32.pth \
  --layer_cache_spec "9:0:cache,1:cache,2:cache" \
  --interpolation_mode bilinear \
  --num_samples 1000 \
  --output_dir bilinear_interp

# Bicubic插值 (最好质量)
./run_layer_cache_control.sh \
  --model_path ./checkpoints/var_d16.pth \
  --vae_path ./checkpoints/vae_ch256v4096z32.pth \
  --layer_cache_spec "9:0:cache,1:cache,2:cache" \
  --interpolation_mode bicubic \
  --num_samples 1000 \
  --output_dir bicubic_interp
```

## 质量评估示例

### 10. FID评估

生成图像并计算FID分数：

```bash
./run_layer_cache_control.sh \
  --model_path ./checkpoints/var_d16.pth \
  --vae_path ./checkpoints/vae_ch256v4096z32.pth \
  --layer_cache_spec "9:0:cache,1:cache,2:cache" \
  --layer_blend_ratios "9:0:0.8,1:0.7,2:0.6" \
  --compute_fid true \
  --real_img_dir ./datasets/imagenet_val \
  --num_samples 10000 \
  --batch_size 50 \
  --output_dir fid_evaluation
```

### 11. 批量FID测试

测试不同配置的FID分数：

```bash
#!/bin/bash

# 定义测试配置
configs=(
  "baseline:::" 
  "single_layer:9:0:cache::"
  "three_layers:9:0:cache,1:cache,2:cache::"
  "with_blend:9:0:cache,1:cache,2:cache:9:0:0.8,1:0.7,2:0.6:"
  "multi_stage:9:0:cache,1:cache;8:0:cache,15:cache::"
)

for config in "${configs[@]}"; do
  IFS=':' read -r name layer_spec blend_ratios <<< "$config"
  
  echo "Testing configuration: $name"
  
  cmd="./run_layer_cache_control.sh \
    --model_path ./checkpoints/var_d16.pth \
    --vae_path ./checkpoints/vae_ch256v4096z32.pth \
    --compute_fid true \
    --real_img_dir ./datasets/imagenet_val \
    --num_samples 5000 \
    --batch_size 50 \
    --output_dir fid_test_$name"
  
  if [ ! -z "$layer_spec" ]; then
    cmd="$cmd --layer_cache_spec '$layer_spec'"
  fi
  
  if [ ! -z "$blend_ratios" ]; then
    cmd="$cmd --layer_blend_ratios '$blend_ratios'"
  fi
  
  echo "Running: $cmd"
  eval $cmd
  
  echo "Completed: $name"
  echo "------------------------"
done
```

### 12. 大规模生成

```bash
./run_layer_cache_control.sh \
  --model_path ./checkpoints/var_d16.pth \
  --vae_path ./checkpoints/vae_ch256v4096z32.pth \
  --layer_cache_spec "9:0:cache,1:cache,2:cache,3:cache;8:0:cache,14:cache,15:cache" \
  --layer_blend_ratios "9:0:0.85,1:0.8,2:0.75,3:0.7;8:0:0.8,14:0.75,15:0.7" \
  --num_samples 50000 \
  --batch_size 200 \
  --save_images true \
  --save_npy true \
  --output_dir large_scale_generation
```

## 故障排除示例

### 13. 内存优化

如果遇到内存不足问题：

```bash
# 减少batch size
./run_layer_cache_control.sh \
  --model_path ./checkpoints/var_d16.pth \
  --vae_path ./checkpoints/vae_ch256v4096z32.pth \
  --layer_cache_spec "9:0:cache,1:cache" \
  --num_samples 1000 \
  --batch_size 20 \
  --output_dir memory_optimized

# 减少缓存层数
./run_layer_cache_control.sh \
  --model_path ./checkpoints/var_d16.pth \
  --vae_path ./checkpoints/vae_ch256v4096z32.pth \
  --layer_cache_spec "9:0:cache" \
  --num_samples 1000 \
  --batch_size 50 \
  --output_dir reduced_cache
```

### 14. 调试模式

启用详细输出进行调试：

```bash
# Python脚本直接调用，便于看到详细错误信息
python var_evaluate_layer_cache_control.py \
  --model_path ./checkpoints/var_d16.pth \
  --vae_path ./checkpoints/vae_ch256v4096z32.pth \
  --layer_cache_spec "9:0:cache,1:cache,2:cache" \
  --num_samples 100 \
  --batch_size 10 \
  --output_dir debug_test
```

### 15. 配置验证

测试配置是否正确：

```bash
# 运行测试脚本
python test_layer_cache_control.py

# 验证特定配置
python -c "
from models.var_layer_cache_control import parse_layer_cache_spec
result = parse_layer_cache_spec('9:0:cache,1:cache,2:cache')
print('Parsed config:', result)
"
```

## 批量实验示例

### 16. 自动化参数搜索

```bash
#!/bin/bash

# 参数搜索脚本
layer_configs=(
  "9:0:cache"
  "9:0:cache,1:cache"
  "9:0:cache,1:cache,2:cache"
  "9:0:cache,1:cache,2:cache,3:cache"
)

blend_ratios=(
  ""
  "9:0:0.9,1:0.9,2:0.9"
  "9:0:0.8,1:0.7,2:0.6"
  "9:0:0.7,1:0.6,2:0.5"
)

interp_modes=("linear" "bilinear" "bicubic")

for layer_config in "${layer_configs[@]}"; do
  for blend_ratio in "${blend_ratios[@]}"; do
    for interp_mode in "${interp_modes[@]}"; do
      
      # 创建实验名称
      exp_name="layers_$(echo $layer_config | tr ':,' '_')"
      if [ ! -z "$blend_ratio" ]; then
        exp_name="${exp_name}_blend"
      fi
      exp_name="${exp_name}_${interp_mode}"
      
      echo "Running experiment: $exp_name"
      
      cmd="./run_layer_cache_control.sh \
        --model_path ./checkpoints/var_d16.pth \
        --vae_path ./checkpoints/vae_ch256v4096z32.pth \
        --layer_cache_spec '$layer_config' \
        --interpolation_mode $interp_mode \
        --num_samples 1000 \
        --batch_size 50 \
        --output_dir experiments/$exp_name"
      
      if [ ! -z "$blend_ratio" ]; then
        cmd="$cmd --layer_blend_ratios '$blend_ratio'"
      fi
      
      eval $cmd
      
    done
  done
done
```

### 17. 性能基准测试

```bash
#!/bin/bash

# 性能基准测试
echo "开始性能基准测试..."

# 基准测试 (无缓存)
echo "测试基准性能 (无缓存)..."
time ./run_layer_cache_control.sh \
  --model_path ./checkpoints/var_d16.pth \
  --vae_path ./checkpoints/vae_ch256v4096z32.pth \
  --num_samples 1000 \
  --batch_size 50 \
  --output_dir benchmark_baseline \
  2>&1 | tee benchmark_baseline.log

# 单层缓存
echo "测试单层缓存..."
time ./run_layer_cache_control.sh \
  --model_path ./checkpoints/var_d16.pth \
  --vae_path ./checkpoints/vae_ch256v4096z32.pth \
  --layer_cache_spec "9:0:cache" \
  --num_samples 1000 \
  --batch_size 50 \
  --output_dir benchmark_1layer \
  2>&1 | tee benchmark_1layer.log

# 多层缓存
echo "测试多层缓存..."
time ./run_layer_cache_control.sh \
  --model_path ./checkpoints/var_d16.pth \
  --vae_path ./checkpoints/vae_ch256v4096z32.pth \
  --layer_cache_spec "9:0:cache,1:cache,2:cache" \
  --num_samples 1000 \
  --batch_size 50 \
  --output_dir benchmark_3layers \
  2>&1 | tee benchmark_3layers.log

# 混合比例缓存
echo "测试混合比例缓存..."
time ./run_layer_cache_control.sh \
  --model_path ./checkpoints/var_d16.pth \
  --vae_path ./checkpoints/vae_ch256v4096z32.pth \
  --layer_cache_spec "9:0:cache,1:cache,2:cache" \
  --layer_blend_ratios "9:0:0.8,1:0.7,2:0.6" \
  --num_samples 1000 \
  --batch_size 50 \
  --output_dir benchmark_blend \
  2>&1 | tee benchmark_blend.log

echo "基准测试完成！查看各个.log文件以获取详细时间信息。"
```

## 最佳实践总结

### 建议的测试流程

1. **从简单开始**: 先测试单层缓存 `"9:0:cache"`
2. **逐步增加**: 增加更多层 `"9:0:cache,1:cache,2:cache"`
3. **添加混合比例**: 使用适中的比例 `"9:0:0.8,1:0.7,2:0.6"`
4. **评估质量**: 计算FID分数验证质量损失
5. **优化性能**: 根据需要调整参数

### 参数选择指南

- **层数选择**: 通常前3-5层效果最好
- **混合比例**: 0.6-0.9之间，从高到低递减
- **插值模式**: bilinear作为默认，bicubic用于高质量需求
- **阶段选择**: Stage 9(最高分辨率)通常是最有效的优化目标

通过这些示例，您应该能够找到适合您特定需求的配置。建议从简单配置开始，逐步优化以获得最佳的质量-性能平衡。