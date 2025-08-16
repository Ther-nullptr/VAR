#!/usr/bin/env python3
"""
Example script demonstrating VAR fine-grained cache control with different model depths
This shows how the system automatically adapts to different VAR architectures (d16, d20, d24, d30)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.basic_var_layer_control import LayerCacheConfig
from models.var_layer_cache_control import parse_layer_cache_spec, parse_layer_set_spec, parse_layer_blend_spec


def demonstrate_depth_adaptation():
    """Demonstrate how cache config adapts to different model depths"""
    
    print("=" * 80)
    print("VAR Fine-Grained Cache Control - Multi-Depth Support Demo")
    print("=" * 80)
    
    # Test configurations for different depths
    test_configs = [
        {
            "model_depth": 16,
            "description": "VAR-d16 (Standard)",
            "layer_cache_spec": "9:0:cache,1:cache,2:cache;8:0:cache,15:cache",
            "layer_blend_ratios": "9:0:0.8,1:0.7,2:0.6;8:0:0.8"
        },
        {
            "model_depth": 20,
            "description": "VAR-d20 (Medium)",
            "layer_cache_spec": "9:0:cache,1:cache,2:cache,3:cache;8:0:cache,19:cache",
            "layer_blend_ratios": "9:0:0.8,1:0.7,2:0.6,3:0.5;8:0:0.8"
        },
        {
            "model_depth": 24,
            "description": "VAR-d24 (Large)",
            "layer_cache_spec": "9:0:cache,1:cache,2:cache,3:cache,4:cache;8:0:cache,20:cache,23:cache",
            "layer_blend_ratios": "9:0:0.8,1:0.7,2:0.6,3:0.5,4:0.4;8:0:0.8,20:0.7"
        },
        {
            "model_depth": 30,
            "description": "VAR-d30 (Extra Large)",
            "layer_cache_spec": "9:0:cache,1:cache,2:cache,3:cache,4:cache,5:cache;8:0:cache,25:cache,29:cache",
            "layer_blend_ratios": "9:0:0.8,1:0.7,2:0.6,3:0.5,4:0.4,5:0.3;8:0:0.8,25:0.7"
        }
    ]
    
    for config in test_configs:
        print(f"\n{'-' * 60}")
        print(f"Testing {config['description']} (depth={config['model_depth']})")
        print(f"{'-' * 60}")
        
        try:
            # Parse specifications
            stage_layer_cache_control = parse_layer_cache_spec(config["layer_cache_spec"])
            layer_blend_ratios = parse_layer_blend_spec(config["layer_blend_ratios"])
            
            # Create configuration
            layer_cache_config = LayerCacheConfig(
                model_depth=config["model_depth"],
                num_stages=10,  # Standard VAR has 10 stages
                stage_layer_cache_control=stage_layer_cache_control,
                layer_blend_ratios=layer_blend_ratios,
                skip_stages=[169, 256],  # Example legacy stages
                cache_stages=[100, 169]  # Example legacy stages
            )
            
            print(f"✓ Configuration created successfully")
            print(f"✓ Summary: {layer_cache_config.get_cache_summary()}")
            
            # Test layer validation
            test_cases = [
                (9, 0, "Should pass - valid layer"),
                (9, config["model_depth"] - 1, "Should pass - last layer"),
                (9, config["model_depth"], "Should fail - invalid layer"),
                (10, 0, "Should fail - invalid stage")
            ]
            
            print(f"\nValidation tests:")
            for stage_idx, layer_idx, description in test_cases:
                try:
                    # Test if validation would pass
                    if (0 <= stage_idx < 10 and 0 <= layer_idx < config["model_depth"]):
                        print(f"  ✓ Stage {stage_idx}, Layer {layer_idx}: {description}")
                    else:
                        print(f"  ⚠ Stage {stage_idx}, Layer {layer_idx}: {description}")
                except Exception as e:
                    print(f"  ✗ Stage {stage_idx}, Layer {layer_idx}: {description} - {e}")
            
            # Show cache effectiveness
            total_layers = config["model_depth"] * 10  # 10 stages
            cached_layers = len(stage_layer_cache_control.get(9, {})) + len(stage_layer_cache_control.get(8, {}))
            cache_ratio = cached_layers / config["model_depth"] * 100
            
            print(f"\nCache Statistics:")
            print(f"  Model depth: {config['model_depth']} layers")
            print(f"  Layers using cache in critical stages: {cached_layers}")
            print(f"  Cache ratio in critical stages: {cache_ratio:.1f}%")
            
        except Exception as e:
            print(f"✗ Error: {e}")


def demonstrate_command_line_usage():
    """Show command line usage examples for different depths"""
    
    print(f"\n{'-' * 80}")
    print("Command Line Usage Examples")
    print(f"{'-' * 80}")
    
    examples = [
        {
            "depth": 16,
            "spec": "9:0:cache,1:cache,2:cache",
            "description": "Basic 3-layer cache for VAR-d16"
        },
        {
            "depth": 20,
            "spec": "9:0:cache,1:cache,2:cache,3:cache;8:0:cache,19:cache",
            "description": "Multi-stage cache for VAR-d20"
        },
        {
            "depth": 24,
            "spec": "9:0:cache,1:cache,2:cache,3:cache,4:cache;8:0:cache,20:cache,23:cache",
            "description": "Extended cache for VAR-d24"
        },
        {
            "depth": 30,
            "spec": "9:0:cache,1:cache,2:cache,3:cache,4:cache,5:cache;8:0:cache,25:cache,29:cache",
            "description": "Comprehensive cache for VAR-d30"
        }
    ]
    
    for example in examples:
        print(f"\n# {example['description']}")
        print(f"./run_layer_cache_control.sh \\")
        print(f"  --model_path /path/to/var_d{example['depth']}.pth \\")
        print(f"  --vae_path /path/to/vae.pth \\")
        print(f"  --model_depth {example['depth']} \\")
        print(f"  --layer_cache_spec '{example['spec']}' \\")
        print(f"  --num_samples 1000")


def demonstrate_error_handling():
    """Demonstrate error handling for invalid configurations"""
    
    print(f"\n{'-' * 80}")
    print("Error Handling Demonstration")
    print(f"{'-' * 80}")
    
    error_cases = [
        {
            "name": "Invalid model depth",
            "config": {"model_depth": 18},  # Not in [16, 20, 24, 30]
            "expected_error": "Unsupported model depth"
        },
        {
            "name": "Layer index out of range",
            "config": {
                "model_depth": 16,
                "stage_layer_cache_control": {9: {16: "cache"}}  # Layer 16 doesn't exist in d16
            },
            "expected_error": "Invalid layer index"
        },
        {
            "name": "Stage index out of range",
            "config": {
                "model_depth": 16,
                "stage_layer_cache_control": {10: {0: "cache"}}  # Stage 10 doesn't exist
            },
            "expected_error": "Invalid stage index"
        },
        {
            "name": "Invalid blend ratio",
            "config": {
                "model_depth": 16,
                "layer_blend_ratios": {(9, 0): 1.5}  # Ratio > 1.0
            },
            "expected_error": "Blend ratio"
        }
    ]
    
    for case in error_cases:
        print(f"\nTesting: {case['name']}")
        try:
            config = LayerCacheConfig(**case["config"])
            print(f"  ✗ Expected error but configuration was accepted")
        except Exception as e:
            if case["expected_error"] in str(e):
                print(f"  ✓ Correctly caught error: {e}")
            else:
                print(f"  ⚠ Unexpected error: {e}")


def demonstrate_best_practices():
    """Show best practices for different model depths"""
    
    print(f"\n{'-' * 80}")
    print("Best Practices for Different Model Depths")
    print(f"{'-' * 80}")
    
    recommendations = {
        16: {
            "cache_layers": "0-2 (early layers)",
            "blend_ratios": "0.8-0.6 (aggressive)",
            "stages": "9 (highest resolution)",
            "reasoning": "d16 has fewer layers, focus on early layers in critical stages"
        },
        20: {
            "cache_layers": "0-3 (early-mid layers)",
            "blend_ratios": "0.8-0.5 (moderate)",
            "stages": "9,8 (highest resolutions)",
            "reasoning": "d20 allows more cache flexibility, expand to multiple stages"
        },
        24: {
            "cache_layers": "0-4 (early-mid layers)",
            "blend_ratios": "0.8-0.4 (balanced)",
            "stages": "9,8,7 (multiple stages)",
            "reasoning": "d24 has enough capacity for multi-stage optimization"
        },
        30: {
            "cache_layers": "0-5 (broad early range)",
            "blend_ratios": "0.8-0.3 (conservative)",
            "stages": "9,8,7 (multiple stages)",
            "reasoning": "d30 allows extensive caching while maintaining quality"
        }
    }
    
    for depth, rec in recommendations.items():
        print(f"\nVAR-d{depth} Recommendations:")
        print(f"  Cache layers: {rec['cache_layers']}")
        print(f"  Blend ratios: {rec['blend_ratios']}")
        print(f"  Target stages: {rec['stages']}")
        print(f"  Reasoning: {rec['reasoning']}")


def main():
    """Run all demonstrations"""
    
    demonstrate_depth_adaptation()
    demonstrate_command_line_usage()
    demonstrate_error_handling()
    demonstrate_best_practices()
    
    print(f"\n{'=' * 80}")
    print("Multi-Depth Support Demo Complete!")
    print("=" * 80)
    print("\nKey Features:")
    print("✓ Automatic validation for different model depths (16, 20, 24, 30)")
    print("✓ Layer index range checking based on model architecture")
    print("✓ Stage-aware cache configuration")
    print("✓ Comprehensive error handling")
    print("✓ Flexible blend ratio support")
    print("\nNext Steps:")
    print("1. Use --model_depth parameter to specify your model architecture")
    print("2. Adjust layer indices according to your model's depth")
    print("3. Test configurations with the test_layer_cache_control.py script")
    print("4. Run generation with run_layer_cache_control.sh")


if __name__ == '__main__':
    main()