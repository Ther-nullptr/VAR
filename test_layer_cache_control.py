#!/usr/bin/env python3
"""
Test script for fine-grained layer cache control functionality
This script validates the parsing and configuration of the new fine-grained cache control features
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.var_layer_cache_control import parse_layer_cache_spec, parse_layer_set_spec, parse_layer_blend_spec
from models.basic_var_layer_control import LayerCacheConfig


def test_layer_cache_spec_parsing():
    """Test parsing of layer cache specifications"""
    print("Testing layer cache spec parsing...")
    
    # Test basic specification
    spec = "9:0:cache,1:cache,2:cache;8:0:cache,15:cache"
    result = parse_layer_cache_spec(spec)
    expected = {
        9: {0: 'cache', 1: 'cache', 2: 'cache'},
        8: {0: 'cache', 15: 'cache'}
    }
    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ Basic layer cache spec parsing passed")
    
    # Test with different modes
    spec = "9:0:cache,1:skip,2:normal;8:5:cache"
    result = parse_layer_cache_spec(spec)
    expected = {
        9: {0: 'cache', 1: 'skip', 2: 'normal'},
        8: {5: 'cache'}
    }
    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ Multi-mode layer cache spec parsing passed")
    
    # Test empty specification
    result = parse_layer_cache_spec("")
    assert result == {}, f"Expected empty dict, got {result}"
    print("✓ Empty layer cache spec parsing passed")


def test_layer_set_spec_parsing():
    """Test parsing of layer set specifications"""
    print("\nTesting layer set spec parsing...")
    
    # Test basic specification
    spec = "9:0,1,2;8:0,15"
    result = parse_layer_set_spec(spec)
    expected = {
        9: {0, 1, 2},
        8: {0, 15}
    }
    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ Basic layer set spec parsing passed")
    
    # Test single layer per stage
    spec = "9:5;8:10;7:2"
    result = parse_layer_set_spec(spec)
    expected = {
        9: {5},
        8: {10},
        7: {2}
    }
    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ Single layer set spec parsing passed")


def test_layer_blend_spec_parsing():
    """Test parsing of layer blend ratio specifications"""
    print("\nTesting layer blend spec parsing...")
    
    # Test basic specification
    spec = "9:0:0.8,1:0.6,2:0.9;8:0:0.7"
    result = parse_layer_blend_spec(spec)
    expected = {
        (9, 0): 0.8,
        (9, 1): 0.6,
        (9, 2): 0.9,
        (8, 0): 0.7
    }
    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ Basic layer blend spec parsing passed")
    
    # Test single ratio
    spec = "9:5:0.5"
    result = parse_layer_blend_spec(spec)
    expected = {(9, 5): 0.5}
    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ Single layer blend spec parsing passed")


def test_layer_cache_config():
    """Test LayerCacheConfig with fine-grained control"""
    print("\nTesting LayerCacheConfig...")
    
    # Create configuration with fine-grained control
    stage_layer_cache_control = {
        9: {0: 'cache', 1: 'cache', 2: 'cache'},
        8: {0: 'cache', 15: 'cache'}
    }
    cache_attn_layers = {
        9: {0, 1, 2},
        8: {0, 15}
    }
    cache_mlp_layers = {
        9: {1, 2},
        8: {15}
    }
    layer_blend_ratios = {
        (9, 0): 0.8,
        (9, 1): 0.6,
        (8, 0): 0.7
    }
    
    config = LayerCacheConfig(
        stage_layer_cache_control=stage_layer_cache_control,
        cache_attn_layers=cache_attn_layers,
        cache_mlp_layers=cache_mlp_layers,
        layer_blend_ratios=layer_blend_ratios
    )
    
    # Test should_cache_layer_attn method
    assert config.should_cache_layer_attn(9, 0, 256) == True, "Stage 9, Layer 0 should cache attn"
    assert config.should_cache_layer_attn(9, 5, 256) == False, "Stage 9, Layer 5 should not cache attn"
    assert config.should_cache_layer_attn(7, 0, 256) == False, "Stage 7, Layer 0 should not cache attn"
    print("✓ should_cache_layer_attn logic passed")
    
    # Test should_cache_layer_mlp method
    assert config.should_cache_layer_mlp(9, 1, 256) == True, "Stage 9, Layer 1 should cache mlp"
    assert config.should_cache_layer_mlp(9, 0, 256) == True, "Stage 9, Layer 0 should cache mlp (in fine-grained control)"
    assert config.should_cache_layer_mlp(8, 15, 256) == True, "Stage 8, Layer 15 should cache mlp"
    assert config.should_cache_layer_mlp(7, 0, 256) == False, "Stage 7, Layer 0 should not cache mlp"
    print("✓ should_cache_layer_mlp logic passed")
    
    # Test layer blend ratios
    assert config.get_layer_blend_ratio(9, 0) == 0.8, "Stage 9, Layer 0 blend ratio should be 0.8"
    assert config.get_layer_blend_ratio(9, 5) == 1.0, "Stage 9, Layer 5 blend ratio should be 1.0 (default)"
    print("✓ Layer blend ratio logic passed")
    
    # Test cache summary
    summary = config.get_cache_summary()
    print(f"✓ Cache summary: {summary}")


def test_integration():
    """Test integration of all parsing functions"""
    print("\nTesting integration...")
    
    # Parse complex specifications
    layer_cache_spec = "9:0:cache,1:cache,2:cache;8:0:cache,15:cache"
    cache_attn_layers_spec = "9:0,1,2;8:0,15"
    cache_mlp_layers_spec = "9:1,2;8:15"
    layer_blend_ratios_spec = "9:0:0.8,1:0.6;8:0:0.7"
    
    stage_layer_cache_control = parse_layer_cache_spec(layer_cache_spec)
    cache_attn_layers = parse_layer_set_spec(cache_attn_layers_spec)
    cache_mlp_layers = parse_layer_set_spec(cache_mlp_layers_spec)
    layer_blend_ratios = parse_layer_blend_spec(layer_blend_ratios_spec)
    
    # Create complete configuration
    config = LayerCacheConfig(
        skip_stages=[169, 256],
        cache_stages=[100, 169],
        enable_attn_cache=True,
        enable_mlp_cache=True,
        threshold=0.7,
        max_skip_stages=9,
        adaptive_threshold=False,
        interpolation_mode='bilinear',
        stage_layer_cache_control=stage_layer_cache_control,
        cache_attn_layers=cache_attn_layers,
        cache_mlp_layers=cache_mlp_layers,
        layer_blend_ratios=layer_blend_ratios
    )
    
    print(f"✓ Integration test complete. Config summary: {config.get_cache_summary()}")


def main():
    """Run all tests"""
    print("Running fine-grained layer cache control tests...\n")
    
    try:
        test_layer_cache_spec_parsing()
        test_layer_set_spec_parsing()
        test_layer_blend_spec_parsing()
        test_layer_cache_config()
        test_integration()
        
        print("\n" + "="*60)
        print("🎉 All tests passed! Fine-grained layer cache control is working correctly.")
        print("="*60)
        
        # Show usage examples
        print("\nUsage Examples:")
        print("1. Cache first 3 layers in stage 9:")
        print("   --layer_cache_spec '9:0:cache,1:cache,2:cache'")
        
        print("\n2. Cache specific attention layers:")
        print("   --cache_attn_layers '9:0,1,2;8:0,15'")
        
        print("\n3. Use blend ratios for smoother cache transitions:")
        print("   --layer_blend_ratios '9:0:0.8,1:0.6,2:0.9'")
        
        print("\n4. Complete example with bash script:")
        print("   ./run_layer_cache_control.sh \\")
        print("     --model_path /path/to/var.pth \\")
        print("     --vae_path /path/to/vae.pth \\")
        print("     --layer_cache_spec '9:0:cache,1:cache,2:cache;8:0:cache,15:cache' \\")
        print("     --layer_blend_ratios '9:0:0.8,1:0.6;8:0:0.7' \\")
        print("     --num_samples 1000")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)