"""
Enhanced VAR evaluation script with flexible multi-stage caching control
"""

import argparse
import os
import sys
import time
from typing import List, Optional, Tuple
import json

import torch
import torch.nn as nn
import torch.distributed as dist
from torchvision.utils import save_image
import numpy as np

# Add model path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.var_enhanced import VAREnhanced, CacheConfig, create_var_enhanced_model
from models.vqvae import VQVAE
from utils.arg_util import arg_util


def parse_list_arg(arg_str: str) -> List[int]:
    """Parse comma-separated list of integers from command line argument"""
    if not arg_str:
        return []
    return [int(x.strip()) for x in arg_str.split(',') if x.strip()]


def create_cache_config_from_args(args) -> CacheConfig:
    """Create CacheConfig from command line arguments"""
    return CacheConfig(
        skip_stages=parse_list_arg(args.skip_stages),
        cache_stages=parse_list_arg(args.cache_stages),
        enable_attn_cache=args.enable_attn_cache,
        enable_mlp_cache=args.enable_mlp_cache,
        threshold=args.cache_threshold,
        max_skip_stages=args.max_skip_stages,
        adaptive_threshold=args.adaptive_threshold,
        interpolation_mode=args.interpolation_mode
    )


def validate_cache_stages(skip_stages: List[int], cache_stages: List[int]) -> bool:
    """Validate that cache stages make sense"""
    valid_stages = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256]
    
    for stage in skip_stages + cache_stages:
        if stage not in valid_stages:
            print(f"Error: Invalid stage {stage}. Valid stages are: {valid_stages}")
            return False
    
    # Check logical consistency
    if len(skip_stages) == 0 and len(cache_stages) > 0:
        print("Error: Cannot have cache stages without skip stages")
        return False
    
    return True


def print_cache_config(cache_config: CacheConfig):
    """Print cache configuration summary"""
    print(f"\n{'='*60}")
    print(f"Enhanced VAR Cache Configuration")
    print(f"{'='*60}")
    print(f"Skip stages: {cache_config.skip_stages}")
    print(f"Cache stages: {cache_config.cache_stages}")
    print(f"Attention cache enabled: {cache_config.enable_attn_cache}")
    print(f"MLP cache enabled: {cache_config.enable_mlp_cache}")
    print(f"Similarity threshold: {cache_config.threshold}")
    print(f"Max skip stages: {cache_config.max_skip_stages}")
    print(f"Adaptive threshold: {cache_config.adaptive_threshold}")
    print(f"Interpolation mode: {cache_config.interpolation_mode}")
    print(f"{'='*60}\n")


def benchmark_generation_speed(
    model: VAREnhanced, 
    batch_size: int = 16, 
    num_classes: int = 1000,
    num_runs: int = 5,
    device: str = 'cuda'
) -> dict:
    """Benchmark generation speed with current cache configuration"""
    model.eval()
    torch.cuda.empty_cache()
    
    times = []
    memory_usage = []
    
    print(f"Benchmarking generation speed (batch_size={batch_size}, runs={num_runs})...")
    
    for run in range(num_runs):
        # Random class labels
        labels = torch.randint(0, num_classes, (batch_size,), device=device)
        
        # Measure generation time
        torch.cuda.synchronize()
        start_time = time.time()
        
        # Memory before generation
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            mem_before = torch.cuda.memory_allocated() / 1024**2  # MB
        
        with torch.no_grad():
            generated_images = model.autoregressive_infer_cfg(
                B=batch_size,
                label_B=labels,
                g_seed=42 + run,
                cfg=1.5,
                top_k=900,
                top_p=0.96
            )
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        # Memory after generation
        if device == 'cuda':
            mem_after = torch.cuda.max_memory_allocated() / 1024**2  # MB
            memory_usage.append(mem_after - mem_before)
        
        elapsed = end_time - start_time
        times.append(elapsed)
        
        print(f"Run {run + 1}/{num_runs}: {elapsed:.2f}s, Memory: {memory_usage[-1]:.1f}MB")
    
    # Statistics
    mean_time = np.mean(times)
    std_time = np.std(times)
    mean_memory = np.mean(memory_usage) if memory_usage else 0
    
    stats = {
        'mean_time': mean_time,
        'std_time': std_time,
        'mean_memory_mb': mean_memory,
        'throughput_img_per_sec': batch_size / mean_time,
        'all_times': times,
        'all_memory': memory_usage
    }
    
    print(f"\nBenchmark Results:")
    print(f"Mean time: {mean_time:.2f} ± {std_time:.2f}s")
    print(f"Throughput: {stats['throughput_img_per_sec']:.2f} images/sec")
    print(f"Mean memory: {mean_memory:.1f}MB")
    
    return stats


def run_cache_calibration(
    model: VAREnhanced,
    batch_size: int = 16,
    num_classes: int = 1000,
    calibration_samples: int = 100,
    device: str = 'cuda'
) -> dict:
    """Run cache calibration to compute similarity statistics"""
    print(f"Running cache calibration with {calibration_samples} samples...")
    
    model.enable_calibration_mode(True)
    model.eval()
    
    num_batches = (calibration_samples + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        current_batch_size = min(batch_size, calibration_samples - batch_idx * batch_size)
        labels = torch.randint(0, num_classes, (current_batch_size,), device=device)
        
        with torch.no_grad():
            _ = model.autoregressive_infer_cfg(
                B=current_batch_size,
                label_B=labels,
                g_seed=42 + batch_idx,
                cfg=1.5,
                top_k=900,
                top_p=0.96
            )
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Calibration progress: {batch_idx + 1}/{num_batches} batches")
    
    model.enable_calibration_mode(False)
    
    # Get calibration statistics
    stats = model.get_cache_statistics()
    print("Calibration completed!")
    
    return stats


def save_results(results: dict, output_path: str):
    """Save benchmark results to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Enhanced VAR Evaluation with Flexible Caching')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to VAR model checkpoint')
    parser.add_argument('--vae_path', type=str, required=True,
                        help='Path to VAE checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    
    # Cache configuration arguments
    parser.add_argument('--skip_stages', type=str, default='',
                        help='Comma-separated list of stages to skip (e.g., "169,256")')
    parser.add_argument('--cache_stages', type=str, default='',
                        help='Comma-separated list of stages to cache (e.g., "100,169")')
    parser.add_argument('--enable_attn_cache', action='store_true', default=True,
                        help='Enable attention layer caching')
    parser.add_argument('--disable_attn_cache', action='store_false', dest='enable_attn_cache',
                        help='Disable attention layer caching')
    parser.add_argument('--enable_mlp_cache', action='store_true', default=True,
                        help='Enable MLP layer caching')
    parser.add_argument('--disable_mlp_cache', action='store_false', dest='enable_mlp_cache',
                        help='Disable MLP layer caching')
    parser.add_argument('--cache_threshold', type=float, default=0.7,
                        help='Similarity threshold for cache reuse')
    parser.add_argument('--max_skip_stages', type=int, default=9,
                        help='Maximum number of stages to skip')
    parser.add_argument('--adaptive_threshold', action='store_true',
                        help='Use adaptive similarity threshold')
    parser.add_argument('--interpolation_mode', type=str, default='bilinear',
                        choices=['bilinear', 'nearest', 'bicubic'],
                        help='Interpolation mode for feature upsampling')
    
    # Evaluation arguments
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to generate')
    parser.add_argument('--num_classes', type=int, default=1000,
                        help='Number of classes')
    parser.add_argument('--cfg_scale', type=float, default=1.5,
                        help='Classifier-free guidance scale')
    parser.add_argument('--top_k', type=int, default=900,
                        help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=0.96,
                        help='Top-p (nucleus) sampling')
    
    # Operation modes
    parser.add_argument('--calibrate', action='store_true',
                        help='Run cache calibration')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run generation speed benchmark')
    parser.add_argument('--generate', action='store_true',
                        help='Generate sample images')
    parser.add_argument('--compare_configs', action='store_true',
                        help='Compare multiple cache configurations')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./enhanced_var_output',
                        help='Output directory for generated images and results')
    parser.add_argument('--save_images', action='store_true',
                        help='Save generated images')
    parser.add_argument('--similarity_data_path', type=str, default='',
                        help='Path to save/load similarity calibration data')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Validate cache configuration
    skip_stages = parse_list_arg(args.skip_stages)
    cache_stages = parse_list_arg(args.cache_stages)
    
    if not validate_cache_stages(skip_stages, cache_stages):
        return
    
    # Create cache configuration
    cache_config = create_cache_config_from_args(args)
    print_cache_config(cache_config)
    
    # Load models
    print("Loading models...")
    device = torch.device(args.device)
    
    # Load VAE
    vae_model = VQVAE.from_pretrained(args.vae_path)
    vae_model.to(device)
    vae_model.eval()
    
    # Load VAR model with enhanced caching
    var_model = VAREnhanced.from_pretrained(args.model_path)
    var_model.set_cache_config(cache_config)
    var_model.to(device)
    var_model.eval()
    
    # Load similarity data if provided
    if args.similarity_data_path and os.path.exists(args.similarity_data_path):
        var_model.load_similarity_data(args.similarity_data_path)
    
    print(f"Model loaded. Using device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in var_model.parameters()):,}")
    
    results = {}
    
    # Run calibration
    if args.calibrate:
        print("\n" + "="*60)
        print("Running Cache Calibration")
        print("="*60)
        calibration_stats = run_cache_calibration(
            var_model, 
            batch_size=args.batch_size,
            num_classes=args.num_classes,
            calibration_samples=args.num_samples,
            device=args.device
        )
        results['calibration'] = calibration_stats
        
        # Save calibration data
        if args.similarity_data_path:
            var_model.save_similarity_data(args.similarity_data_path)
    
    # Run benchmark
    if args.benchmark:
        print("\n" + "="*60)
        print("Running Speed Benchmark")
        print("="*60)
        benchmark_stats = benchmark_generation_speed(
            var_model,
            batch_size=args.batch_size,
            num_classes=args.num_classes,
            device=args.device
        )
        results['benchmark'] = benchmark_stats
    
    # Generate samples
    if args.generate:
        print("\n" + "="*60)
        print("Generating Sample Images")
        print("="*60)
        
        num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
        all_images = []
        
        for batch_idx in range(num_batches):
            current_batch_size = min(args.batch_size, args.num_samples - batch_idx * args.batch_size)
            labels = torch.randint(0, args.num_classes, (current_batch_size,), device=device)
            
            with torch.no_grad():
                images = var_model.autoregressive_infer_cfg(
                    B=current_batch_size,
                    label_B=labels,
                    g_seed=42 + batch_idx,
                    cfg=args.cfg_scale,
                    top_k=args.top_k,
                    top_p=args.top_p
                )
            
            all_images.append(images.cpu())
            print(f"Generated batch {batch_idx + 1}/{num_batches}")
        
        # Save images
        if args.save_images:
            all_images = torch.cat(all_images, dim=0)
            save_path = os.path.join(args.output_dir, 'generated_samples.png')
            save_image(all_images, save_path, nrow=8, normalize=True)
            print(f"Saved {len(all_images)} images to {save_path}")
    
    # Compare configurations
    if args.compare_configs:
        print("\n" + "="*60)
        print("Comparing Cache Configurations")
        print("="*60)
        
        configs_to_test = [
            # No caching
            CacheConfig(skip_stages=[], cache_stages=[]),
            # Original VAR caching
            CacheConfig(skip_stages=[169, 256], cache_stages=[100, 169]),
            # Conservative caching (fewer skips)
            CacheConfig(skip_stages=[256], cache_stages=[169]),
            # Aggressive caching (more skips)
            CacheConfig(skip_stages=[100, 169, 256], cache_stages=[64, 100, 169]),
            # MLP-only caching
            CacheConfig(skip_stages=[169, 256], cache_stages=[100, 169], 
                       enable_attn_cache=False, enable_mlp_cache=True),
            # Attention-only caching
            CacheConfig(skip_stages=[169, 256], cache_stages=[100, 169],
                       enable_attn_cache=True, enable_mlp_cache=False),
        ]
        
        config_names = [
            'no_cache', 'original', 'conservative', 'aggressive', 'mlp_only', 'attn_only'
        ]
        
        comparison_results = {}
        
        for config, name in zip(configs_to_test, config_names):
            print(f"\nTesting configuration: {name}")
            print(f"Skip stages: {config.skip_stages}, Cache stages: {config.cache_stages}")
            print(f"Attention cache: {config.enable_attn_cache}, MLP cache: {config.enable_mlp_cache}")
            
            var_model.set_cache_config(config)
            
            # Run benchmark for this configuration
            stats = benchmark_generation_speed(
                var_model, 
                batch_size=args.batch_size//2,  # Smaller batch for comparison
                num_classes=args.num_classes,
                num_runs=3,  # Fewer runs for comparison
                device=args.device
            )
            
            comparison_results[name] = {
                'config': {
                    'skip_stages': config.skip_stages,
                    'cache_stages': config.cache_stages,
                    'enable_attn_cache': config.enable_attn_cache,
                    'enable_mlp_cache': config.enable_mlp_cache,
                },
                'performance': stats
            }
        
        results['comparison'] = comparison_results
        
        # Print comparison summary
        print("\n" + "="*60)
        print("Configuration Comparison Summary")
        print("="*60)
        print(f"{'Config':<12} {'Time (s)':<10} {'Speedup':<8} {'Memory (MB)':<12} {'Throughput (img/s)':<15}")
        print("-" * 70)
        
        baseline_time = comparison_results['no_cache']['performance']['mean_time']
        for name, result in comparison_results.items():
            perf = result['performance']
            speedup = baseline_time / perf['mean_time']
            print(f"{name:<12} {perf['mean_time']:<10.2f} {speedup:<8.2f} "
                  f"{perf['mean_memory_mb']:<12.1f} {perf['throughput_img_per_sec']:<15.2f}")
    
    # Save all results
    if results:
        results_path = os.path.join(args.output_dir, 'enhanced_var_results.json')
        save_results(results, results_path)
    
    print("\nEnhanced VAR evaluation completed!")


if __name__ == '__main__':
    main()