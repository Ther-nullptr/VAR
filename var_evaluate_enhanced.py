"""
Enhanced VAR evaluation script with flexible multi-stage caching control
Compatible with original var_evaluate.py structure
"""

import argparse
import os
import os.path as osp
import sys
import time
from typing import List, Optional, Tuple
import json

import torch
import torchvision
import random
from tqdm import tqdm
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw

# Disable default parameter init for faster speed
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)

# Import models using original structure
from models.vqvae import VQVAE
from models.basic_var_enhanced import CacheConfig
import torch_fidelity


def parse_list_arg(arg_str: str) -> List[int]:
    """Parse comma-separated list of integers from command line argument"""
    if not arg_str:
        return []
    return [int(x.strip()) for x in arg_str.split(',') if x.strip()]


def build_vae_var_enhanced(
    # Shared args
    device, patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
    # VQVAE args
    V=4096, Cvae=32, ch=160, share_quant_resi=4,
    # VAR args
    num_classes=1000, depth=16, shared_aln=False, attn_l2_norm=True,
    flash_if_available=True, fused_if_available=True,
    init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=-1,    # init_std < 0: automated
    # Enhanced caching args
    cache_config: Optional[CacheConfig] = None,
    calibration=False, sim_path=None
) -> Tuple[VQVAE, 'VAREnhanced']:
    """
    Build VAE and enhanced VAR model with caching configuration
    Compatible with original build_vae_var interface
    """
    from models.var_enhanced import VAREnhanced
    
    heads = depth
    width = depth * 64
    dpr = 0.1 * depth/24
    
    # disable built-in initialization for speed
    for clz in (torch.nn.Linear, torch.nn.LayerNorm, torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d):
        setattr(clz, 'reset_parameters', lambda self: None)
    
    # build VQVAE
    vae_local = VQVAE(vocab_size=V, z_channels=Cvae, ch=ch, test_mode=True, share_quant_resi=share_quant_resi, v_patch_nums=patch_nums).to(device)
    
    # build enhanced VAR
    var_wo_ddp = VAREnhanced(
        vae_local=vae_local,
        num_classes=num_classes, depth=depth, embed_dim=width, num_heads=heads, 
        drop_rate=0., attn_drop_rate=0., drop_path_rate=dpr,
        norm_eps=1e-6, shared_aln=shared_aln, cond_drop_rate=0.1,
        attn_l2_norm=attn_l2_norm,
        patch_nums=patch_nums,
        flash_if_available=flash_if_available, fused_if_available=fused_if_available,
        # Enhanced caching parameters
        skip_stages=cache_config.skip_stages if cache_config else [],
        cache_stages=cache_config.cache_stages if cache_config else [],
        enable_attn_cache=cache_config.enable_attn_cache if cache_config else True,
        enable_mlp_cache=cache_config.enable_mlp_cache if cache_config else True,
        cache_threshold=cache_config.threshold if cache_config else 0,
        max_skip_stages=cache_config.max_skip_stages if cache_config else 9,
        adaptive_threshold=cache_config.adaptive_threshold if cache_config else False,
        interpolation_mode=cache_config.interpolation_mode if cache_config else 'bilinear',
        # Legacy compatibility
        calibration=calibration, sim_path=sim_path
    ).to(device)
    
    # Initialize weights
    var_wo_ddp.init_weights(init_adaln=init_adaln, init_adaln_gamma=init_adaln_gamma, init_head=init_head, init_std=init_std)
    
    return vae_local, var_wo_ddp


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
    var_model, 
    batch_size: int = 16, 
    num_classes: int = 1000,
    num_runs: int = 5,
    device: str = 'cuda',
    cfg: float = 1.5,
    seed: int = 42
) -> dict:
    """Benchmark generation speed with current cache configuration"""
    var_model.eval()
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
            generated_images = var_model.autoregressive_infer_cfg(
                B=batch_size,
                label_B=labels,
                g_seed=seed + run,
                cfg=cfg,
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
    var_model,
    batch_size: int = 16,
    num_classes: int = 1000,
    calibration_samples: int = 100,
    device: str = 'cuda',
    cfg: float = 1.5,
    seed: int = 42
) -> dict:
    """Run cache calibration to compute similarity statistics"""
    print(f"Running cache calibration with {calibration_samples} samples...")
    
    var_model.enable_calibration_mode(True)
    var_model.eval()
    
    num_batches = (calibration_samples + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        current_batch_size = min(batch_size, calibration_samples - batch_idx * batch_size)
        labels = torch.randint(0, num_classes, (current_batch_size,), device=device)
        
        with torch.no_grad():
            _ = var_model.autoregressive_infer_cfg(
                B=current_batch_size,
                label_B=labels,
                g_seed=seed,
                cfg=cfg,
                top_k=900,
                top_p=0.96
            )
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Calibration progress: {batch_idx + 1}/{num_batches} batches")
    
    var_model.enable_calibration_mode(False)
    
    # Get calibration statistics
    stats = var_model.get_cache_statistics()
    print("Calibration completed!")
    
    return stats


def parse_args():
    parser = argparse.ArgumentParser(description='Enhanced VAR model evaluation with configurable caching')
    
    # Model configuration (matching bash script parameter names)
    parser.add_argument('--model-depth', type=int, default=16, choices=[16, 20, 24, 30], 
                       help='Model depth (16, 20, 24, or 30)')
    parser.add_argument('--vae_path', type=str, default='', 
                       help='Path to VAE checkpoint')
    parser.add_argument('--model_path', type=str, default='', 
                       help='Path to VAR checkpoint')
    parser.add_argument('--device', type=str, default='cuda', 
                       help='Device to use (cuda/cpu)')
    
    # Generation parameters (matching bash script parameter names) 
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--cfg', type=float, default=1.5, 
                       help='Classifier-free guidance scale (1-10)')
    parser.add_argument('--more-smooth', action='store_true', 
                       help='Enable for smoother output')
    parser.add_argument('--batch_size', type=int, default=64, 
                       help='Batch size for sampling')
    parser.add_argument('--num_samples', type=int, default=50, 
                       help='Number of samples to generate per class')
    parser.add_argument('--output_dir', type=str, default='./samples', 
                       help='Output directory for generated images')
    parser.add_argument('--tf32', action='store_true', 
                       help='Enable TF32 for faster computation')
    
    # Enhanced cache configuration arguments
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
    parser.add_argument('--cache_threshold', type=float, default=0,
                        help='Similarity threshold for cache reuse')
    parser.add_argument('--max_skip_stages', type=int, default=9,
                        help='Maximum number of stages to skip')
    parser.add_argument('--adaptive_threshold', action='store_true',
                        help='Use adaptive similarity threshold')
    parser.add_argument('--interpolation_mode', type=str, default='bilinear',
                        choices=['bilinear', 'nearest', 'bicubic'],
                        help='Interpolation mode for feature upsampling')
    
    # Operation modes
    parser.add_argument('--calibrate', action='store_true',
                        help='Run cache calibration')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run generation speed benchmark')
    parser.add_argument('--compare_configs', action='store_true',
                        help='Compare multiple cache configurations')
    parser.add_argument('--generate', action='store_true',
                        help='Generate samples (used by bash script)')
    parser.add_argument('--generate_fid', action='store_true',
                        help='Generate samples and compute FID (like original var_evaluate)')
    parser.add_argument('--save_images', action='store_true',
                        help='Save generated images')
    
    # FID computation arguments (matching original)
    parser.add_argument('--fid_statistics_file', type=str, 
                        default='/home/wyj24/project/lpd/fid_stats/adm_in256_stats.npz',
                        help='Path to FID statistics file')
    
    # Similarity data management
    parser.add_argument('--similarity_data_path', type=str, default='',
                        help='Path to save/load similarity calibration data')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validate cache configuration
    skip_stages = parse_list_arg(args.skip_stages)
    cache_stages = parse_list_arg(args.cache_stages)
    
    if not validate_cache_stages(skip_stages, cache_stages):
        return
    
    # Create cache configuration
    cache_config = create_cache_config_from_args(args)
    print_cache_config(cache_config)
    
    # Generate descriptive output directory name with key parameters
    def generate_output_dir_name(base_dir: str, args, cache_config: CacheConfig) -> str:
        """Generate output directory name that includes key training parameters"""
        components = []
        
        # Model configuration
        components.append(f"d{args.model_depth}")
        components.append(f"cfg{args.cfg}")
        components.append(f"bs{args.batch_size}")
        components.append(f"samples{args.num_samples}")
        components.append(f"seed{args.seed}")
        
        # Cache configuration
        if cache_config.skip_stages:
            components.append(f"skip{'_'.join(map(str, cache_config.skip_stages))}")
        if cache_config.cache_stages:
            components.append(f"cache{'_'.join(map(str, cache_config.cache_stages))}")
        
        # Cache layer types
        cache_layers = []
        if cache_config.enable_attn_cache:
            cache_layers.append("attn")
        if cache_config.enable_mlp_cache:
            cache_layers.append("mlp")
        if cache_layers:
            components.append(f"layers{'_'.join(cache_layers)}")
        elif not cache_config.skip_stages:  # No caching at all
            components.append("no_cache")
        
        # Cache threshold
        if cache_config.threshold != 0.0:  # Only include if not default
            components.append(f"th{cache_config.threshold:.2f}")
        
        # Adaptive threshold
        if cache_config.adaptive_threshold:
            components.append("adaptive")
        
        # Interpolation mode (only if not default)
        if cache_config.interpolation_mode != 'bilinear':
            components.append(f"interp{cache_config.interpolation_mode}")
        
        # TF32 flag
        if args.tf32:
            components.append("tf32")
        
        # More smooth flag
        if args.more_smooth:
            components.append("smooth")
        
        dir_name = "var_enhanced_" + "_".join(components)
        return os.path.join(base_dir, dir_name)
    
    # Create base output directory and parameter-specific subdirectory
    base_output_dir = args.output_dir
    param_output_dir = generate_output_dir_name(base_output_dir, args, cache_config)
    os.makedirs(param_output_dir, exist_ok=True)
    
    # Update args.output_dir to use the parameter-specific directory
    args.output_dir = param_output_dir
    print(f"Using parameter-specific output directory: {param_output_dir}")
    
    ################## 1. Download checkpoints and build models (matching original structure)
    # Set up checkpoint paths
    hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
    var_ckpt_dir = '/home/wyj24/models/VAR'  # Default path from original
    
    # Use provided paths or default ones
    if args.vae_path:
        vae_ckpt = args.vae_path
    else:
        vae_ckpt = f'{var_ckpt_dir}/vae_ch160v4096z32.pth'
        if not osp.exists(vae_ckpt): 
            print(f"Downloading VAE checkpoint...")
            os.system(f'wget {hf_home}/vae_ch160v4096z32.pth -O {vae_ckpt}')
    
    if args.model_path:
        var_ckpt = args.model_path
    else:
        var_ckpt = f'{var_ckpt_dir}/var_d{args.model_depth}_new.pth'
        if not osp.exists(var_ckpt):
            print(f"Downloading VAR checkpoint...")
            os.system(f'wget {hf_home}/var_d{args.model_depth}_new.pth -O {var_ckpt}')
    
    # build vae, var (using original structure but with enhancements)
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    device = args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu'
    
    print("Building enhanced VAR model...")
    vae, var = build_vae_var_enhanced(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=args.model_depth, shared_aln=False,
        cache_config=cache_config,
        calibration=args.calibrate,
        sim_path=args.similarity_data_path if args.similarity_data_path else None,
    )
    
    # load checkpoints (original style)
    print("Loading model checkpoints...")
    vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)
    vae.eval(), var.eval()
    for p in vae.parameters(): p.requires_grad_(False)
    for p in var.parameters(): p.requires_grad_(False)
    print(f'Model preparation finished.')
    
    # Load similarity data if provided
    if args.similarity_data_path and os.path.exists(args.similarity_data_path):
        var.load_similarity_data(args.similarity_data_path)
        print(f"Loaded similarity data from {args.similarity_data_path}")
    
    ############################# 2. Run evaluation modes
    
    # seed (matching original)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Enable TF32 if requested
    if args.tf32:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
    
    # Convert to float16 for efficiency (matching original)
    vae, var = vae.to(torch.float16), var.to(torch.float16)
    
    print(f"Model loaded on {device}. Using dtype: {next(var.parameters()).dtype}")
    print(f"Model parameters: {sum(p.numel() for p in var.parameters()):,}")
    
    results = {}
    
    # Run calibration
    if args.calibrate:
        print("\n" + "="*60)
        print("Running Cache Calibration")
        print("="*60)
        calibration_stats = run_cache_calibration(
            var, 
            batch_size=min(args.batch_size, 16),  # Smaller batch for calibration
            calibration_samples=100,
            device=device,
            cfg=args.cfg,
            seed=args.seed
        )
        results['calibration'] = calibration_stats
        
        # Save calibration data
        if args.similarity_data_path:
            var.save_similarity_data(args.similarity_data_path)
    
    # Run benchmark
    if args.benchmark:
        print("\n" + "="*60)
        print("Running Speed Benchmark")
        print("="*60)
        benchmark_stats = benchmark_generation_speed(
            var,
            batch_size=min(args.batch_size, 16),  # Reasonable batch size for benchmarking
            device=device,
            cfg=args.cfg,
            seed=args.seed
        )
        results['benchmark'] = benchmark_stats
    
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
            # Conservative caching
            CacheConfig(skip_stages=[256], cache_stages=[169]),
            # Aggressive caching
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
            
            var.set_cache_config(config)
            
            # Run benchmark for this configuration
            stats = benchmark_generation_speed(
                var, 
                batch_size=min(args.batch_size//2, 8),  # Smaller batch for comparison
                num_runs=3,  # Fewer runs for comparison
                device=device,
                cfg=args.cfg,
                seed=args.seed
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
        
        # Restore original cache config
        var.set_cache_config(cache_config)
    
    # Generate samples and compute FID (matching original var_evaluate.py behavior)
    if args.generate_fid or args.generate:
        print("\n" + "="*60)
        print("Generating Samples and Computing FID")
        print("="*60)
        
        B = args.batch_size
        samples_per_class = args.num_samples
        iterations = (samples_per_class + B - 1) // B  # Ceiling division
        
        # Use the parameter-specific output directory for sample generation
        output_dir = os.path.join(args.output_dir, 'generated_samples')
        if not osp.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        print(f"Generating {samples_per_class} samples per class for 1000 classes...")
        print(f"Output directory: {output_dir}")
        
        # Only save images if requested or if computing FID
        save_images = args.save_images or args.generate_fid
        
        for img_cls in tqdm(range(1000)):
            for i in range(iterations):
                current_batch = min(B, samples_per_class - i * B)
                label_B = torch.tensor([img_cls] * current_batch, device=device)
                
                with torch.no_grad():
                    recon_B3HW = var.autoregressive_infer_cfg(
                        B=current_batch, 
                        label_B=label_B, 
                        cfg=args.cfg, 
                        top_k=900, 
                        top_p=0.96, 
                        g_seed=args.seed, 
                        more_smooth=False
                    )
                
                if save_images:
                    bchw = recon_B3HW.permute(0, 2, 3, 1).mul_(255).cpu().numpy()
                    bchw = bchw.astype(np.uint8)
                    for j in range(current_batch):
                        img = PImage.fromarray(bchw[j])
                        img.save(osp.join(output_dir, f"sample_{img_cls * samples_per_class + i * B + j}.png"))
        
        # compute FID only if explicitly requested and images were saved
        if args.generate_fid and save_images:
            print("Computing FID and Inception Score...")
            if osp.exists(args.fid_statistics_file):
                metrics_dict = torch_fidelity.calculate_metrics(
                    input1=output_dir,
                    input2=None,
                    fid_statistics_file=args.fid_statistics_file,
                    cuda=True,
                    isc=True,
                    fid=True,
                    kid=False,
                    prc=False,
                    verbose=False,
                )
                fid = metrics_dict['frechet_inception_distance']
                inception_score = metrics_dict['inception_score_mean']
                print("FID: {:.4f}, Inception Score: {:.4f}".format(fid, inception_score))
                
                results['fid_evaluation'] = {
                    'fid': float(fid),
                    'inception_score': float(inception_score),
                    'output_dir': output_dir
                }
            else:
                print(f"FID statistics file not found: {args.fid_statistics_file}")
                print("Skipping FID computation.")

            for img_file in os.listdir(output_dir):
                if img_file.endswith('.png'):
                    # extract the sample index from the filename
                    sample_index = int(img_file.split('_')[-1].split('.')[0])
                    if sample_index % samples_per_class != 0:
                        os.remove(osp.join(output_dir, img_file))
    
    # Save all results with parameter information
    if results:
        # Add parameter info to results
        results['parameters'] = {
            'model_depth': args.model_depth,
            'cfg': args.cfg,
            'batch_size': args.batch_size,
            'num_samples': args.num_samples,
            'seed': args.seed,
            'tf32': args.tf32,
            'more_smooth': args.more_smooth,
            'cache_config': {
                'skip_stages': cache_config.skip_stages,
                'cache_stages': cache_config.cache_stages,
                'enable_attn_cache': cache_config.enable_attn_cache,
                'enable_mlp_cache': cache_config.enable_mlp_cache,
                'threshold': cache_config.threshold,
                'adaptive_threshold': cache_config.adaptive_threshold,
                'interpolation_mode': cache_config.interpolation_mode,
                'max_skip_stages': cache_config.max_skip_stages
            }
        }
        
        results_path = os.path.join(args.output_dir, 'enhanced_var_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)  # default=str to handle numpy types
        print(f"\nAll evaluation results saved to {results_path}")
        print(f"Parameter-specific output directory: {args.output_dir}")
    
    print("\nEnhanced VAR evaluation completed!")


if __name__ == '__main__':
    main()