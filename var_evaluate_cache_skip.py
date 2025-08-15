"""
VAR evaluation script with stage-specific layer cache-interpolation skipping functionality
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

# Import models
from models.vqvae import VQVAE
from models.var_layer_skip_cache import VARLayerSkipCache, LayerCacheSkipConfig
import torch_fidelity


def parse_list_arg(arg_str: str) -> List[int]:
    """Parse comma-separated list of integers from command line argument"""
    if not arg_str:
        return []
    return [int(x.strip()) for x in arg_str.split(',') if x.strip()]


def parse_stage_layer_skip(arg_str: str) -> dict:
    """
    Parse stage-layer cache-skip configuration from string
    Format: "stage1:layer1,layer2;stage2:layer3,layer4"
    Example: "9:0,1,2" means use cache for layers 0,1,2 in stage 9
    """
    if not arg_str:
        return {}
    
    result = {}
    for stage_spec in arg_str.split(';'):
        if ':' not in stage_spec:
            continue
        stage_idx, layers_str = stage_spec.split(':', 1)
        stage_idx = int(stage_idx.strip())
        layers = set(parse_list_arg(layers_str))
        if layers:
            result[stage_idx] = layers
    
    return result


def parse_stage_threshold_skip(arg_str: str) -> dict:
    """
    Parse stage-threshold cache-skip configuration from string
    Format: "stage1:threshold1;stage2:threshold2"
    Example: "9:3" means use cache for all layers with index < 3 in stage 9
    """
    if not arg_str:
        return {}
    
    result = {}
    for stage_spec in arg_str.split(';'):
        if ':' not in stage_spec:
            continue
        stage_idx, threshold_str = stage_spec.split(':', 1)
        stage_idx = int(stage_idx.strip())
        threshold = int(threshold_str.strip())
        result[stage_idx] = threshold
    
    return result


def build_vae_var_cache_skip(
    # Shared args
    device, patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
    # VQVAE args
    V=4096, Cvae=32, ch=160, share_quant_resi=4,
    # VAR args
    num_classes=1000, depth=16, shared_aln=False, attn_l2_norm=True,
    flash_if_available=True, fused_if_available=True,
    init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=-1,    # init_std < 0: automated
    # Layer cache-skip args
    layer_cache_skip_config: Optional[LayerCacheSkipConfig] = None
) -> Tuple[VQVAE, VARLayerSkipCache]:
    """
    Build VAE and VAR model with layer cache-interpolation skipping functionality
    """
    heads = depth
    width = depth * 64
    dpr = 0.1 * depth/24
    
    # disable built-in initialization for speed
    for clz in (torch.nn.Linear, torch.nn.LayerNorm, torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d):
        setattr(clz, 'reset_parameters', lambda self: None)
    
    # build VQVAE
    vae_local = VQVAE(vocab_size=V, z_channels=Cvae, ch=ch, test_mode=True, share_quant_resi=share_quant_resi, v_patch_nums=patch_nums).to(device)
    
    # build VAR with layer cache-skip
    var_wo_ddp = VARLayerSkipCache(
        vae_local=vae_local,
        num_classes=num_classes, depth=depth, embed_dim=width, num_heads=heads, 
        drop_rate=0., attn_drop_rate=0., drop_path_rate=dpr,
        norm_eps=1e-6, shared_aln=shared_aln, cond_drop_rate=0.1,
        attn_l2_norm=attn_l2_norm,
        patch_nums=patch_nums,
        flash_if_available=flash_if_available, fused_if_available=fused_if_available,
        layer_cache_skip_config=layer_cache_skip_config
    ).to(device)
    
    # Initialize weights
    var_wo_ddp.init_weights(init_adaln=init_adaln, init_adaln_gamma=init_adaln_gamma, init_head=init_head, init_std=init_std)
    
    return vae_local, var_wo_ddp


def benchmark_generation_speed(
    var_model, 
    batch_size: int = 16, 
    num_classes: int = 1000,
    num_runs: int = 5,
    device: str = 'cuda',
    cfg: float = 1.5,
    seed: int = 42
) -> dict:
    """Benchmark generation speed with current configuration"""
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


def parse_args():
    parser = argparse.ArgumentParser(description='VAR model evaluation with stage-specific layer cache-interpolation skipping')
    
    # Model configuration
    parser.add_argument('--model-depth', type=int, default=16, choices=[16, 20, 24, 30], 
                       help='Model depth (16, 20, 24, or 30)')
    parser.add_argument('--vae_path', type=str, default='', 
                       help='Path to VAE checkpoint')
    parser.add_argument('--model_path', type=str, default='', 
                       help='Path to VAR checkpoint')
    parser.add_argument('--device', type=str, default='cuda', 
                       help='Device to use (cuda/cpu)')
    
    # Generation parameters
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
    
    # Layer cache-skip configuration
    parser.add_argument('--stage_layer_skip', type=str, default='',
                        help='Stage-specific layer cache-skip. Format: "stage1:layer1,layer2;stage2:layer3,layer4". Example: "9:0,1,2" uses cache for layers 0,1,2 in stage 9')
    parser.add_argument('--skip_layers_below', type=str, default='',
                        help='Use cache for all layers below threshold in specific stages. Format: "stage1:threshold1;stage2:threshold2". Example: "9:3" uses cache for layers 0,1,2 in stage 9')
    parser.add_argument('--skip_layers_above', type=str, default='',
                        help='Use cache for all layers above threshold in specific stages. Format: "stage1:threshold1;stage2:threshold2". Example: "9:13" uses cache for layers 14,15 in stage 9')
    parser.add_argument('--disable_cache_skip', action='store_true',
                        help='Disable all layer cache-skipping')
    parser.add_argument('--interpolation_mode', type=str, default='bilinear',
                        choices=['bilinear', 'nearest', 'bicubic'],
                        help='Interpolation mode for cached features')
    parser.add_argument('--cache_blend_ratio', type=float, default=0.8,
                        help='Ratio for blending cached and current features (0.0-1.0, 1.0=pure cache)')
    parser.add_argument('--adaptive_cache_selection', action='store_true',
                        help='Enable adaptive cache selection based on similarity')
    
    # Operation modes
    parser.add_argument('--benchmark', action='store_true',
                        help='Run generation speed benchmark')
    parser.add_argument('--generate', action='store_true',
                        help='Generate samples')
    parser.add_argument('--generate_fid', action='store_true',
                        help='Generate samples and compute FID')
    parser.add_argument('--save_images', action='store_true',
                        help='Save generated images')
    
    # FID computation arguments
    parser.add_argument('--fid_statistics_file', type=str, 
                        default='/home/wyj24/project/lpd/fid_stats/adm_in256_stats.npz',
                        help='Path to FID statistics file')
    
    return parser.parse_args()


def create_layer_cache_skip_config_from_args(args) -> LayerCacheSkipConfig:
    """Create LayerCacheSkipConfig from command line arguments"""
    stage_layer_skip = parse_stage_layer_skip(args.stage_layer_skip)
    skip_all_layers_below = parse_stage_threshold_skip(args.skip_layers_below)
    skip_all_layers_above = parse_stage_threshold_skip(args.skip_layers_above)
    
    return LayerCacheSkipConfig(
        stage_layer_skip=stage_layer_skip,
        skip_all_layers_below=skip_all_layers_below,
        skip_all_layers_above=skip_all_layers_above,
        enable_skip=not args.disable_cache_skip,
        interpolation_mode=args.interpolation_mode,
        cache_blend_ratio=args.cache_blend_ratio,
        adaptive_cache_selection=args.adaptive_cache_selection
    )


def generate_output_dir_name(base_dir: str, args, layer_cache_skip_config: LayerCacheSkipConfig) -> str:
    """Generate output directory name that includes layer cache-skip parameters"""
    components = []
    
    # Model configuration
    components.append(f"d{args.model_depth}")
    components.append(f"cfg{args.cfg}")
    components.append(f"bs{args.batch_size}")
    components.append(f"samples{args.num_samples}")
    components.append(f"seed{args.seed}")
    
    # Layer cache-skip configuration
    if layer_cache_skip_config.enable_skip:
        if layer_cache_skip_config.stage_layer_skip:
            skip_str = "_".join([f"s{k}l{'_'.join(map(str, sorted(v)))}" 
                               for k, v in layer_cache_skip_config.stage_layer_skip.items()])
            components.append(f"cacheskip_{skip_str}")
        
        if layer_cache_skip_config.skip_all_layers_below:
            below_str = "_".join([f"s{k}below{v}" 
                                for k, v in layer_cache_skip_config.skip_all_layers_below.items()])
            components.append(f"cachebelow_{below_str}")
        
        if layer_cache_skip_config.skip_all_layers_above:
            above_str = "_".join([f"s{k}above{v}" 
                                for k, v in layer_cache_skip_config.skip_all_layers_above.items()])
            components.append(f"cacheabove_{above_str}")
        
        # Cache blend ratio if not default
        if layer_cache_skip_config.cache_blend_ratio != 0.8:
            components.append(f"blend{layer_cache_skip_config.cache_blend_ratio:.1f}")
        
        # Interpolation mode if not default
        if layer_cache_skip_config.interpolation_mode != 'bilinear':
            components.append(f"interp{layer_cache_skip_config.interpolation_mode}")
    else:
        components.append("no_cache_skip")
    
    # TF32 flag
    if args.tf32:
        components.append("tf32")
    
    # More smooth flag
    if args.more_smooth:
        components.append("smooth")
    
    dir_name = "var_cache_skip_" + "_".join(components)
    return os.path.join(base_dir, dir_name)


def main():
    args = parse_args()
    
    # Create layer cache-skip configuration
    layer_cache_skip_config = create_layer_cache_skip_config_from_args(args)
    
    # Generate descriptive output directory name
    base_output_dir = args.output_dir
    param_output_dir = generate_output_dir_name(base_output_dir, args, layer_cache_skip_config)
    os.makedirs(param_output_dir, exist_ok=True)
    
    # Update args.output_dir to use the parameter-specific directory
    args.output_dir = param_output_dir
    print(f"Using parameter-specific output directory: {param_output_dir}")
    
    print(f"\nLayer Cache-Skip Configuration:")
    print(f"{'='*60}")
    print(layer_cache_skip_config.get_skip_summary(10, args.model_depth))  # 10 stages for default patch_nums
    print(f"Cache blend ratio: {layer_cache_skip_config.cache_blend_ratio}")
    print(f"Interpolation mode: {layer_cache_skip_config.interpolation_mode}")
    print(f"Adaptive cache selection: {layer_cache_skip_config.adaptive_cache_selection}")
    print(f"{'='*60}\n")
    
    ################## 1. Download checkpoints and build models
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
    
    # build vae, var
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    device = args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu'
    
    print("Building VAR model with layer cache-skip...")
    vae, var = build_vae_var_cache_skip(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=args.model_depth, shared_aln=False,
        layer_cache_skip_config=layer_cache_skip_config
    )
    
    # load checkpoints
    print("Loading model checkpoints...")
    vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)
    vae.eval(), var.eval()
    for p in vae.parameters(): p.requires_grad_(False)
    for p in var.parameters(): p.requires_grad_(False)
    print(f'Model preparation finished.')
    
    ############################# 2. Run evaluation modes
    
    # seed
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
    
    # Convert to float16 for efficiency
    vae, var = vae.to(torch.float16), var.to(torch.float16)
    
    print(f"Model loaded on {device}. Using dtype: {next(var.parameters()).dtype}")
    print(f"Model parameters: {sum(p.numel() for p in var.parameters()):,}")
    
    results = {}
    
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
    
    # Generate samples and compute FID
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
                        more_smooth=args.more_smooth
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
            'layer_cache_skip_config': {
                'stage_layer_skip': layer_cache_skip_config.stage_layer_skip,
                'skip_all_layers_below': layer_cache_skip_config.skip_all_layers_below,
                'skip_all_layers_above': layer_cache_skip_config.skip_all_layers_above,
                'enable_skip': layer_cache_skip_config.enable_skip,
                'interpolation_mode': layer_cache_skip_config.interpolation_mode,
                'cache_blend_ratio': layer_cache_skip_config.cache_blend_ratio,
                'adaptive_cache_selection': layer_cache_skip_config.adaptive_cache_selection,
                'skip_summary': layer_cache_skip_config.get_skip_summary(10, args.model_depth)
            }
        }
        
        results_path = os.path.join(args.output_dir, 'cache_skip_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)  # default=str to handle numpy types
        print(f"\nAll evaluation results saved to {results_path}")
        print(f"Parameter-specific output directory: {args.output_dir}")
    
    print("\nVAR layer cache-skip evaluation completed!")


if __name__ == '__main__':
    main()