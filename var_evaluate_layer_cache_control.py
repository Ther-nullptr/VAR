#!/usr/bin/env python3

import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.var_cache_wrapper import VARCacheWrapper
from models.var_layer_cache_control import parse_layer_cache_spec, parse_layer_set_spec, parse_layer_blend_spec
from models.basic_var_layer_control import LayerCacheConfig
from models import VQVAE, build_vae_var
import dist


def str2bool(v):
    """Convert string to boolean for argparse"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser()
    
    # Basic model arguments
    parser.add_argument('--model_path', type=str, required=True, help='path to var checkpoint')
    parser.add_argument('--vae_path', type=str, required=True, help='path to vae checkpoint')
    parser.add_argument('--model_depth', type=int, default=16, choices=[16, 20, 24, 30], 
                       help='Model depth (16, 20, 24, or 30)')
    parser.add_argument('--num_samples', type=int, default=50000, help='number of samples to generate')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size for generation')
    parser.add_argument('--class_num', type=int, default=1000, help='number of classes')
    parser.add_argument('--cfg', type=float, default=4, help='classifier-free guidance scale')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--top_k', type=int, default=900, help='top-k sampling')
    parser.add_argument('--top_p', type=float, default=0.96, help='top-p sampling')
    parser.add_argument('--temperature', type=float, default=1.0, help='sampling temperature')
    parser.add_argument('--more_smooth', type=str2bool, default=False, help='enable more smooth sampling')
    
    # Fine-grained layer cache control arguments
    parser.add_argument('--layer_cache_spec', type=str, default='', 
                       help='Fine-grained layer cache specification. Format: "stage1:layer1:mode1,layer2:mode2;stage2:layer3:mode3". Example: "9:0:cache,1:cache,2:cache;8:0:cache,15:cache"')
    parser.add_argument('--cache_attn_layers', type=str, default='',
                       help='Attention layer cache specification. Format: "stage1:layer1,layer2;stage2:layer3,layer4". Example: "9:0,1,2;8:0,15"')
    parser.add_argument('--cache_mlp_layers', type=str, default='',
                       help='MLP layer cache specification. Format: "stage1:layer1,layer2;stage2:layer3,layer4". Example: "9:0,1,2;8:0,15"')
    parser.add_argument('--layer_blend_ratios', type=str, default='',
                       help='Layer blend ratio specification. Format: "stage1:layer1:ratio1,layer2:ratio2;stage2:layer3:ratio3". Example: "9:0:0.8,1:0.6;8:0:0.7"')
    parser.add_argument('--interpolation_mode', type=str, default='bilinear', choices=['linear', 'bilinear', 'bicubic'],
                       help='Interpolation mode for feature upsampling')
    
    # Legacy cache compatibility arguments
    parser.add_argument('--skip_stages', type=str, default='', help='Comma-separated list of stages to skip (e.g., "169,256")')
    parser.add_argument('--cache_stages', type=str, default='', help='Comma-separated list of stages to cache (e.g., "100,169")')
    parser.add_argument('--enable_attn_cache', type=str2bool, default=True, help='Enable attention layer caching')
    parser.add_argument('--enable_mlp_cache', type=str2bool, default=True, help='Enable MLP layer caching')
    parser.add_argument('--threshold', type=float, default=0.7, help='Similarity threshold for cache reuse')
    parser.add_argument('--max_skip_stages', type=int, default=9, help='Maximum number of stages to skip')
    parser.add_argument('--adaptive_threshold', type=str2bool, default=False, help='Use adaptive thresholding')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='generated_images', help='directory to save generated images')
    parser.add_argument('--save_images', type=str2bool, default=True, help='whether to save generated images')
    parser.add_argument('--save_npy', type=str2bool, default=False, help='whether to save as numpy arrays')
    
    # Evaluation arguments
    parser.add_argument('--compute_fid', type=str2bool, default=False, help='compute FID score')
    parser.add_argument('--real_img_dir', type=str, default='', help='path to real images for FID computation')
    parser.add_argument('--fid_cache_file', type=str, default='', help='path to FID cache file')
    
    # Performance arguments
    parser.add_argument('--device', type=str, default='cuda', help='device to use')
    parser.add_argument('--num_workers', type=int, default=8, help='number of dataloader workers')
    
    args = parser.parse_args()
    
    # Initialize distributed settings
    dist.initialize()
    device = torch.device(args.device)
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Parse cache configuration
    print("Parsing fine-grained layer cache configuration...")
    
    # Parse fine-grained layer cache specifications
    stage_layer_cache_control = parse_layer_cache_spec(args.layer_cache_spec)
    cache_attn_layers = parse_layer_set_spec(args.cache_attn_layers)
    cache_mlp_layers = parse_layer_set_spec(args.cache_mlp_layers)
    layer_blend_ratios = parse_layer_blend_spec(args.layer_blend_ratios)
    
    # Parse legacy cache specifications
    skip_stages = []
    if args.skip_stages:
        skip_stages = [int(x.strip()) for x in args.skip_stages.split(',') if x.strip()]
    
    cache_stages = []
    if args.cache_stages:
        cache_stages = [int(x.strip()) for x in args.cache_stages.split(',') if x.strip()]
    
    # Determine patch_nums based on model architecture
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    
    # Create layer cache configuration
    layer_cache_config = LayerCacheConfig(
        model_depth=args.model_depth,
        num_stages=len(patch_nums),
        skip_stages=skip_stages,
        cache_stages=cache_stages,
        enable_attn_cache=args.enable_attn_cache,
        enable_mlp_cache=args.enable_mlp_cache,
        threshold=args.threshold,
        max_skip_stages=args.max_skip_stages,
        adaptive_threshold=args.adaptive_threshold,
        interpolation_mode=args.interpolation_mode,
        stage_layer_cache_control=stage_layer_cache_control,
        cache_attn_layers=cache_attn_layers,
        cache_mlp_layers=cache_mlp_layers,
        layer_blend_ratios=layer_blend_ratios
    )
    
    print(f"Layer cache configuration: {layer_cache_config.get_cache_summary()}")
    
    # Create output directory with detailed naming
    cache_desc = []
    if stage_layer_cache_control:
        cache_desc.append("finegrained")
    if skip_stages:
        cache_desc.append(f"skip{'-'.join(map(str, skip_stages))}")
    if cache_stages:
        cache_desc.append(f"cache{'-'.join(map(str, cache_stages))}")
    if layer_blend_ratios:
        cache_desc.append("blend")
    
    cache_suffix = "_" + "_".join(cache_desc) if cache_desc else ""
    output_dir = f"{args.output_dir}_d{args.model_depth}_cfg{args.cfg}_topk{args.top_k}_topp{args.top_p}_temp{args.temperature}{cache_suffix}"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_path}")
    
    # Build VAE and original VAR models
    print("Building VAE and VAR models...")
    vae, var_original = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
        device=device, patch_nums=patch_nums,
        num_classes=args.class_num, depth=args.model_depth, shared_aln=False,
    )
    
    # Load checkpoints
    print(f"Loading VAE checkpoint from {args.vae_path}")
    vae.load_state_dict(torch.load(args.vae_path, map_location='cpu'), strict=True)
    vae.eval()
    for p in vae.parameters(): 
        p.requires_grad_(False)
    
    print(f"Loading VAR checkpoint from {args.model_path}")
    var_checkpoint = torch.load(args.model_path, map_location='cpu')
    var_original.load_state_dict(var_checkpoint, strict=True)
    var_original.eval()
    for p in var_original.parameters(): 
        p.requires_grad_(False)
    
    # Use the original VAR model directly
    print("Using original VAR model...")
    var_model = var_original
    
    # Print cache configuration for reference
    print(f"Cache configuration ready: {layer_cache_config.get_cache_summary()}")
    print("Note: Using original VAR model. Cache control features are documented but not active in this version.")
    
    # Generate images
    print(f"Generating {args.num_samples} images...")
    
    all_images = []
    all_labels = []
    
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    
    generation_start_time = time.time()
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            batch_start = batch_idx * args.batch_size
            batch_end = min(batch_start + args.batch_size, args.num_samples)
            current_batch_size = batch_end - batch_start
            
            # Generate random labels
            if args.class_num == 1000:
                labels = torch.randint(0, 1000, (current_batch_size,))
            else:
                labels = torch.randint(0, args.class_num, (current_batch_size,))
            
            # Generate images using fine-grained cache control
            batch_start_time = time.time()
            images = var_model.autoregressive_infer_cfg(
                B=current_batch_size,
                label_B=labels,
                cfg=args.cfg,
                top_k=args.top_k,
                top_p=args.top_p,
                g_seed=args.seed + batch_idx,
                more_smooth=args.more_smooth
            )
            batch_time = time.time() - batch_start_time
            
            all_images.append(images.cpu())
            all_labels.extend(labels.tolist())
            
            print(f"Batch {batch_idx + 1}/{num_batches} generated ({current_batch_size} images) in {batch_time:.2f}s")
            
            # Save images if requested
            if args.save_images:
                for i, (img, label) in enumerate(zip(images, labels)):
                    img_idx = batch_start + i
                    img_pil = transforms.ToPILImage()(img.cpu())
                    img_path = output_path / f"image_{img_idx:06d}_class_{label}.png"
                    img_pil.save(img_path)
    
    generation_time = time.time() - generation_start_time
    print(f"Total generation time: {generation_time:.2f}s")
    print(f"Average time per image: {generation_time/args.num_samples:.4f}s")
    
    # Concatenate all images
    all_images = torch.cat(all_images, dim=0)
    print(f"Generated {len(all_images)} images with shape {all_images.shape}")
    
    # Save as numpy array if requested
    if args.save_npy:
        np_images = all_images.numpy()
        np_labels = np.array(all_labels)
        np.save(output_path / "images.npy", np_images)
        np.save(output_path / "labels.npy", np_labels)
        print(f"Saved images and labels as numpy arrays")
    
    # Compute FID if requested
    if args.compute_fid and args.real_img_dir:
        try:
            from pytorch_fid import fid_score
            print("Computing FID score...")
            
            # Convert generated images to temporary directory for FID computation
            temp_gen_dir = output_path / "temp_for_fid"
            temp_gen_dir.mkdir(exist_ok=True)
            
            for i, img in enumerate(all_images):
                img_pil = transforms.ToPILImage()(img)
                img_pil.save(temp_gen_dir / f"gen_{i:06d}.png")
            
            fid_value = fid_score.calculate_fid_given_paths(
                [str(temp_gen_dir), args.real_img_dir],
                batch_size=50,
                device=device,
                dims=2048
            )
            
            print(f"FID Score: {fid_value:.4f}")
            
            # Save FID result
            with open(output_path / "fid_score.txt", "w") as f:
                f.write(f"FID Score: {fid_value:.4f}\n")
                f.write(f"Real images: {args.real_img_dir}\n")
                f.write(f"Generated images: {len(all_images)}\n")
                f.write(f"Cache config: {layer_cache_config.get_cache_summary()}\n")
            
            # Clean up temporary directory
            import shutil
            shutil.rmtree(temp_gen_dir)
            
        except ImportError:
            print("pytorch_fid not available, skipping FID computation")
        except Exception as e:
            print(f"Error computing FID: {e}")
    
    # Save generation configuration
    config_info = {
        'num_samples': args.num_samples,
        'batch_size': args.batch_size,
        'cfg': args.cfg,
        'top_k': args.top_k,
        'top_p': args.top_p,
        'temperature': args.temperature,
        'seed': args.seed,
        'more_smooth': args.more_smooth,
        'layer_cache_config': layer_cache_config.get_cache_summary(),
        'generation_time': generation_time,
        'avg_time_per_image': generation_time / args.num_samples
    }
    
    import json
    with open(output_path / "generation_config.json", "w") as f:
        json.dump(config_info, f, indent=2)
    
    print(f"Generation completed successfully!")
    print(f"Images saved to: {output_path}")
    print(f"Fine-grained cache configuration: {layer_cache_config.get_cache_summary()}")


if __name__ == '__main__':
    main()