import argparse
import os
import os.path as osp
import torch, torchvision
import random
from tqdm import tqdm
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from models import VQVAE, build_vae_var

import torch_fidelity

def parse_args():
    parser = argparse.ArgumentParser(description='VAR model sampling with configurable parameters')
    parser.add_argument('--model-depth', type=int, default=16, choices=[16, 20, 24, 30], 
                       help='Model depth (16, 20, 24, or 30)')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--num-sampling-steps', type=int, default=250, 
                       help='Number of sampling steps (0-1000)')
    parser.add_argument('--cfg', type=float, default=1.5, 
                       help='Classifier-free guidance scale (1-10)')
    parser.add_argument('--more-smooth', action='store_true', 
                       help='Enable for smoother output')
    parser.add_argument('--batch-size', type=int, default=64, 
                       help='Batch size for sampling')
    parser.add_argument('--output-dir', type=str, default='./samples', 
                       help='Output directory for generated images')
    parser.add_argument('--tf32', action='store_true', 
                       help='Enable TF32 for faster computation')
    parser.add_argument('--samples-per-class', type=int, default=50, 
                       help='Number of samples to generate per class')
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='Threshold for cache similarity')
    parser.add_argument('--use-cache', action='store_true',
                       help='Enable cache usage for faster inference')
    parser.add_argument('--calibration', action='store_true',
                       help='Enable calibration for cache similarity')
    parser.add_argument('--sim-path', type=str, default=None,
                       help='Path to save or load cache similarity data')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    ################## 1. Download checkpoints and build models
    # download checkpoint
    hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
    var_ckpt_dir = '/home/wyj24/models/VAR'
    vae_ckpt, var_ckpt = f'{var_ckpt_dir}/vae_ch160v4096z32.pth', f'{var_ckpt_dir}/var_d{args.model_depth}_new.pth'
    if not osp.exists(vae_ckpt): os.system(f'wget {hf_home}/{vae_ckpt}')
    if not osp.exists(var_ckpt): os.system(f'wget {hf_home}/{var_ckpt}')

    # build vae, var
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if 'vae' not in globals() or 'var' not in globals():
        vae, var = build_vae_var(
            V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
            device=device, patch_nums=patch_nums,
            num_classes=1000, depth=args.model_depth, shared_aln=False,
            use_cache=args.use_cache, calibration=args.calibration, sim_path=args.sim_path,
            threshold=args.threshold
        )
    # import ipdb; ipdb.set_trace()
    # load checkpoints
    vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)
    vae.eval(), var.eval()
    for p in vae.parameters(): p.requires_grad_(False)
    for p in var.parameters(): p.requires_grad_(False)
    print(f'prepare finished.')

    ############################# 2. Sample with classifier-free guidance

    # seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # run faster
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.set_float32_matmul_precision('high' if args.tf32 else 'highest')
    
    vae, var = vae.to(torch.float16), var.to(torch.float16)

    # sample
    B = args.batch_size
    samples_per_class = args.samples_per_class
    iterations = (samples_per_class + B - 1) // B  # Ceiling division
    
    # generate images
    output_dir = f'var_d{args.model_depth}_cfg{args.cfg}_seed{args.seed}_cache_{args.threshold}_10_13_16_attn'
    if not osp.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    for img_cls in tqdm(range(1000)):
        for i in range(iterations):
            current_batch = min(B, samples_per_class - i * B)
            label_B = torch.tensor([img_cls] * current_batch, device=device)
            
            # with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):
            recon_B3HW = var.autoregressive_infer_cfg(
                B=current_batch, 
                label_B=label_B, 
                cfg=args.cfg, 
                top_k=900, 
                top_p=0.96, 
                g_seed=args.seed, 
                more_smooth=False
            )
            bchw = recon_B3HW.permute(0, 2, 3, 1).mul_(255).cpu().numpy()
            bchw = bchw.astype(np.uint8)
            for j in range(current_batch):
                img = PImage.fromarray(bchw[j])
                img.save(osp.join(output_dir, f"sample_{img_cls * samples_per_class + i * B + j}.png"))
                
    # compute FID
    if not args.calibration:
        fid_statistics_file = '/home/wyj24/project/lpd/fid_stats/adm_in256_stats.npz'
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=output_dir,
            input2=None,
            fid_statistics_file=fid_statistics_file,
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
        # after evaluation, the generated images will be deleted
        for img_file in os.listdir(output_dir):
            if img_file.endswith('.png'):
                # extract the sample index from the filename
                sample_index = int(img_file.split('_')[-1].split('.')[0])
                if sample_index % samples_per_class != 0:
                    os.remove(osp.join(output_dir, img_file))
        print(f"Generated images saved in {output_dir}.")
    else:
        # save the calibration values
        cache_similarity_mlp = var.cache_similarity_mlp
        cache_similarity_attn = var.cache_similarity_attn
        data = {
            'mlp': cache_similarity_mlp,
            'attn': cache_similarity_attn
        }
        torch.save(data, osp.join(output_dir, 'calibration_threshold.pt'))

if __name__ == '__main__':
    main()