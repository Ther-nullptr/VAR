import math
from functools import partial
from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

import dist
from models.basic_var_enhanced import AdaLNBeforeHead, AdaLNSelfAttnEnhanced, CacheConfig
from models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from models.vqvae import VQVAE, VectorQuantizer2


class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)   # B16C


class VAREnhanced(nn.Module, PyTorchModelHubMixin):
    """
    Enhanced VAR model with flexible multi-stage caching
    """
    def __init__(
        self, vae_local: VQVAE,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., 
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True,
        # Enhanced caching parameters
        skip_stages: List[int] = None,          # Stages to skip (e.g., [169, 256])
        cache_stages: List[int] = None,         # Stages to cache (e.g., [100, 169])
        enable_attn_cache: bool = True,         # Enable attention caching
        enable_mlp_cache: bool = True,          # Enable MLP caching
        cache_threshold: float = 0.7,           # Cache similarity threshold
        max_skip_stages: int = 9,               # Maximum stages to skip
        adaptive_threshold: bool = False,       # Use adaptive thresholding
        interpolation_mode: str = 'bilinear',   # Interpolation mode
        # Legacy parameters for compatibility
        use_cache=False, calibration=False, sim_path=None, threshold=0.7
    ):
        super().__init__()
        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        
        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1   # progressive training

        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2))
            cur += pn ** 2
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())
        
        # Enhanced cache configuration
        if use_cache:  # Legacy compatibility
            skip_stages = skip_stages or [169, 256]
            cache_stages = cache_stages or [100, 169]
        
        self.cache_config = CacheConfig(
            skip_stages=skip_stages or [],
            cache_stages=cache_stages or [],
            enable_attn_cache=enable_attn_cache,
            enable_mlp_cache=enable_mlp_cache,
            threshold=cache_threshold or threshold,
            max_skip_stages=max_skip_stages,
            adaptive_threshold=adaptive_threshold,
            interpolation_mode=interpolation_mode
        )
        
        # Cache state
        self.calibration_mode = calibration
        self.use_cache = bool(self.cache_config.skip_stages or self.cache_config.cache_stages)
        
        # 1. input (word) embedding
        quant: VectorQuantizer2 = vae_local.quantize
        self.vae_proxy: Tuple[VQVAE] = (vae_local,)
        self.vae_quant_proxy: Tuple[VectorQuantizer2] = (quant,)
        self.word_embed = nn.Linear(self.Cvae, self.C)
        
        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=dist.get_device())
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        
        # 3. absolute position embedding
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn*pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        # level embedding
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # 4. backbone blocks with enhanced caching
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            AdaLNSelfAttnEnhanced(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, 
                num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], 
                last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
                cache_config=self.cache_config,
            )
            for block_idx in range(depth)
        ])
        
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        
        # 5. output head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer)
        self.head = nn.Linear(self.C, self.V, bias=False)
        
        # 6. initialize cache similarity matrices
        self.init_cache_similarity()
        
        # Load similarity data if provided
        if sim_path and self.use_cache:
            self.load_similarity_data(sim_path)
        
        print(f'[VAREnhanced] Enhanced caching enabled: {self.use_cache}')
        if self.use_cache:
            print(f'[VAREnhanced] Skip stages: {self.cache_config.skip_stages}')
            print(f'[VAREnhanced] Cache stages: {self.cache_config.cache_stages}')
            print(f'[VAREnhanced] Attention cache: {self.cache_config.enable_attn_cache}')
            print(f'[VAREnhanced] MLP cache: {self.cache_config.enable_mlp_cache}')
            print(f'[VAREnhanced] Threshold: {self.cache_config.threshold}')
            print(f'[VAREnhanced] Adaptive threshold: {self.cache_config.adaptive_threshold}')
    
    def init_cache_similarity(self):
        """Initialize cache similarity matrices"""
        if not self.use_cache:
            self.cache_similarity_attn = None
            self.cache_similarity_mlp = None
            return
        
        # Initialize similarity matrices for all blocks and stages
        max_iterations = len(self.patch_nums)
        self.cache_similarity_attn = [
            [0.0] * max_iterations for _ in range(self.depth)
        ]
        self.cache_similarity_mlp = [
            [0.0] * max_iterations for _ in range(self.depth)
        ]
    
    def load_similarity_data(self, sim_path: str):
        """Load pre-computed similarity data"""
        try:
            data = torch.load(sim_path, map_location='cpu')
            if 'cache_similarity_attn' in data:
                self.cache_similarity_attn = data['cache_similarity_attn']
            if 'cache_similarity_mlp' in data:
                self.cache_similarity_mlp = data['cache_similarity_mlp']
            print(f'[VAREnhanced] Loaded similarity data from {sim_path}')
        except Exception as e:
            print(f'[VAREnhanced] Failed to load similarity data from {sim_path}: {e}')
    
    def save_similarity_data(self, sim_path: str):
        """Save computed similarity data"""
        if not self.use_cache:
            return
        
        data = {
            'cache_similarity_attn': self.cache_similarity_attn,
            'cache_similarity_mlp': self.cache_similarity_mlp,
            'cache_config': {
                'skip_stages': self.cache_config.skip_stages,
                'cache_stages': self.cache_config.cache_stages,
                'threshold': self.cache_config.threshold,
            }
        }
        torch.save(data, sim_path)
        print(f'[VAREnhanced] Saved similarity data to {sim_path}')
    
    def set_cache_config(self, cache_config: CacheConfig):
        """Update cache configuration"""
        self.cache_config = cache_config
        self.use_cache = bool(cache_config.skip_stages or cache_config.cache_stages)
        
        # Update cache config in all blocks
        for block in self.blocks:
            block.cache_config = cache_config
            block.attn.cache_config = cache_config
            block.ffn.cache_config = cache_config
        
        # Reinitialize similarity matrices if needed
        if self.use_cache and self.cache_similarity_attn is None:
            self.init_cache_similarity()
    
    def enable_calibration_mode(self, enable: bool = True):
        """Enable/disable calibration mode"""
        self.calibration_mode = enable
    
    def get_cache_statistics(self):
        """Get cache usage statistics"""
        if not self.use_cache:
            return {}
        
        stats = {
            'attn_similarities': self.cache_similarity_attn,
            'mlp_similarities': self.cache_similarity_mlp,
            'effective_thresholds': {}
        }
        
        # Get adaptive thresholds if enabled
        if self.cache_config.adaptive_threshold:
            for i, block in enumerate(self.blocks):
                stats['effective_thresholds'][f'attn_block_{i}'] = block.attn.get_effective_threshold()
                stats['effective_thresholds'][f'mlp_block_{i}'] = block.ffn.get_effective_threshold()
        
        return stats
    
    def extra_repr(self):
        return f'patch_nums={self.patch_nums}, skip_stages={self.cache_config.skip_stages}, cache_stages={self.cache_config.cache_stages}'
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Override state_dict to exclude cache data"""
        state = super().state_dict(destination, prefix, keep_vars)
        
        # Remove cache-related state that shouldn't be saved with model weights
        cache_keys = [k for k in state.keys() if 'cache' in k.lower() and 'config' not in k.lower()]
        for key in cache_keys:
            state.pop(key, None)
        
        return state
    
    def forward(self, label_B: Optional[torch.LongTensor], x_BLCv_wo_first_l: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with enhanced caching
        :param label_B: class labels
        :param x_BLCv_wo_first_l: input tokens without first level
        :return: logits for next token prediction
        """
        with torch.cuda.amp.autocast(enabled=False):
            B = x_BLCv_wo_first_l.shape[0]
            
            # Initialize cache if needed
            cache_attn = [None] * self.depth if self.use_cache else None
            cache_mlp = [None] * self.depth if self.use_cache else None
            
            # Class conditioning
            if label_B is None:
                label_B = torch.multinomial(self.uniform_prob.expand(B, -1), num_samples=1, generator=self.rng).reshape(B)
            elif self.training and self.cond_drop_rate > 0:
                mask = torch.rand(B, device=label_B.device) < self.cond_drop_rate
                label_B = torch.where(mask, self.num_classes, label_B)
            
            class_emb = self.class_emb(label_B)
            
            # Process input tokens
            x_BLC = self.word_embed(x_BLCv_wo_first_l.view(B, -1, self.Cvae).float())
            
            # Add positional embeddings
            pos_emb = self.pos_1LC[:, :x_BLC.shape[1], :] 
            lvl_emb = torch.cat([
                self.lvl_embed(torch.full((1, pn*pn), i, dtype=torch.long, device=x_BLC.device))
                for i, pn in enumerate(self.patch_nums) if sum(pn*pn for pn in self.patch_nums[:i]) < x_BLC.shape[1]
            ], dim=1)[:, :x_BLC.shape[1], :]
            
            x_BLC = x_BLC + pos_emb + lvl_emb
            
            # Prepare shared AdaLN
            if isinstance(self.shared_ada_lin, nn.Identity):
                cond_BD = class_emb.unsqueeze(1).expand(B, 1, self.D)
            else:
                cond_BD = self.shared_ada_lin(class_emb.unsqueeze(1).expand(B, 1, self.D))
            
            # Transformer blocks with enhanced caching
            attn_bias = None  # Could be used for causal masking
            
            for block in self.blocks:
                x_BLC = block(
                    x_BLC, cond_BD, attn_bias,
                    cache_mlp, cache_attn,
                    self.cache_similarity_mlp, self.cache_similarity_attn,
                    calibration=self.calibration_mode
                )
            
            # Output head
            x_BLC = self.head_nm(x_BLC, cond_BD)
            logits_BLV = self.head(x_BLC)
            
            return logits_BLV
    
    @torch.inference_mode()
    def autoregressive_infer_cfg(
        self, B: int, label_B: Optional[torch.Tensor], g_seed: Optional[int], 
        cfg=1.5, top_k=0, top_p=0.96, more_smooth=False, **kwargs
    ):
        """
        Autoregressive inference with enhanced caching
        """
        if g_seed is None: 
            rng = None
        else: 
            self.rng.manual_seed(g_seed)
            rng = self.rng
        
        # Initialize cache
        if self.use_cache:
            cache_attn = [None] * self.depth
            cache_mlp = [None] * self.depth
        else:
            cache_attn = cache_mlp = None
        
        device = next(self.parameters()).device
        if label_B is None:
            label_B = torch.multinomial(self.uniform_prob.expand(B, -1), num_samples=1, generator=rng).view(B)
        
        # Prepare conditioning
        sos = cond_BD = self.class_emb(label_B).unsqueeze(1).expand(B, 1, self.D)
        if not isinstance(self.shared_ada_lin, nn.Identity):
            cond_BD = self.shared_ada_lin(cond_BD)
        
        # Autoregressive generation
        lvl_pos = 0
        cur_L = 0
        f_hat = sos.new_zeros(B, self.L, self.Cvae, dtype=torch.float32)
        
        for b, (begin, end) in enumerate(self.begin_ends):
            if self.prog_si >= 0 and b > self.prog_si:
                break
            
            cur_L += end - begin
            
            # Forward pass for current level
            x_BLC = self.word_embed(f_hat[:, :cur_L].float())
            x_BLC = x_BLC + self.pos_1LC[:, :cur_L] + self.lvl_embed.weight[:b+1].unsqueeze(0).expand_as(x_BLC)
            
            # Apply transformer blocks
            for block in self.blocks:
                x_BLC = block(
                    x_BLC, cond_BD, None,  # attn_bias=None during inference
                    cache_mlp, cache_attn,
                    self.cache_similarity_mlp, self.cache_similarity_attn,
                    calibration=False
                )
            
            # Predict next tokens
            x_BLC = self.head_nm(x_BLC, cond_BD)
            logits_BLV = self.head(x_BLC)
            
            # Sample tokens for current level
            if b == self.num_stages_minus_1:
                # Last stage
                f_hat[:, begin:end] = self.vae_proxy[0].quantize.embedding.weight[
                    sample_with_top_k_top_p_(logits_BLV[:, begin:end], rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
                ]
            else:
                # Intermediate stages
                t = cfg * 2 if more_smooth else cfg
                gumbel_hard_or_soft = True if more_smooth else False
                f_hat[:, begin:end] = gumbel_softmax_with_rng(
                    logits_BLV[:, begin:end].div(t), hard=gumbel_hard_or_soft, dim=-1, rng=rng
                ) @ self.vae_proxy[0].quantize.embedding.weight.unsqueeze(0)
        
        # Decode to images
        return self.vae_proxy[0].fhat_to_img(f_hat).add(1).div(2).clamp(0, 1)


# Factory function for easy model creation
def create_var_enhanced_model(
    vae_local: VQVAE,
    skip_stages: List[int] = None,
    cache_stages: List[int] = None,
    enable_attn_cache: bool = True,
    enable_mlp_cache: bool = True,
    **kwargs
) -> VAREnhanced:
    """
    Factory function to create VAREnhanced model with specified cache configuration
    """
    return VAREnhanced(
        vae_local=vae_local,
        skip_stages=skip_stages,
        cache_stages=cache_stages,
        enable_attn_cache=enable_attn_cache,
        enable_mlp_cache=enable_mlp_cache,
        **kwargs
    )