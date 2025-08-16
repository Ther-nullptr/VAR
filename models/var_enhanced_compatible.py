"""
Enhanced VAR model with fine-grained layer cache control
This version maintains full compatibility with original VAR checkpoints
"""

import math
from functools import partial
from typing import Optional, Tuple, Union, List, Dict, Set

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

import dist
from models.basic_var_layer_control import LayerCacheConfig, feature_interpolate, length2iteration
from models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from models.vqvae import VQVAE, VectorQuantizer2

# Import original VAR components
from models.var import SharedAdaLin, AdaLNBeforeHead
from models.basic_var import AdaLNSelfAttn


class EnhancedAdaLNSelfAttn(AdaLNSelfAttn):
    """
    Enhanced version of AdaLNSelfAttn with fine-grained cache control
    Inherits from original to maintain checkpoint compatibility
    """
    
    def __init__(
        self, cond_dim: int, shared_aln: bool, block_idx: int,
        embed_dim: int, norm_layer: nn.Module = nn.LayerNorm, 
        num_heads: int = 8, mlp_ratio: float = 4.,
        drop: float = 0., attn_drop: float = 0., drop_path: float = 0.,
        last_drop_p: float = 0., attn_l2_norm: bool = False,
        flash_if_available: bool = True, fused_if_available: bool = True,
        use_cache: bool = False, calibration: bool = False, threshold: float = 0.7,
        # New parameters for fine-grained control
        layer_cache_config: LayerCacheConfig = None,
    ):
        # Initialize parent class with all original parameters
        super().__init__(
            cond_dim=cond_dim, shared_aln=shared_aln, block_idx=block_idx,
            embed_dim=embed_dim, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
            drop=drop, attn_drop=attn_drop, drop_path=drop_path, last_drop_p=last_drop_p,
            attn_l2_norm=attn_l2_norm, flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            use_cache=use_cache, calibration=calibration, threshold=threshold
        )
        
        # Add fine-grained cache control
        self.layer_cache_config = layer_cache_config
        self.current_stage = 0
        
        # Enhanced cache storage for fine-grained control
        self.fine_cache_attn = {}  # {stage_idx: cached_features}
        self.fine_cache_mlp = {}   # {stage_idx: cached_features}
    
    def set_current_stage(self, stage_idx: int):
        """Set current generation stage for cache decisions"""
        self.current_stage = stage_idx
    
    def forward(self, x, cond_BD, attn_bias, cache_mlp, cache_attn, cache_similarity_mlp, cache_similarity_attn):
        """Enhanced forward with fine-grained cache control"""
        
        # If no fine-grained config, use original behavior
        if self.layer_cache_config is None:
            return super().forward(x, cond_BD, attn_bias, cache_mlp, cache_attn, cache_similarity_mlp, cache_similarity_attn)
        
        L = x.shape[1]
        
        # Get original conditioning
        if self.shared_aln:
            gamma1, gamma2, scale1, scale2, shift1, shift2 = (self.ada_gss + cond_BD).unbind(2)
        else:
            gamma1, gamma2, scale1, scale2, shift1, shift2 = self.ada_lin(cond_BD).view(-1, 1, 6, self.C).unbind(2)
        
        # === Attention with fine-grained cache control ===
        attn_input = self.ln_wo_grad(x) * (scale1.add(1)) + shift1
        
        # Check if attention should use fine-grained cache
        should_cache_attn = self.layer_cache_config.should_cache_layer_attn(
            self.current_stage, self.block_idx, L
        )
        
        if should_cache_attn and self.current_stage in self.fine_cache_attn:
            # Use cached attention features
            cached_attn = self.fine_cache_attn[self.current_stage]
            
            if cached_attn.shape[1] != L:
                # Interpolate to current resolution
                interp_mode = self.layer_cache_config.get_layer_interpolation_mode(
                    self.current_stage, self.block_idx
                )
                cached_attn = feature_interpolate(cached_attn, L, mode=interp_mode)
            
            # Get blend ratio
            blend_ratio = self.layer_cache_config.get_layer_blend_ratio(
                self.current_stage, self.block_idx
            )
            
            if blend_ratio >= 1.0:
                # Pure cache
                attn_out = cached_attn
            elif blend_ratio > 0.0:
                # Blend cache and computation
                computed_attn = self.attn(attn_input, attn_bias=attn_bias, cache_attn=cache_attn, cache_similarity_attn=cache_similarity_attn)
                attn_out = blend_ratio * cached_attn + (1.0 - blend_ratio) * computed_attn
            else:
                # Normal computation
                attn_out = self.attn(attn_input, attn_bias=attn_bias, cache_attn=cache_attn, cache_similarity_attn=cache_similarity_attn)
        else:
            # Normal attention computation
            attn_out = self.attn(attn_input, attn_bias=attn_bias, cache_attn=cache_attn, cache_similarity_attn=cache_similarity_attn)
        
        # Store attention output for future stages if needed
        if self.layer_cache_config.should_cache_attn(L):
            self.fine_cache_attn[self.current_stage] = attn_out.detach().clone()
        
        # Apply attention output
        x = x + self.drop_path(attn_out.mul_(gamma1))
        
        # === FFN with fine-grained cache control ===
        ffn_input = self.ln_wo_grad(x) * (scale2.add(1)) + shift2
        
        # Check if FFN should use fine-grained cache
        should_cache_mlp = self.layer_cache_config.should_cache_layer_mlp(
            self.current_stage, self.block_idx, L
        )
        
        if should_cache_mlp and self.current_stage in self.fine_cache_mlp:
            # Use cached MLP features
            cached_mlp = self.fine_cache_mlp[self.current_stage]
            
            if cached_mlp.shape[1] != L:
                # Interpolate to current resolution
                interp_mode = self.layer_cache_config.get_layer_interpolation_mode(
                    self.current_stage, self.block_idx
                )
                cached_mlp = feature_interpolate(cached_mlp, L, mode=interp_mode)
            
            # Get blend ratio
            blend_ratio = self.layer_cache_config.get_layer_blend_ratio(
                self.current_stage, self.block_idx
            )
            
            if blend_ratio >= 1.0:
                # Pure cache
                ffn_out = cached_mlp
            elif blend_ratio > 0.0:
                # Blend cache and computation
                computed_ffn = self.ffn(ffn_input, cache_mlp=cache_mlp, cache_similarity_mlp=cache_similarity_mlp)
                ffn_out = blend_ratio * cached_mlp + (1.0 - blend_ratio) * computed_ffn
            else:
                # Normal computation
                ffn_out = self.ffn(ffn_input, cache_mlp=cache_mlp, cache_similarity_mlp=cache_similarity_mlp)
        else:
            # Normal FFN computation
            ffn_out = self.ffn(ffn_input, cache_mlp=cache_mlp, cache_similarity_mlp=cache_similarity_mlp)
        
        # Store FFN output for future stages if needed
        if self.layer_cache_config.should_cache_mlp(L):
            self.fine_cache_mlp[self.current_stage] = ffn_out.detach().clone()
        
        # Apply FFN output
        x = x + self.drop_path(ffn_out.mul(gamma2))
        
        return x


class VAREnhancedCompatible(nn.Module, PyTorchModelHubMixin):
    """
    Enhanced VAR model with fine-grained layer cache control
    Maintains full compatibility with original VAR checkpoints
    """
    
    def __init__(
        self, vae_local: VQVAE,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., 
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True,
        use_cache=False, calibration=False, sim_path=None, threshold=0.7,
        # Enhanced parameters
        layer_cache_config: LayerCacheConfig = None,
    ):
        super().__init__()
        
        # 0. hyperparameters (same as original)
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
        
        # Fine-grained layer cache configuration
        if layer_cache_config is None:
            layer_cache_config = LayerCacheConfig(
                model_depth=depth,
                num_stages=len(patch_nums)
            )
        self.layer_cache_config = layer_cache_config
        
        # 1. input (word) embedding (same as original)
        quant: VectorQuantizer2 = vae_local.quantize
        self.vae_proxy: Tuple[VQVAE] = (vae_local,)
        self.vae_quant_proxy: Tuple[VectorQuantizer2] = (quant,)
        self.word_embed = nn.Linear(self.Cvae, self.C)
        
        # 2. class embedding (same as original)
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=dist.get_device())
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        
        # 3. absolute position embedding (same as original)
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn*pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        
        # level embedding (same as original)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # 4. backbone blocks (ENHANCED)
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Use enhanced blocks instead of original ones
        self.blocks = nn.ModuleList([
            EnhancedAdaLNSelfAttn(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
                use_cache=use_cache, calibration=calibration, threshold=threshold,
                # Enhanced parameter
                layer_cache_config=layer_cache_config,
            )
            for block_idx in range(depth)
        ])
        
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[Enhanced VAR]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})\n'
            f'    [cache config] {layer_cache_config.get_cache_summary()}',
            end='\n\n', flush=True
        )
        
        # 5. attention mask (same as original)
        d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        
        # 6. classifier head (same as original)
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)
        
        # 7. cache for autoregressive inference (same as original)
        self.use_cache = use_cache
        self.calibration = calibration
        if self.use_cache:
            self.cache_mlp = [None for _ in range(depth)]
            self.cache_attn = [None for _ in range(depth)]
            
            if self.calibration:
                self.cache_similarity_mlp = torch.zeros((depth, len(patch_nums)-1))
                self.cache_similarity_attn = torch.zeros((depth, len(patch_nums)-1))
                print('initial calibration for similarity')
            else:
                if sim_path:
                    data = torch.load(sim_path)
                    self.cache_similarity_mlp = data['mlp']
                    self.cache_similarity_attn = data['attn']
                    print(f'load similarity data from {sim_path}')
                else:
                    self.cache_similarity_mlp = torch.zeros((depth, len(patch_nums)-1))
                    self.cache_similarity_attn = torch.zeros((depth, len(patch_nums)-1))
        else:
            self.cache_mlp = None
            self.cache_attn = None
            self.cache_similarity_mlp = None
            self.cache_similarity_attn = None
    
    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        """Same as original VAR"""
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + h
        else: h = h_or_h_and_residual
        return self.head(self.head_nm(h, cond_BD)).float()
    
    def forward(self, label_B: torch.LongTensor, x_BLCv_wo_first_l: torch.Tensor) -> torch.Tensor:
        """Same as original VAR forward for training"""
        with torch.cuda.amp.autocast(enabled=False):
            B, L, Cv = x_BLCv_wo_first_l.shape
            H = W = round(L ** 0.5)
            cond_BD = self.shared_ada_lin(self.class_emb(label_B))
            
            # Handle progressive training
            if self.prog_si >= 0:
                cond_BD = cond_BD[:B]  # fixed bug on Mar 8
                x_BLC = self.word_embed(x_BLCv_wo_first_l)
                x_BLC += self.lvl_embed(self.lvl_1L[:, :L].expand(B, -1)) + self.pos_1LC[:, :L]
                attn_bias = self.attn_bias_for_masking[:, :, :L, :L]
                
                for si, block in enumerate(self.blocks):
                    # Set current stage for each block
                    if hasattr(block, 'set_current_stage'):
                        block.set_current_stage(self.prog_si)
                    x_BLC = block(x=x_BLC, cond_BD=cond_BD, attn_bias=attn_bias, 
                                cache_mlp=self.cache_mlp, cache_attn=self.cache_attn,
                                cache_similarity_mlp=self.cache_similarity_mlp, cache_similarity_attn=self.cache_similarity_attn)
            else:
                # Full sequence training
                sos = cond_BD = self.class_emb(label_B)
                lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
                x_BLC = self.word_embed(x_BLCv_wo_first_l) + lvl_pos[:, self.first_l:]
                x_BLC = torch.cat((sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1) + lvl_pos[:, :self.first_l], x_BLC), dim=1)
                attn_bias = self.attn_bias_for_masking
                
                for si, block in enumerate(self.blocks):
                    x_BLC = block(x=x_BLC, cond_BD=cond_BD, attn_bias=attn_bias,
                                cache_mlp=self.cache_mlp, cache_attn=self.cache_attn,
                                cache_similarity_mlp=self.cache_similarity_mlp, cache_similarity_attn=self.cache_similarity_attn)
        
        return self.get_logits(x_BLC, cond_BD)
    
    def autoregressive_infer_cfg(
        self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
        more_smooth=False,
    ) -> torch.Tensor:
        """
        Enhanced autoregressive inference with fine-grained stage+layer cache control
        """
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        
        if label_B is None:
            label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device)
        
        sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
        
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        
        cur_L = 0
        for b in self.blocks: b.attn.kv_caching(True)
        
        # Track cache usage for reporting
        cache_usage_log = []
        
        for si, pn in enumerate(self.patch_nums):  # si: i-th segment
            # Set current stage for all blocks
            for block in self.blocks:
                if hasattr(block, 'set_current_stage'):
                    block.set_current_stage(si)
            
            L = pn * pn
            
            # Log cache decisions for this stage
            stage_cache_info = {
                'stage': si,
                'patch_size': pn,
                'sequence_length': L,
                'cached_attn_layers': [],
                'cached_mlp_layers': []
            }
            
            # Check which layers will use cache
            for layer_idx in range(len(self.blocks)):
                if self.layer_cache_config.should_cache_layer_attn(si, layer_idx, L):
                    stage_cache_info['cached_attn_layers'].append(layer_idx)
                if self.layer_cache_config.should_cache_layer_mlp(si, layer_idx, L):
                    stage_cache_info['cached_mlp_layers'].append(layer_idx)
            
            cache_usage_log.append(stage_cache_info)
            
            if si == 0:
                x = next_token_map
            else:
                # Sample next tokens
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16, cache_enabled=True):
                    for i in range(L):
                        cur_L += 1
                        cond_BD_or_gss = self.shared_ada_lin(cond_BD)
                        x = x + lvl_pos[:, cur_L-1:cur_L]
                        
                        for block in self.blocks:
                            x = block(x=x, cond_BD=cond_BD_or_gss, attn_bias=None,
                                    cache_mlp=self.cache_mlp, cache_attn=self.cache_attn,
                                    cache_similarity_mlp=self.cache_similarity_mlp, cache_similarity_attn=self.cache_similarity_attn)
                        
                        logits_BlV = self.get_logits(x, cond_BD)
                        logits_BlV = logits_BlV[:B]  # remove unconditional logits
                        
                        if cfg > 1:
                            logits_BlV -= cfg * (logits_BlV - self.get_logits(x[B:], cond_BD[B:]))
                        
                        idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, -1:]
                        if not more_smooth:
                            idx_Bl = idx_Bl.clamp_max(self.V - 1)
                            h_BChw = self.vae_proxy[0].quantize.embedding(idx_Bl).transpose_(1, 2)
                        else:
                            h_BChw = gumbel_softmax_with_rng(logits_BlV[:, -1:].div(0.27), tau=1, dim=-1, rng=rng) @ self.vae_proxy[0].quantize.embedding.weight.unsqueeze(0)
                            h_BChw = h_BChw.transpose_(1, 2)
                        
                        x = torch.cat((x, self.word_embed(h_BChw)), dim=1)
                        next_token_map = torch.cat((next_token_map, self.word_embed(h_BChw.expand(2 * B, -1, -1))), dim=1)
                
                x = next_token_map
        
        for b in self.blocks: b.attn.kv_caching(False)
        
        # Print cache usage summary
        total_layers_processed = 0
        total_layers_cached = 0
        
        for stage_info in cache_usage_log:
            stage_cached = len(stage_info['cached_attn_layers']) + len(stage_info['cached_mlp_layers'])
            stage_total = len(self.blocks) * 2  # attn + mlp
            total_layers_processed += stage_total
            total_layers_cached += stage_cached
            
            if stage_cached > 0:
                print(f"Stage {stage_info['stage']} (L={stage_info['sequence_length']}): "
                      f"{stage_cached}/{stage_total} layers cached "
                      f"(attn: {stage_info['cached_attn_layers']}, mlp: {stage_info['cached_mlp_layers']})")
        
        cache_efficiency = (total_layers_cached / total_layers_processed) * 100 if total_layers_processed > 0 else 0
        print(f"Overall cache efficiency: {total_layers_cached}/{total_layers_processed} layers ({cache_efficiency:.1f}%)")
        
        # Decode with VAE
        h_BChw = self.vae_proxy[0].fh_to_fhw(next_token_map[:B, self.first_l:], self.patch_nums)
        return self.vae_proxy[0].decode(h_BChw)


# Factory function for easy model creation
def create_var_enhanced_compatible(
    vae_local: VQVAE,
    layer_cache_config: LayerCacheConfig = None,
    **kwargs
) -> VAREnhancedCompatible:
    """
    Create enhanced VAR model with fine-grained cache control
    """
    return VAREnhancedCompatible(
        vae_local=vae_local,
        layer_cache_config=layer_cache_config,
        **kwargs
    )