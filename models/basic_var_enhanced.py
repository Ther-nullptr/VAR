import math
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.helpers import DropPath, drop_path


# this file provides enhanced VAR transformer blocks with flexible multi-stage caching
__all__ = ['FFNEnhanced', 'AdaLNSelfAttnEnhanced', 'AdaLNBeforeHead', 'CacheConfig']


# automatically import fused operators
dropout_add_layer_norm = fused_mlp_func = memory_efficient_attention = flash_attn_func = None
try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
    from flash_attn.ops.fused_dense import fused_mlp_func
except ImportError: pass
# automatically import faster attention implementations
try: from xformers.ops import memory_efficient_attention
except ImportError: pass
try: from flash_attn import flash_attn_func              # qkv: BLHc, ret: BLHcq
except ImportError: pass
try: from torch.nn.functional import scaled_dot_product_attention as slow_attn    # q, k, v: BHLc
except ImportError:
    def slow_attn(query, key, value, scale: float, attn_mask=None, dropout_p=0.0):
        attn = query.mul(scale) @ key.transpose(-2, -1) # BHLc @ BHcL => BHLL
        if attn_mask is not None: attn.add_(attn_mask)
        return (F.dropout(attn.softmax(dim=-1), p=dropout_p, inplace=True) if dropout_p > 0 else attn.softmax(dim=-1)) @ value


# Enhanced length to iteration mapping
length2iteration = {
    1: 0, 4: 1, 9: 2, 16: 3, 25: 4, 36: 5, 49: 6, 64: 7, 81: 8, 100: 9, 121: 10, 144: 11, 169: 12, 196: 13, 225: 14, 256: 15,
}  

next_patch_size = {
    1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15, 15: 16, 16: 16
}


class CacheConfig:
    """
    Configuration class for enhanced multi-stage caching
    """
    def __init__(
        self,
        skip_stages: List[int] = None,  # List of stages to skip (e.g., [169, 256])
        cache_stages: List[int] = None,  # List of stages to cache (e.g., [100, 169])
        enable_attn_cache: bool = True,  # Enable attention layer caching
        enable_mlp_cache: bool = True,   # Enable MLP layer caching
        threshold: float = 0.7,          # Similarity threshold for cache reuse
        max_skip_stages: int = 9,        # Maximum number of stages to skip
        adaptive_threshold: bool = False, # Use adaptive thresholding
        interpolation_mode: str = 'bilinear'  # Interpolation mode for feature upsampling
    ):
        self.skip_stages = skip_stages or []
        self.cache_stages = cache_stages or []
        self.enable_attn_cache = enable_attn_cache
        self.enable_mlp_cache = enable_mlp_cache
        self.threshold = threshold
        self.max_skip_stages = min(max_skip_stages, 9)  # Cap at 9 stages
        self.adaptive_threshold = adaptive_threshold
        self.interpolation_mode = interpolation_mode
        
        # Validate configuration
        self.validate()
    
    def validate(self):
        """Validate cache configuration"""
        if len(self.skip_stages) > self.max_skip_stages:
            raise ValueError(f"Cannot skip more than {self.max_skip_stages} stages")
        
        if len(self.skip_stages) < 1 and (self.skip_stages or self.cache_stages):
            if not self.skip_stages and not self.cache_stages:
                pass  # No caching enabled
            elif len(self.skip_stages) == 0:
                raise ValueError("Must skip at least 1 stage when caching is enabled")
        
        # Ensure skip stages are valid sequence lengths
        valid_lengths = set(length2iteration.keys())
        for stage in self.skip_stages:
            if stage not in valid_lengths:
                raise ValueError(f"Invalid skip stage {stage}. Must be one of {sorted(valid_lengths)}")
        
        for stage in self.cache_stages:
            if stage not in valid_lengths:
                raise ValueError(f"Invalid cache stage {stage}. Must be one of {sorted(valid_lengths)}")
    
    def should_skip_attn(self, L: int) -> bool:
        """Check if attention computation should be skipped for given sequence length"""
        return self.enable_attn_cache and L in self.skip_stages
    
    def should_cache_attn(self, L: int) -> bool:
        """Check if attention result should be cached for given sequence length"""
        return self.enable_attn_cache and L in self.cache_stages
    
    def should_skip_mlp(self, L: int) -> bool:
        """Check if MLP computation should be skipped for given sequence length"""
        return self.enable_mlp_cache and L in self.skip_stages
    
    def should_cache_mlp(self, L: int) -> bool:
        """Check if MLP result should be cached for given sequence length"""
        return self.enable_mlp_cache and L in self.cache_stages


def feature_interpolate(x, target_size=None, mode='bilinear', align_corners=True):
    """
    Enhanced feature interpolation with flexible target size
    """
    if x is None:
        return None
    
    # input: [B, L1, C]
    # output: [B, L2, C] where L2 is determined by target_size or next_patch_size
    hw = int(math.sqrt(x.shape[1]))
    
    if target_size is None:
        target_hw = next_patch_size.get(hw, hw + 1)
    else:
        target_hw = int(math.sqrt(target_size))
    
    if target_hw == hw:
        return x  # No interpolation needed
    
    result = x.view(x.shape[0], hw, hw, -1)
    result = torch.nn.functional.interpolate(
        result.permute(0, 3, 1, 2), 
        size=(target_hw, target_hw), 
        mode=mode, 
        align_corners=align_corners
    ).permute(0, 2, 3, 1)
    result = result.view(result.shape[0], -1, result.shape[-1])
    return result


class FFNEnhanced(nn.Module):
    """Enhanced FFN with flexible multi-stage caching"""
    
    def __init__(
        self, 
        block_idx,
        in_features, 
        hidden_features=None, 
        out_features=None, 
        drop=0., 
        fused_if_available=True,
        cache_config: CacheConfig = None
    ):
        super().__init__()
        self.fused_mlp_func = fused_mlp_func if fused_if_available else None
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop, inplace=True) if drop > 0 else nn.Identity()
        
        self.block_idx = block_idx
        self.cache_config = cache_config or CacheConfig()
        
        # Caching state
        self.temp_cache = None  # For calibration
        self.calibration_iter = 0
        
        # Adaptive threshold tracking
        self.similarity_history = []
        self.adaptive_threshold = self.cache_config.threshold
    
    def _compute_similarity(self, cache, result):
        """Compute cosine similarity between cached and computed features"""
        cache_flat = cache.view(-1, cache.shape[-1])
        result_flat = result.view(-1, result.shape[-1])
        return torch.nn.functional.cosine_similarity(cache_flat, result_flat, dim=1).mean().item()
    
    def _update_adaptive_threshold(self, similarity):
        """Update adaptive threshold based on similarity history"""
        if not self.cache_config.adaptive_threshold:
            return
        
        self.similarity_history.append(similarity)
        if len(self.similarity_history) > 100:  # Keep last 100 similarities
            self.similarity_history.pop(0)
        
        # Adaptive threshold is the 80th percentile of recent similarities
        if len(self.similarity_history) >= 10:
            self.adaptive_threshold = torch.tensor(self.similarity_history).quantile(0.8).item()
    
    def get_effective_threshold(self):
        """Get the effective threshold (adaptive or fixed)"""
        return self.adaptive_threshold if self.cache_config.adaptive_threshold else self.cache_config.threshold
    
    def forward(self, x, cache_similarity_mlp, cache_mlp, calibration=False):
        L = x.shape[1]
        
        if self.fused_mlp_func is not None:
            return self.drop(self.fused_mlp_func(
                x=x, weight1=self.fc1.weight, weight2=self.fc2.weight, 
                bias1=self.fc1.bias, bias2=self.fc2.bias,
                activation='gelu_approx', save_pre_act=self.training, return_residual=False, 
                checkpoint_lvl=0, heuristic=0, process_group=None,
            ))
        
        if calibration:
            # Calibration mode: compute features and track similarity
            result = self.drop(self.fc2(self.act(self.fc1(x))))
            
            if L == 1 and self.temp_cache is not None:
                self.temp_cache = None
                self.calibration_iter += 1
            
            if self.temp_cache is not None:
                # Interpolate cache to current size
                cache = feature_interpolate(
                    self.temp_cache, 
                    target_size=L,
                    mode=self.cache_config.interpolation_mode
                )
                
                # Compute and track similarity
                cos_sim = self._compute_similarity(cache, result)
                
                # Update similarity tracking
                if length2iteration[L] - 1 >= 0:
                    prev_sim = cache_similarity_mlp[self.block_idx][length2iteration[L] - 1]
                    cache_similarity_mlp[self.block_idx][length2iteration[L] - 1] = (
                        cos_sim + prev_sim * self.calibration_iter
                    ) / (self.calibration_iter + 1)
                
                self._update_adaptive_threshold(cos_sim)
            
            self.temp_cache = result.clone()
            return result
        
        # Inference mode with caching
        effective_threshold = self.get_effective_threshold()
        
        # Check if we should skip computation and use cache
        if self.cache_config.should_skip_mlp(L) and cache_mlp[self.block_idx] is not None:
            if (L in length2iteration and 
                length2iteration[L] - 1 >= 0 and 
                cache_similarity_mlp[self.block_idx][length2iteration[L] - 1] > effective_threshold):
                
                result = feature_interpolate(
                    cache_mlp[self.block_idx],
                    target_size=L,
                    mode=self.cache_config.interpolation_mode
                )
            else:
                result = self.drop(self.fc2(self.act(self.fc1(x))))
        else:
            result = self.drop(self.fc2(self.act(self.fc1(x))))
        
        # Clear previous cache
        cache_mlp[self.block_idx] = None
        
        # Check if we should cache the result
        if self.cache_config.should_cache_mlp(L):
            if (L in length2iteration and 
                length2iteration[L] < len(cache_similarity_mlp[self.block_idx]) and
                cache_similarity_mlp[self.block_idx][length2iteration[L]] > effective_threshold):
                
                cache_mlp[self.block_idx] = result.clone()
        
        return result
    
    def extra_repr(self) -> str:
        return f'fused_mlp_func={self.fused_mlp_func is not None}, cache_enabled={bool(self.cache_config.cache_stages)}'


class SelfAttentionEnhanced(nn.Module):
    """Enhanced Self-Attention with flexible multi-stage caching"""
    
    def __init__(
        self, block_idx, embed_dim=768, num_heads=12,
        attn_drop=0., proj_drop=0., attn_l2_norm=False, flash_if_available=True,
        cache_config: CacheConfig = None
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.block_idx, self.num_heads, self.head_dim = block_idx, num_heads, embed_dim // num_heads
        self.attn_l2_norm = attn_l2_norm
        self.cache_config = cache_config or CacheConfig()
        
        if self.attn_l2_norm:
            self.scale = 1
            self.scale_mul_1H11 = nn.Parameter(torch.full(size=(1, self.num_heads, 1, 1), fill_value=4.0).log(), requires_grad=True)
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 0.25 / math.sqrt(self.head_dim)
        
        self.mat_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True) if proj_drop > 0 else nn.Identity()
        self.attn_drop: float = attn_drop
        self.using_flash = flash_if_available and flash_attn_func is not None
        self.using_xform = flash_if_available and memory_efficient_attention is not None
        
        self.softmax = torch.nn.Softmax(dim=-1)
        
        # KV caching for autoregressive generation
        self.caching, self.cached_k, self.cached_v = False, None, None
        
        # Feature caching state
        self.temp_cache = None
        self.calibration_iter = 0
        
        # Adaptive threshold tracking
        self.similarity_history = []
        self.adaptive_threshold = self.cache_config.threshold
    
    def kv_caching(self, enable: bool): 
        self.caching, self.cached_k, self.cached_v = enable, None, None
    
    def _compute_similarity(self, cache, result):
        """Compute cosine similarity between cached and computed features"""
        cache_flat = cache.view(-1, cache.shape[-1])
        result_flat = result.view(-1, result.shape[-1])
        return torch.nn.functional.cosine_similarity(cache_flat, result_flat, dim=1).mean().item()
    
    def _update_adaptive_threshold(self, similarity):
        """Update adaptive threshold based on similarity history"""
        if not self.cache_config.adaptive_threshold:
            return
        
        self.similarity_history.append(similarity)
        if len(self.similarity_history) > 100:
            self.similarity_history.pop(0)
        
        if len(self.similarity_history) >= 10:
            self.adaptive_threshold = torch.tensor(self.similarity_history).quantile(0.8).item()
    
    def get_effective_threshold(self):
        """Get the effective threshold (adaptive or fixed)"""
        return self.adaptive_threshold if self.cache_config.adaptive_threshold else self.cache_config.threshold
    
    def forward_inner(self, x, attn_bias):
        """Core attention computation"""
        B, L, C = x.shape
        qkv = self.mat_qkv(x).view(B, L, 3, self.num_heads, self.head_dim)
        main_type = qkv.dtype
        
        using_flash = self.using_flash and attn_bias is None and qkv.dtype != torch.float32
        if using_flash or self.using_xform: 
            q, k, v = qkv.unbind(dim=2)
            dim_cat = 1
        else: 
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
            dim_cat = 2
        
        if self.attn_l2_norm:
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
            if using_flash or self.using_xform: 
                scale_mul = scale_mul.transpose(1, 2)
            q = F.normalize(q, dim=-1).mul(scale_mul)
            k = F.normalize(k, dim=-1)
        
        if self.caching:
            if self.cached_k is None: 
                self.cached_k = k
                self.cached_v = v
            else:
                k = self.cached_k = torch.cat((self.cached_k, k), dim=dim_cat)
                v = self.cached_v = torch.cat((self.cached_v, v), dim=dim_cat)
        
        dropout_p = self.attn_drop if self.training else 0.0
        if using_flash:
            oup = flash_attn_func(q.to(dtype=main_type), k.to(dtype=main_type), v.to(dtype=main_type), 
                                dropout_p=dropout_p, softmax_scale=self.scale).view(B, L, C)
        elif self.using_xform:
            oup = memory_efficient_attention(q.to(dtype=main_type), k.to(dtype=main_type), v.to(dtype=main_type), 
                                           attn_bias=None if attn_bias is None else attn_bias.to(dtype=main_type).expand(B, self.num_heads, -1, -1), 
                                           p=dropout_p, scale=self.scale).view(B, L, C)
        else:
            attn = q.mul(self.scale) @ k.transpose(-2, -1)
            if attn_bias is not None: 
                attn.add_(attn_bias)
            oup = self.softmax(attn) @ v
            oup = oup.transpose(1, 2).reshape(B, L, C)
        
        result = self.proj_drop(self.proj(oup))
        return result
    
    def forward(self, x, attn_bias, cache_similarity_attn, cache_attn, calibration=False):
        L = x.shape[1]
        
        if calibration:
            # Calibration mode
            result = self.forward_inner(x, attn_bias)
            
            if L == 1 and self.temp_cache is not None:
                self.temp_cache = None
                self.calibration_iter += 1
            
            if self.temp_cache is not None:
                cache = feature_interpolate(
                    self.temp_cache, 
                    target_size=L,
                    mode=self.cache_config.interpolation_mode
                )
                cos_sim = self._compute_similarity(cache, result)
                
                if length2iteration[L] - 1 >= 0:
                    prev_sim = cache_similarity_attn[self.block_idx][length2iteration[L] - 1]
                    cache_similarity_attn[self.block_idx][length2iteration[L] - 1] = (
                        cos_sim + prev_sim * self.calibration_iter
                    ) / (self.calibration_iter + 1)
                
                self._update_adaptive_threshold(cos_sim)
            
            self.temp_cache = result.clone()
            return result
        
        # Inference mode with caching
        effective_threshold = self.get_effective_threshold()
        
        # Check if we should skip computation and use cache
        if self.cache_config.should_skip_attn(L) and cache_attn[self.block_idx] is not None:
            if (L in length2iteration and 
                length2iteration[L] - 1 >= 0 and 
                cache_similarity_attn[self.block_idx][length2iteration[L] - 1] > effective_threshold):
                
                result = feature_interpolate(
                    cache_attn[self.block_idx],
                    target_size=L,
                    mode=self.cache_config.interpolation_mode
                )
            else:
                result = self.forward_inner(x, attn_bias)
        else:
            result = self.forward_inner(x, attn_bias)
        
        # Clear previous cache
        cache_attn[self.block_idx] = None
        
        # Check if we should cache the result
        if self.cache_config.should_cache_attn(L):
            if (L in length2iteration and 
                length2iteration[L] < len(cache_similarity_attn[self.block_idx]) and
                cache_similarity_attn[self.block_idx][length2iteration[L]] > effective_threshold):
                
                cache_attn[self.block_idx] = result.clone()
        
        return result
    
    def extra_repr(self) -> str:
        return f'using_flash={self.using_flash}, using_xform={self.using_xform}, attn_l2_norm={self.attn_l2_norm}, cache_enabled={bool(self.cache_config.cache_stages)}'


class AdaLNSelfAttnEnhanced(nn.Module):
    """Enhanced AdaLN Self-Attention with flexible multi-stage caching"""
    
    def __init__(
        self, block_idx, last_drop_p, embed_dim, cond_dim, shared_aln: bool, norm_layer,
        num_heads, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0., attn_l2_norm=False,
        flash_if_available=False, fused_if_available=True,
        cache_config: CacheConfig = None
    ):
        super(AdaLNSelfAttnEnhanced, self).__init__()
        self.block_idx, self.last_drop_p, self.C = block_idx, last_drop_p, embed_dim
        self.C, self.D = embed_dim, cond_dim
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.cache_config = cache_config or CacheConfig()
        
        self.attn = SelfAttentionEnhanced(
            block_idx=block_idx, embed_dim=embed_dim, num_heads=num_heads, 
            attn_drop=attn_drop, proj_drop=drop, attn_l2_norm=attn_l2_norm, 
            flash_if_available=flash_if_available, cache_config=cache_config
        )
        self.ffn = FFNEnhanced(
            block_idx=block_idx, in_features=embed_dim, 
            hidden_features=round(embed_dim * mlp_ratio), drop=drop, 
            fused_if_available=fused_if_available, cache_config=cache_config
        )
        
        self.ln_wo_grad = norm_layer(embed_dim, elementwise_affine=False)
        self.shared_aln = shared_aln
        if self.shared_aln:
            self.ada_gss = nn.Parameter(torch.randn(1, 1, 6, embed_dim) / embed_dim**0.5)
        else:
            lin = nn.Linear(cond_dim, 6*embed_dim)
            self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), lin)
        
        self.fused_add_norm_fn = None
    
    def forward(self, x, cond_BD, attn_bias, cache_mlp, cache_attn, 
                cache_similarity_mlp, cache_similarity_attn, calibration=False):
        if self.shared_aln:
            gamma1, gamma2, scale1, scale2, shift1, shift2 = (self.ada_gss + cond_BD).unbind(2)
        else:
            gamma1, gamma2, scale1, scale2, shift1, shift2 = self.ada_lin(cond_BD).view(-1, 1, 6, self.C).unbind(2)
        
        # Attention with enhanced caching
        attn_input = self.ln_wo_grad(x) * (scale1.add(1)) + (shift1)
        attn_out = self.attn(
            attn_input, attn_bias=attn_bias, 
            cache_attn=cache_attn, cache_similarity_attn=cache_similarity_attn,
            calibration=calibration
        )
        x = x + self.drop_path(attn_out.mul_(gamma1))
        
        # FFN with enhanced caching
        ffn_input = self.ln_wo_grad(x) * (scale2.add(1)) + (shift2)
        ffn_out = self.ffn(
            ffn_input, cache_mlp=cache_mlp, 
            cache_similarity_mlp=cache_similarity_mlp,
            calibration=calibration
        )
        x = x + self.drop_path(ffn_out.mul(gamma2))
        
        return x
    
    def extra_repr(self) -> str:
        return f'shared_aln={self.shared_aln}, skip_stages={self.cache_config.skip_stages}, cache_stages={self.cache_config.cache_stages}'


class AdaLNBeforeHead(nn.Module):
    """AdaLN normalization before head (unchanged from original)"""
    def __init__(self, C, D, norm_layer):
        super().__init__()
        self.C, self.D = C, D
        self.ln_wo_grad = norm_layer(C, elementwise_affine=False)
        self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), nn.Linear(D, 2*C))
    
    def forward(self, x_BLC: torch.Tensor, cond_BD: torch.Tensor):
        scale, shift = self.ada_lin(cond_BD).view(-1, 1, 2, self.C).unbind(2)
        return self.ln_wo_grad(x_BLC).mul(scale.add(1)).add_(shift)
