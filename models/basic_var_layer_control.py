"""
Enhanced VAR transformer blocks with fine-grained stage+layer caching control
Extends the existing cache-interpolation strategy with precise layer-level control
"""

import math
from typing import List, Dict, Optional, Tuple, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.helpers import DropPath, drop_path


# automatically import fused operators
dropout_add_layer_norm = fused_mlp_func = memory_efficient_attention = flash_attn_func = None
try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
    from flash_attn.ops.fused_dense import fused_mlp_func
except ImportError: pass
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


# Enhanced length to iteration mapping for fine-grained control
length2iteration = {
    1:0, 4:1, 9:2, 16:3, 25:4, 36:5, 64:6, 100:7, 169:8, 256:9,
}
next_patch_size = {
    1:2, 2:3, 3:4, 4:5, 5:6, 6:8, 8:10, 10:13, 13:16
}


def feature_interpolate(x, target_length: int, mode='bilinear', align_corners=True):
    """
    Interpolate features from cached sequence length to target sequence length
    
    Args:
        x: Input tensor [B, L1, C]
        target_length: Target sequence length L2
        mode: Interpolation mode
        align_corners: Whether to align corners in interpolation
    
    Returns:
        Interpolated tensor [B, L2, C]
    """
    if x.shape[1] == target_length:
        return x
    
    B, L, C = x.shape
    
    # Convert to 2D spatial representation for better interpolation
    hw_src = int(math.sqrt(L))
    hw_tgt = int(math.sqrt(target_length))
    
    if hw_src * hw_src != L or hw_tgt * hw_tgt != target_length:
        # Fallback to linear interpolation for non-square sequences
        return F.interpolate(x.transpose(1, 2), size=target_length, mode='linear', align_corners=align_corners).transpose(1, 2)
    
    # Reshape to spatial format and interpolate
    x_spatial = x.view(B, hw_src, hw_src, C).permute(0, 3, 1, 2)  # B, C, H, W
    x_interp = F.interpolate(x_spatial, size=(hw_tgt, hw_tgt), mode=mode, align_corners=align_corners)
    x_output = x_interp.permute(0, 2, 3, 1).view(B, target_length, C)  # B, L2, C
    
    return x_output


class LayerCacheConfig:
    """
    Enhanced configuration class for fine-grained stage+layer caching control
    """
    def __init__(
        self,
        # Original stage-based controls
        skip_stages: List[int] = None,  # List of stages to skip (e.g., [169, 256])
        cache_stages: List[int] = None,  # List of stages to cache (e.g., [100, 169])
        enable_attn_cache: bool = True,  # Enable attention layer caching
        enable_mlp_cache: bool = True,   # Enable MLP layer caching
        threshold: float = 0.7,          # Similarity threshold for cache reuse
        max_skip_stages: int = 9,        # Maximum number of stages to skip
        adaptive_threshold: bool = False, # Use adaptive thresholding
        interpolation_mode: str = 'bilinear',  # Interpolation mode for feature upsampling
        
        # New fine-grained layer controls
        stage_layer_cache_control: Dict[int, Dict[int, str]] = None,  # {stage_idx: {layer_idx: 'cache'/'skip'/'normal'}}
        layer_cache_priority: Dict[int, List[int]] = None,  # {stage_idx: [layer_indices_priority_order]}
        cache_attn_layers: Dict[int, Set[int]] = None,      # {stage_idx: {layer_indices_for_attn_cache}}
        cache_mlp_layers: Dict[int, Set[int]] = None,       # {stage_idx: {layer_indices_for_mlp_cache}}
        layer_interpolation_modes: Dict[Tuple[int, int], str] = None,  # {(stage_idx, layer_idx): mode}
        layer_blend_ratios: Dict[Tuple[int, int], float] = None,       # {(stage_idx, layer_idx): blend_ratio}
    ):
        # Original configuration
        self.skip_stages = skip_stages or []
        self.cache_stages = cache_stages or []
        self.enable_attn_cache = enable_attn_cache
        self.enable_mlp_cache = enable_mlp_cache
        self.threshold = threshold
        self.max_skip_stages = min(max_skip_stages, 9)
        self.adaptive_threshold = adaptive_threshold
        self.interpolation_mode = interpolation_mode
        
        # Fine-grained layer controls
        self.stage_layer_cache_control = stage_layer_cache_control or {}
        self.layer_cache_priority = layer_cache_priority or {}
        self.cache_attn_layers = cache_attn_layers or {}
        self.cache_mlp_layers = cache_mlp_layers or {}
        self.layer_interpolation_modes = layer_interpolation_modes or {}
        self.layer_blend_ratios = layer_blend_ratios or {}
        
        # Validate configuration
        self.validate()
    
    def validate(self):
        """Validate cache configuration"""
        if len(self.skip_stages) > self.max_skip_stages:
            raise ValueError(f"Cannot skip more than {self.max_skip_stages} stages")
        
        # Validate stage lengths
        valid_lengths = set(length2iteration.keys())
        for stage in self.skip_stages + self.cache_stages:
            if stage not in valid_lengths:
                raise ValueError(f"Invalid stage {stage}. Must be one of {sorted(valid_lengths)}")
        
        # Validate layer blend ratios
        for (stage_idx, layer_idx), ratio in self.layer_blend_ratios.items():
            if not 0.0 <= ratio <= 1.0:
                raise ValueError(f"Blend ratio for stage {stage_idx} layer {layer_idx} must be in [0, 1]")
    
    def should_cache_layer_attn(self, stage_idx: int, layer_idx: int, L: int) -> bool:
        """Check if attention should use cache for given stage and layer"""
        # Fine-grained control has priority
        if stage_idx in self.stage_layer_cache_control:
            if layer_idx in self.stage_layer_cache_control[stage_idx]:
                control = self.stage_layer_cache_control[stage_idx][layer_idx]
                if control == 'cache':
                    return self.enable_attn_cache
                elif control == 'skip' or control == 'normal':
                    return False
        
        # Check specific attention layer cache control
        if stage_idx in self.cache_attn_layers and layer_idx in self.cache_attn_layers[stage_idx]:
            return self.enable_attn_cache
        
        # Fall back to original stage-based logic
        return self.enable_attn_cache and L in self.skip_stages
    
    def should_cache_layer_mlp(self, stage_idx: int, layer_idx: int, L: int) -> bool:
        """Check if MLP should use cache for given stage and layer"""
        # Fine-grained control has priority
        if stage_idx in self.stage_layer_cache_control:
            if layer_idx in self.stage_layer_cache_control[stage_idx]:
                control = self.stage_layer_cache_control[stage_idx][layer_idx]
                if control == 'cache':
                    return self.enable_mlp_cache
                elif control == 'skip' or control == 'normal':
                    return False
        
        # Check specific MLP layer cache control
        if stage_idx in self.cache_mlp_layers and layer_idx in self.cache_mlp_layers[stage_idx]:
            return self.enable_mlp_cache
        
        # Fall back to original stage-based logic
        return self.enable_mlp_cache and L in self.skip_stages
    
    def should_cache_attn(self, L: int) -> bool:
        """Check if attention computation should be cached for given sequence length (original logic)"""
        return self.enable_attn_cache and L in self.cache_stages
    
    def should_cache_mlp(self, L: int) -> bool:
        """Check if MLP computation should be cached for given sequence length (original logic)"""
        return self.enable_mlp_cache and L in self.cache_stages
    
    def get_layer_interpolation_mode(self, stage_idx: int, layer_idx: int) -> str:
        """Get interpolation mode for specific stage and layer"""
        return self.layer_interpolation_modes.get((stage_idx, layer_idx), self.interpolation_mode)
    
    def get_layer_blend_ratio(self, stage_idx: int, layer_idx: int) -> float:
        """Get blend ratio for specific stage and layer (1.0 = pure cache, 0.0 = pure computation)"""
        return self.layer_blend_ratios.get((stage_idx, layer_idx), 1.0)  # Default to pure cache
    
    def get_cache_summary(self, num_stages: int, num_layers: int) -> str:
        """Get a summary string of the cache configuration"""
        summary_parts = []
        
        # Stage-based summary
        if self.skip_stages or self.cache_stages:
            summary_parts.append(f"Stage-based: skip {self.skip_stages}, cache {self.cache_stages}")
        
        # Layer-based summary
        layer_summaries = []
        for stage_idx in range(num_stages):
            stage_summary = []
            
            # Check stage-layer cache control
            if stage_idx in self.stage_layer_cache_control:
                for layer_idx, control in self.stage_layer_cache_control[stage_idx].items():
                    stage_summary.append(f"L{layer_idx}:{control}")
            
            # Check specific attention layers
            if stage_idx in self.cache_attn_layers:
                attn_layers = sorted(self.cache_attn_layers[stage_idx])
                stage_summary.append(f"attn:{attn_layers}")
            
            # Check specific MLP layers
            if stage_idx in self.cache_mlp_layers:
                mlp_layers = sorted(self.cache_mlp_layers[stage_idx])
                stage_summary.append(f"mlp:{mlp_layers}")
            
            if stage_summary:
                layer_summaries.append(f"S{stage_idx}[{','.join(stage_summary)}]")
        
        if layer_summaries:
            summary_parts.append(f"Layer-based: {' '.join(layer_summaries)}")
        
        return "; ".join(summary_parts) if summary_parts else "No caching enabled"


class SelfAttentionLayerControl(nn.Module):
    """
    Enhanced self-attention with fine-grained stage+layer cache control
    """
    def __init__(
        self, embed_dim: int, num_heads: int, bias: bool = True,
        proj_drop: float = 0, attn_drop: float = 0, norm_layer: nn.Module = nn.LayerNorm,
        attn_l2_norm: bool = False, flash_if_available: bool = True,
        cache_config: LayerCacheConfig = None,
        block_idx: int = 0,  # Layer index within the transformer
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.block_idx = block_idx
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.attn_l2_norm = attn_l2_norm
        self.cache_config = cache_config or LayerCacheConfig()
        
        self.mat_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True) if proj_drop > 0 else nn.Identity()
        self.attn_drop: float = attn_drop
        self.using_flash = flash_if_available and flash_attn_func is not None
        self.using_xform = flash_if_available and memory_efficient_attention is not None
        
        # KV caching for autoregressive generation
        self.caching, self.cached_k, self.cached_v = False, None, None
        
        # Feature caching for interpolation
        self.feature_cache = {}  # {stage_idx: cached_features}
        self.cache_metadata = {}  # {stage_idx: metadata}
        
        # Current stage tracking
        self.current_stage = 0
    
    def kv_caching(self, enable: bool): 
        self.caching, self.cached_k, self.cached_v = enable, None, None
    
    def set_current_stage(self, stage_idx: int):
        """Set current generation stage for cache control"""
        self.current_stage = stage_idx
    
    def _compute_similarity(self, cache, result):
        """Compute cosine similarity between cached and computed features"""
        cache_flat = cache.view(-1, cache.shape[-1])
        result_flat = result.view(-1, result.shape[-1])
        return torch.nn.functional.cosine_similarity(cache_flat, result_flat, dim=1).mean().item()
    
    def forward(self, x: torch.Tensor, attn_bias: Optional[torch.Tensor], 
                cache_similarity_attn=None, cache_similarity_mlp=None, 
                cache_mlp=None, cache_attn=None, calibration=False):
        """
        Forward pass with fine-grained cache control
        """
        B, L, C = x.shape
        
        # Check if we should use cached features for this layer in current stage
        should_use_cache = self.cache_config.should_cache_layer_attn(
            self.current_stage, self.block_idx, L
        )
        
        if should_use_cache and self.current_stage in self.feature_cache:
            # Use cached features with interpolation
            cached_features = self.feature_cache[self.current_stage]
            
            if cached_features.shape[1] != L:
                # Interpolate cached features to current resolution
                interp_mode = self.cache_config.get_layer_interpolation_mode(
                    self.current_stage, self.block_idx
                )
                cached_features = feature_interpolate(cached_features, L, mode=interp_mode)
            
            # Get blend ratio for this specific layer
            blend_ratio = self.cache_config.get_layer_blend_ratio(
                self.current_stage, self.block_idx
            )
            
            if blend_ratio >= 1.0:
                # Pure cache
                return cached_features
            elif blend_ratio > 0.0:
                # Compute current result for blending
                current_result = self.forward_compute(x, attn_bias)
                # Blend cached and current features
                result = blend_ratio * cached_features + (1.0 - blend_ratio) * current_result
                return result
        
        # Normal computation
        result = self.forward_compute(x, attn_bias)
        
        # Cache the result for future stages
        if self.cache_config.should_cache_attn(L):
            self.feature_cache[self.current_stage] = result.detach().clone()
            self.cache_metadata[self.current_stage] = {
                'stage': self.current_stage,
                'sequence_length': L,
                'layer_idx': self.block_idx
            }
        
        return result
    
    def forward_compute(self, x: torch.Tensor, attn_bias: Optional[torch.Tensor]):
        """Actual attention computation"""
        B, L, C = x.shape
        qkv = self.mat_qkv(x).view(B, L, 3, self.num_heads, self.head_dim)
        
        if self.caching:
            # Autoregressive generation with KV caching
            if self.cached_k is None:
                q, k, v = qkv.unbind(dim=2)
                self.cached_k, self.cached_v = k, v
            else:
                q = qkv[:, :, 0]
                k, v = qkv[:, :, 1:].unbind(dim=2)
                self.cached_k = torch.cat([self.cached_k, k], dim=1)
                self.cached_v = torch.cat([self.cached_v, v], dim=1)
                k, v = self.cached_k, self.cached_v
        else:
            q, k, v = qkv.unbind(dim=2)
        
        if self.attn_l2_norm:
            q = F.normalize(q.float(), dim=-1).type_as(q)
            k = F.normalize(k.float(), dim=-1).type_as(k)
        
        # Attention computation
        if self.using_flash:
            x = flash_attn_func(q, k, v, dropout_p=self.attn_drop if self.training else 0.0)
        elif self.using_xform:
            x = memory_efficient_attention(q, k, v, attn_bias=attn_bias, p=self.attn_drop if self.training else 0.0)
        else:
            x = slow_attn(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), 
                         scale=self.scale, attn_mask=attn_bias, dropout_p=self.attn_drop if self.training else 0.0)
            x = x.transpose(1, 2)
        
        x = x.view(B, L, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FFNLayerControl(nn.Module):
    """
    Enhanced FFN with fine-grained stage+layer cache control
    """
    def __init__(
        self, embed_dim: int, mlp_ratio: float = 4., bias: bool = True, 
        drop: float = 0., norm_layer: nn.Module = nn.LayerNorm, 
        fused_if_available: bool = True,
        cache_config: LayerCacheConfig = None,
        block_idx: int = 0,
    ):
        super().__init__()
        self.block_idx = block_idx
        self.embed_dim = embed_dim
        self.hidden_dim = int(embed_dim * mlp_ratio)
        self.cache_config = cache_config or LayerCacheConfig()
        
        # FFN layers
        self.fc1 = nn.Linear(embed_dim, self.hidden_dim, bias=bias)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(self.hidden_dim, embed_dim, bias=bias)
        self.drop = nn.Dropout(drop, inplace=True) if drop > 0 else nn.Identity()
        
        # Feature caching for interpolation
        self.feature_cache = {}  # {stage_idx: cached_features}
        self.cache_metadata = {}  # {stage_idx: metadata}
        
        # Current stage tracking
        self.current_stage = 0
        
        # Try to use fused MLP if available
        self.fused_mlp_func = fused_mlp_func if fused_if_available else None
    
    def set_current_stage(self, stage_idx: int):
        """Set current generation stage for cache control"""
        self.current_stage = stage_idx
    
    def forward(self, x: torch.Tensor, cache_similarity_mlp=None, cache_mlp=None, calibration=False):
        """
        Forward pass with fine-grained cache control
        """
        B, L, C = x.shape
        
        # Check if we should use cached features for this layer in current stage
        should_use_cache = self.cache_config.should_cache_layer_mlp(
            self.current_stage, self.block_idx, L
        )
        
        if should_use_cache and self.current_stage in self.feature_cache:
            # Use cached features with interpolation
            cached_features = self.feature_cache[self.current_stage]
            
            if cached_features.shape[1] != L:
                # Interpolate cached features to current resolution
                interp_mode = self.cache_config.get_layer_interpolation_mode(
                    self.current_stage, self.block_idx
                )
                cached_features = feature_interpolate(cached_features, L, mode=interp_mode)
            
            # Get blend ratio for this specific layer
            blend_ratio = self.cache_config.get_layer_blend_ratio(
                self.current_stage, self.block_idx
            )
            
            if blend_ratio >= 1.0:
                # Pure cache
                return cached_features
            elif blend_ratio > 0.0:
                # Compute current result for blending
                current_result = self.forward_compute(x)
                # Blend cached and current features
                result = blend_ratio * cached_features + (1.0 - blend_ratio) * current_result
                return result
        
        # Normal computation
        result = self.forward_compute(x)
        
        # Cache the result for future stages
        if self.cache_config.should_cache_mlp(L):
            self.feature_cache[self.current_stage] = result.detach().clone()
            self.cache_metadata[self.current_stage] = {
                'stage': self.current_stage,
                'sequence_length': L,
                'layer_idx': self.block_idx
            }
        
        return result
    
    def forward_compute(self, x: torch.Tensor):
        """Actual FFN computation"""
        if self.fused_mlp_func is not None:
            return self.fused_mlp_func(
                x=x, weight1=self.fc1.weight, weight2=self.fc2.weight, 
                bias1=self.fc1.bias, bias2=self.fc2.bias,
                activation="gelu_new", save_pre_act=self.training,
                return_residual=False, checkpoint_lvl=0, 
                heuristic=0, process_group=None,
            )
        else:
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
            return x


class AdaLNSelfAttnLayerControl(nn.Module):
    """
    Complete transformer block with fine-grained stage+layer cache control
    """
    def __init__(
        self, cond_dim: int, shared_aln: bool, block_idx: int,
        embed_dim: int, norm_layer: nn.Module = nn.LayerNorm, 
        num_heads: int = 8, mlp_ratio: float = 4.,
        drop: float = 0., attn_drop: float = 0., drop_path: float = 0.,
        last_drop_p: float = 0., attn_l2_norm: bool = False,
        flash_if_available: bool = True, fused_if_available: bool = True,
        cache_config: LayerCacheConfig = None,
    ):
        super().__init__()
        self.block_idx = block_idx
        self.cache_config = cache_config or LayerCacheConfig()
        
        # Attention and MLP modules with layer control
        self.attn = SelfAttentionLayerControl(
            embed_dim, num_heads, True, 0., attn_drop, norm_layer, attn_l2_norm, 
            flash_if_available, cache_config, block_idx
        )
        self.ffn = FFNLayerControl(
            embed_dim, mlp_ratio, True, drop, norm_layer, fused_if_available,
            cache_config, block_idx
        )
        
        # Adaptive LayerNorm
        self.ln1 = norm_layer(embed_dim)
        self.ln2 = norm_layer(embed_dim)
        
        # Conditional components
        if shared_aln:
            self.ada_gss = nn.Parameter(torch.randn(1, 1, 6, embed_dim) / embed_dim**0.5)
        else:
            lin = nn.Linear(cond_dim, 6 * embed_dim)
            self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), lin)
        
        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # Fused operations
        self.fused_add_norm_fn = None
        if fused_if_available and dropout_add_layer_norm is not None:
            self.fused_add_norm_fn = dropout_add_layer_norm
    
    def set_current_stage(self, stage_idx: int):
        """Set current generation stage for both attention and FFN"""
        self.attn.set_current_stage(stage_idx)
        self.ffn.set_current_stage(stage_idx)
    
    def forward(self, x: torch.Tensor, cond_BD: torch.Tensor, attn_bias: Optional[torch.Tensor],
                cache_similarity_attn=None, cache_similarity_mlp=None, 
                cache_mlp=None, cache_attn=None, calibration=False):
        """
        Forward pass with adaptive layer norm and fine-grained cache control
        """
        # Get conditioning
        if hasattr(self, 'ada_gss'):
            gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.ada_gss.unbind(dim=2)
        else:
            gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.ada_lin(cond_BD).view(-1, 1, 6, self.attn.embed_dim).unbind(dim=2)
        
        # Attention block with residual connection
        if self.fused_add_norm_fn is not None:
            x_norm1 = self.ln1(x)
            attn_out = self.attn(x_norm1, attn_bias, cache_similarity_attn, cache_similarity_mlp, cache_mlp, cache_attn, calibration)
            attn_out = alpha1 * attn_out
            x, _ = self.fused_add_norm_fn(self.drop_path(attn_out), x, self.ln2.weight, self.ln2.bias, eps=self.ln2.eps)
        else:
            x_norm1 = (1 + gamma1) * self.ln1(x) + beta1
            attn_out = self.attn(x_norm1, attn_bias, cache_similarity_attn, cache_similarity_mlp, cache_mlp, cache_attn, calibration)
            x = x + self.drop_path(alpha1 * attn_out)
            
            x_norm2 = (1 + gamma2) * self.ln2(x) + beta2
            ffn_out = self.ffn(x_norm2, cache_similarity_mlp, cache_mlp, calibration)
            x = x + self.drop_path(alpha2 * ffn_out)
            return x
        
        # FFN block
        ffn_out = self.ffn(x, cache_similarity_mlp, cache_mlp, calibration)
        x = x + self.drop_path(alpha2 * ffn_out)
        
        return x


# Re-export original classes for compatibility
from models.basic_var_enhanced import AdaLNBeforeHead

# Export all enhanced classes
__all__ = [
    'LayerCacheConfig', 'SelfAttentionLayerControl', 'FFNLayerControl', 
    'AdaLNSelfAttnLayerControl', 'AdaLNBeforeHead', 'feature_interpolate'
]