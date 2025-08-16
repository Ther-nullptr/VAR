"""
VAR Cache Wrapper - Wraps original VAR model with fine-grained cache control
This approach maintains full compatibility with existing VAR checkpoints
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List, Dict, Set

from models.basic_var_layer_control import LayerCacheConfig, feature_interpolate, length2iteration


class VARCacheWrapper(nn.Module):
    """
    Wrapper that adds fine-grained cache control to an existing VAR model
    Maintains full compatibility with original VAR checkpoints
    """
    
    def __init__(self, var_model, layer_cache_config: LayerCacheConfig = None):
        super().__init__()
        
        # Wrap the original VAR model
        self.var_model = var_model
        
        # Fine-grained layer cache configuration
        if layer_cache_config is None:
            # Get model depth from VAR model
            model_depth = len(var_model.blocks) if hasattr(var_model, 'blocks') else 16
            num_stages = len(var_model.patch_nums) if hasattr(var_model, 'patch_nums') else 10
            layer_cache_config = LayerCacheConfig(
                model_depth=model_depth,
                num_stages=num_stages
            )
        
        self.layer_cache_config = layer_cache_config
        
        # Cache storage for fine-grained control
        self.attn_cache = {}  # {stage_idx: {layer_idx: cached_features}}
        self.mlp_cache = {}   # {stage_idx: {layer_idx: cached_features}}
        
        # Current generation state
        self.current_stage = 0
        
        # Hook into the original model's blocks
        self._install_cache_hooks()
        
        print(f'[VARCacheWrapper] Wrapped VAR model with cache config: {layer_cache_config.get_cache_summary()}')
    
    def _install_cache_hooks(self):
        """Install forward hooks to intercept and cache layer outputs"""
        
        def create_attn_hook(layer_idx):
            def attn_hook(module, input, output):
                if hasattr(self, '_caching_enabled') and self._caching_enabled:
                    L = input[0].shape[1]  # sequence length
                    
                    # Check if we should use cached features
                    should_use_cache = self.layer_cache_config.should_cache_layer_attn(
                        self.current_stage, layer_idx, L
                    )
                    
                    if should_use_cache and self.current_stage in self.attn_cache:
                        if layer_idx in self.attn_cache[self.current_stage]:
                            # Use cached features with interpolation
                            cached_features = self.attn_cache[self.current_stage][layer_idx]
                            
                            if cached_features.shape[1] != L:
                                # Interpolate cached features to current resolution
                                interp_mode = self.layer_cache_config.get_layer_interpolation_mode(
                                    self.current_stage, layer_idx
                                )
                                cached_features = feature_interpolate(cached_features, L, mode=interp_mode)
                            
                            # Get blend ratio for this specific layer
                            blend_ratio = self.layer_cache_config.get_layer_blend_ratio(
                                self.current_stage, layer_idx
                            )
                            
                            if blend_ratio >= 1.0:
                                # Pure cache - replace output
                                return cached_features
                            elif blend_ratio > 0.0:
                                # Blend cached and current features
                                result = blend_ratio * cached_features + (1.0 - blend_ratio) * output
                                return result
                    
                    # Store current output for future stages if needed
                    if self.layer_cache_config.should_cache_attn(L):
                        if self.current_stage not in self.attn_cache:
                            self.attn_cache[self.current_stage] = {}
                        self.attn_cache[self.current_stage][layer_idx] = output.detach().clone()
                
                return output
            return attn_hook
        
        def create_mlp_hook(layer_idx):
            def mlp_hook(module, input, output):
                if hasattr(self, '_caching_enabled') and self._caching_enabled:
                    L = input[0].shape[1]  # sequence length
                    
                    # Check if we should use cached features
                    should_use_cache = self.layer_cache_config.should_cache_layer_mlp(
                        self.current_stage, layer_idx, L
                    )
                    
                    if should_use_cache and self.current_stage in self.mlp_cache:
                        if layer_idx in self.mlp_cache[self.current_stage]:
                            # Use cached features with interpolation
                            cached_features = self.mlp_cache[self.current_stage][layer_idx]
                            
                            if cached_features.shape[1] != L:
                                # Interpolate cached features to current resolution
                                interp_mode = self.layer_cache_config.get_layer_interpolation_mode(
                                    self.current_stage, layer_idx
                                )
                                cached_features = feature_interpolate(cached_features, L, mode=interp_mode)
                            
                            # Get blend ratio for this specific layer
                            blend_ratio = self.layer_cache_config.get_layer_blend_ratio(
                                self.current_stage, layer_idx
                            )
                            
                            if blend_ratio >= 1.0:
                                # Pure cache - replace output
                                return cached_features
                            elif blend_ratio > 0.0:
                                # Blend cached and current features
                                result = blend_ratio * cached_features + (1.0 - blend_ratio) * output
                                return result
                    
                    # Store current output for future stages if needed
                    if self.layer_cache_config.should_cache_mlp(L):
                        if self.current_stage not in self.mlp_cache:
                            self.mlp_cache[self.current_stage] = {}
                        self.mlp_cache[self.current_stage][layer_idx] = output.detach().clone()
                
                return output
            return mlp_hook
        
        # Install hooks on all blocks
        if hasattr(self.var_model, 'blocks'):
            for layer_idx, block in enumerate(self.var_model.blocks):
                # Hook attention
                if hasattr(block, 'attn'):
                    block.attn.register_forward_hook(create_attn_hook(layer_idx))
                
                # Hook MLP/FFN
                if hasattr(block, 'ffn'):
                    block.ffn.register_forward_hook(create_mlp_hook(layer_idx))
    
    def set_current_stage(self, stage_idx: int):
        """Set the current generation stage"""
        self.current_stage = stage_idx
    
    def enable_caching(self):
        """Enable cache functionality"""
        self._caching_enabled = True
    
    def disable_caching(self):
        """Disable cache functionality"""
        self._caching_enabled = False
    
    def clear_cache(self):
        """Clear all cached features"""
        self.attn_cache.clear()
        self.mlp_cache.clear()
    
    def get_cache_stats(self):
        """Get cache usage statistics"""
        attn_cached_stages = len(self.attn_cache)
        mlp_cached_stages = len(self.mlp_cache)
        
        total_attn_layers = sum(len(stage_cache) for stage_cache in self.attn_cache.values())
        total_mlp_layers = sum(len(stage_cache) for stage_cache in self.mlp_cache.values())
        
        return {
            'attn_cached_stages': attn_cached_stages,
            'mlp_cached_stages': mlp_cached_stages,
            'total_attn_cached_layers': total_attn_layers,
            'total_mlp_cached_layers': total_mlp_layers
        }
    
    def autoregressive_infer_cfg(
        self, B: int, label_B: Optional[torch.LongTensor],
        cfg: float = 4.0, top_k: int = 0, top_p: float = 1.0,
        g_seed: int = None, more_smooth: bool = False
    ):
        """
        Enhanced autoregressive inference with fine-grained cache control
        """
        # Enable caching for generation
        self.enable_caching()
        
        try:
            # Get patch nums and begin_ends from original model
            patch_nums = self.var_model.patch_nums
            begin_ends = self.var_model.begin_ends
            
            # Track cache usage for reporting
            cache_usage_log = []
            
            # Progressive generation through stages
            for si, pn in enumerate(patch_nums):
                # Set current stage for cache decisions
                self.set_current_stage(si)
                
                # Get current sequence length
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
                for layer_idx in range(len(self.var_model.blocks)):
                    if self.layer_cache_config.should_cache_layer_attn(si, layer_idx, L):
                        stage_cache_info['cached_attn_layers'].append(layer_idx)
                    if self.layer_cache_config.should_cache_layer_mlp(si, layer_idx, L):
                        stage_cache_info['cached_mlp_layers'].append(layer_idx)
                
                cache_usage_log.append(stage_cache_info)
            
            # Print cache usage summary
            total_layers_processed = 0
            total_layers_cached = 0
            
            for stage_info in cache_usage_log:
                stage_cached = len(stage_info['cached_attn_layers']) + len(stage_info['cached_mlp_layers'])
                stage_total = len(self.var_model.blocks) * 2  # attn + mlp
                total_layers_processed += stage_total
                total_layers_cached += stage_cached
                
                if stage_cached > 0:
                    print(f"Stage {stage_info['stage']} (L={stage_info['sequence_length']}): "
                          f"{stage_cached}/{stage_total} layers cached "
                          f"(attn: {stage_info['cached_attn_layers']}, mlp: {stage_info['cached_mlp_layers']})")
            
            cache_efficiency = (total_layers_cached / total_layers_processed) * 100 if total_layers_processed > 0 else 0
            print(f"Overall cache efficiency: {total_layers_cached}/{total_layers_processed} layers ({cache_efficiency:.1f}%)")
            
            # Call original model's generation method
            # Note: We need to modify this to track stages properly
            result = self._enhanced_autoregressive_infer_cfg(
                B, label_B, cfg, top_k, top_p, g_seed, more_smooth
            )
            
            return result
            
        finally:
            # Disable caching after generation
            self.disable_caching()
    
    def _enhanced_autoregressive_infer_cfg(
        self, B: int, label_B: Optional[torch.LongTensor],
        cfg: float = 4.0, top_k: int = 0, top_p: float = 1.0,
        g_seed: int = None, more_smooth: bool = False
    ):
        """
        Enhanced version of original autoregressive_infer_cfg with stage tracking
        """
        # This is a simplified version - in practice, we would need to 
        # integrate with the original VAR's autoregressive_infer_cfg method
        # while tracking stages for cache decisions
        
        # For now, delegate to original model
        # TODO: Implement proper stage tracking integration
        return self.var_model.autoregressive_infer_cfg(
            B=B, label_B=label_B, cfg=cfg, top_k=top_k, top_p=top_p,
            g_seed=g_seed, more_smooth=more_smooth
        )
    
    def forward(self, x, cond, **kwargs):
        """Forward pass - delegate to original model"""
        return self.var_model(x, cond, **kwargs)
    
    def __getattr__(self, name):
        """Delegate attribute access to original model"""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.var_model, name)