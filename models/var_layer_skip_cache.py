"""
Enhanced VAR model with stage-specific layer cache-interpolation skipping functionality.
When a layer should be "skipped", it uses cached features from previous stages with interpolation.
"""

import math
from functools import partial
from typing import Optional, Tuple, Union, List, Dict, Set

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

import dist
from models.basic_var_enhanced import AdaLNBeforeHead, AdaLNSelfAttnEnhanced, CacheConfig
from models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from models.vqvae import VQVAE, VectorQuantizer2


def feature_interpolate(x, target_length: int, mode='bilinear'):
    """
    Interpolate features to target sequence length
    Args:
        x: Input tensor [B, L, C]
        target_length: Target sequence length
        mode: Interpolation mode
    Returns:
        Interpolated tensor [B, target_length, C]
    """
    if x.shape[1] == target_length:
        return x
    
    B, L, C = x.shape
    
    # Convert to 2D spatial representation for interpolation
    hw_src = int(math.sqrt(L))
    hw_tgt = int(math.sqrt(target_length))
    
    if hw_src * hw_src != L or hw_tgt * hw_tgt != target_length:
        # Fallback to linear interpolation for non-square sequences
        return F.interpolate(x.transpose(1, 2), size=target_length, mode='linear', align_corners=False).transpose(1, 2)
    
    # Reshape to spatial format and interpolate
    x_spatial = x.view(B, hw_src, hw_src, C).permute(0, 3, 1, 2)  # B, C, H, W
    x_interp = F.interpolate(x_spatial, size=(hw_tgt, hw_tgt), mode=mode, align_corners=False)
    x_output = x_interp.permute(0, 2, 3, 1).view(B, target_length, C)  # B, L, C
    
    return x_output


class LayerCacheSkipConfig:
    """Configuration class for layer cache-interpolation skipping"""
    def __init__(
        self,
        stage_layer_skip: Dict[int, Set[int]] = None,  # {stage_idx: {layer_indices_to_use_cache}}
        skip_all_layers_below: Dict[int, int] = None,  # {stage_idx: min_layer_idx}
        skip_all_layers_above: Dict[int, int] = None,  # {stage_idx: max_layer_idx}
        enable_skip: bool = True,
        interpolation_mode: str = 'bilinear',
        cache_similarity_threshold: float = 0.7,
        adaptive_cache_selection: bool = False,
        cache_blend_ratio: float = 0.8  # How much to blend cached vs current features
    ):
        """
        Args:
            stage_layer_skip: Dictionary mapping stage index to set of layer indices to use cache for
            skip_all_layers_below: Dictionary mapping stage index to minimum layer index (use cache for all below)
            skip_all_layers_above: Dictionary mapping stage index to maximum layer index (use cache for all above)
            enable_skip: Global enable/disable for cache-skipping
            interpolation_mode: Mode for feature interpolation
            cache_similarity_threshold: Threshold for cache similarity
            adaptive_cache_selection: Whether to adaptively select best cache
            cache_blend_ratio: Ratio for blending cached and current features
        
        Examples:
            # Use cache for layers 0,1,2 in the last stage (stage 9)
            LayerCacheSkipConfig(skip_all_layers_below={9: 3})
            
            # Use cache for specific layers in specific stages
            LayerCacheSkipConfig(stage_layer_skip={8: {0, 1, 15}, 9: {0, 1, 2}})
        """
        self.stage_layer_skip = stage_layer_skip or {}
        self.skip_all_layers_below = skip_all_layers_below or {}
        self.skip_all_layers_above = skip_all_layers_above or {}
        self.enable_skip = enable_skip
        self.interpolation_mode = interpolation_mode
        self.cache_similarity_threshold = cache_similarity_threshold
        self.adaptive_cache_selection = adaptive_cache_selection
        self.cache_blend_ratio = cache_blend_ratio
    
    def should_use_cache_for_layer(self, stage_idx: int, layer_idx: int) -> bool:
        """Check if a layer should use cached features instead of computation"""
        if not self.enable_skip:
            return False
        
        # Check specific layer skip rules
        if stage_idx in self.stage_layer_skip and layer_idx in self.stage_layer_skip[stage_idx]:
            return True
        
        # Check skip all layers below threshold
        if stage_idx in self.skip_all_layers_below and layer_idx < self.skip_all_layers_below[stage_idx]:
            return True
        
        # Check skip all layers above threshold
        if stage_idx in self.skip_all_layers_above and layer_idx > self.skip_all_layers_above[stage_idx]:
            return True
        
        return False
    
    def get_skip_summary(self, num_stages: int, num_layers: int) -> str:
        """Get a summary string of the cache-skip configuration"""
        if not self.enable_skip:
            return "Layer cache-skipping disabled"
        
        summary = []
        for stage_idx in range(num_stages):
            cached_layers = []
            for layer_idx in range(num_layers):
                if self.should_use_cache_for_layer(stage_idx, layer_idx):
                    cached_layers.append(layer_idx)
            
            if cached_layers:
                summary.append(f"Stage {stage_idx}: use cache for layers {cached_layers}")
        
        return "; ".join(summary) if summary else "No layers use cache"


class LayerFeatureCache:
    """Cache for storing layer features across stages"""
    def __init__(self, num_layers: int, num_stages: int):
        self.num_layers = num_layers
        self.num_stages = num_stages
        # Cache structure: [layer_idx][stage_idx] -> features
        self.cache = [[None for _ in range(num_stages)] for _ in range(num_layers)]
        self.cache_metadata = [[{} for _ in range(num_stages)] for _ in range(num_layers)]
    
    def store_features(self, layer_idx: int, stage_idx: int, features: torch.Tensor, metadata: dict = None):
        """Store features for a specific layer and stage"""
        if 0 <= layer_idx < self.num_layers and 0 <= stage_idx < self.num_stages:
            self.cache[layer_idx][stage_idx] = features.detach().clone()
            self.cache_metadata[layer_idx][stage_idx] = metadata or {}
    
    def get_cached_features(self, layer_idx: int, target_stage_idx: int, target_length: int, 
                          interpolation_mode: str = 'bilinear') -> Optional[torch.Tensor]:
        """
        Get cached features for a layer, interpolated to target length
        
        Args:
            layer_idx: Layer index to get cache for
            target_stage_idx: Current stage index
            target_length: Target sequence length
            interpolation_mode: Interpolation mode
            
        Returns:
            Interpolated cached features or None if no suitable cache found
        """
        if not (0 <= layer_idx < self.num_layers):
            return None
        
        # Look for cached features in previous stages (most recent first)
        for stage_idx in range(target_stage_idx - 1, -1, -1):
            if self.cache[layer_idx][stage_idx] is not None:
                cached_features = self.cache[layer_idx][stage_idx]
                
                # Interpolate to target length
                if cached_features.shape[1] != target_length:
                    interpolated_features = feature_interpolate(
                        cached_features, target_length, mode=interpolation_mode
                    )
                    return interpolated_features
                else:
                    return cached_features.clone()
        
        return None
    
    def clear_cache(self):
        """Clear all cached features"""
        for layer_idx in range(self.num_layers):
            for stage_idx in range(self.num_stages):
                self.cache[layer_idx][stage_idx] = None
                self.cache_metadata[layer_idx][stage_idx] = {}


class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)   # B16C


class VARLayerSkipCache(nn.Module, PyTorchModelHubMixin):
    """
    VAR model with stage-specific layer cache-interpolation skipping functionality
    """
    def __init__(
        self, vae_local: VQVAE,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., 
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True,
        # Layer cache-skipping parameters
        layer_cache_skip_config: LayerCacheSkipConfig = None,
        # Legacy cache parameters for compatibility
        use_cache=False, calibration=False, sim_path=None, threshold=0.7
    ):
        super().__init__()
        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        
        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1
        
        # VAR specific attributes
        self.patch_nums: Tuple = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2))
            cur += pn ** 2   # progressive training
        
        self.rng = torch.Generator(device=dist.get_device())
        
        # Layer cache-skipping configuration
        self.layer_cache_skip_config = layer_cache_skip_config or LayerCacheSkipConfig()
        self.layer_feature_cache = LayerFeatureCache(depth, len(self.patch_nums))
        
        # Legacy cache compatibility
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
                cache_config=CacheConfig()  # Enhanced blocks for potential future cache integration
            )
            for block_idx in range(depth)
        ])
        
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        
        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        
        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)
        
        print(f'[VARLayerSkipCache] Layer cache-skip config: {self.layer_cache_skip_config.get_skip_summary(len(self.patch_nums), depth)}')
    
    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h, cond_BD)).float()
    
    @torch.no_grad()
    def autoregressive_infer_cfg(
        self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
        more_smooth=False,
    ) -> torch.Tensor:   # returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        Autoregressive inference with stage-specific layer cache-interpolation skipping
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
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        
        # Clear layer feature cache for new generation
        self.layer_feature_cache.clear_cache()
        
        for b in self.blocks: 
            b.attn.kv_caching(True)
        
        # Track cache usage for logging
        total_layers_cached = 0
        total_layers_processed = 0
        
        # self.patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment (stage)
            ratio = si / self.num_stages_minus_1
            # last_L = cur_L
            cur_L += pn * pn
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            current_length = x.shape[1]
            
            # Apply transformer blocks with layer cache-skip logic
            layers_cached_this_stage = 0
            for layer_idx, block in enumerate(self.blocks):
                total_layers_processed += 1
                
                # Check if this layer should use cached features
                if self.layer_cache_skip_config.should_use_cache_for_layer(si, layer_idx):
                    cached_features = self.layer_feature_cache.get_cached_features(
                        layer_idx, si, current_length, 
                        self.layer_cache_skip_config.interpolation_mode
                    )
                    
                    if cached_features is not None:
                        # Use cached features with optional blending
                        if self.layer_cache_skip_config.cache_blend_ratio < 1.0:
                            # Compute current layer output for blending
                            current_output = block(
                                x=x, 
                                cond_BD=cond_BD_or_gss, 
                                attn_bias=None, 
                                cache_similarity_attn=self.cache_similarity_attn, 
                                cache_similarity_mlp=self.cache_similarity_mlp, 
                                cache_mlp=self.cache_mlp, 
                                cache_attn=self.cache_attn
                            )
                            
                            # Blend cached and current features
                            blend_ratio = self.layer_cache_skip_config.cache_blend_ratio
                            x = blend_ratio * cached_features + (1 - blend_ratio) * current_output
                        else:
                            # Use cached features directly
                            x = cached_features
                        
                        layers_cached_this_stage += 1
                        total_layers_cached += 1
                        continue
                
                # Process the layer normally
                x = block(
                    x=x, 
                    cond_BD=cond_BD_or_gss, 
                    attn_bias=None, 
                    cache_similarity_attn=self.cache_similarity_attn, 
                    cache_similarity_mlp=self.cache_similarity_mlp, 
                    cache_mlp=self.cache_mlp, 
                    cache_attn=self.cache_attn
                )
                
                # Store features in cache for future use
                self.layer_feature_cache.store_features(
                    layer_idx, si, x, 
                    metadata={'stage': si, 'patch_size': pn}
                )
            
            if layers_cached_this_stage > 0:
                print(f"Stage {si} (patch_size={pn}x{pn}): used cache for {layers_cached_this_stage}/{len(self.blocks)} layers")
            
            logits_BlV = self.get_logits(x, cond_BD)
            
            t = cfg * ratio
            logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]
            
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            if not more_smooth: # this is the default case
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae
            else:   # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                h_BChw = gumbel_softmax_with_rng(logits_BlV.div(gum_t), hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)
            if si != self.num_stages_minus_1:   # prepare for next stage
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si + 1] ** 2]
                next_token_map = next_token_map.repeat(2, 1, 1)    # double the batch sizes due to CFG
        
        for b in self.blocks: b.attn.kv_caching(False)
        
        cache_ratio = total_layers_cached / total_layers_processed if total_layers_processed > 0 else 0
        print(f"Total layer cache usage: {total_layers_cached}/{total_layers_processed} ({cache_ratio:.1%})")
        
        return self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)   # de-normalize, from [-1, 1] to [0, 1]
    
    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
        if init_std < 0: init_std = (1 / self.C / 3) ** 0.5     # init_std < 0: automated
        
        print(f'[init_weights] {type(self).__name__} with {init_std=:g}')
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()
            # conv: VAR has no conv, only VQVAE has conv
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0: nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else: nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias: m.bias.data.zero_()
        
        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()
        
        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()
        
        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, 'ada_lin'):
                sab.ada_lin[-1].weight.data[2*self.C:].mul_(init_adaln)
                sab.ada_lin[-1].weight.data[:2*self.C].mul_(init_adaln_gamma)
                if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)


# Factory function for easy model creation
def create_var_layer_skip_cache_model(
    vae_local: VQVAE,
    layer_cache_skip_config: LayerCacheSkipConfig = None,
    **kwargs
) -> VARLayerSkipCache:
    """
    Factory function to create VARLayerSkipCache model with specified cache-skip configuration
    """
    return VARLayerSkipCache(
        vae_local=vae_local,
        layer_cache_skip_config=layer_cache_skip_config,
        **kwargs
    )