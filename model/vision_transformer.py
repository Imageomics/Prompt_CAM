from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Set, Tuple, Type, Union, List

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.jit import Final
from timm.layers import DropPath, AttentionPoolLatent, RmsNorm, PatchDropout, SwiGLUPacked, \
    trunc_normal_, lecun_normal_, resample_patch_embed, resample_abs_pos_embed, use_fused_attn, \
    get_act_layer, get_norm_layer, LayerType
from timm.models._builder import build_model_with_cfg
from timm.models._manipulate import named_apply, checkpoint_seq, adapt_input_conv
from timm.models._registry import generate_default_cfgs, register_model, register_model_deprecations
from timm.models.vision_transformer import VisionTransformer
from timm.models.vision_transformer import LayerScale, init_weights_vit_timm, get_init_weights_vit, \
    _load_weights, checkpoint_filter_fn
## added for petl
from utils.setup_logging import get_logger
from model.block import BlockPETL
from model.patch_embed import PatchEmbedPETL
from model.mlp import MlpPETL
from model.vpt import VPT

logger = get_logger("Prompt_CAM")


class VisionTransformerPETL(VisionTransformer):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """
    dynamic_img_size: Final[bool]

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: Literal['', 'avg', 'token', 'map'] = 'token',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            init_values: Optional[float] = None,
            class_token: bool = True,
            no_embed_class: bool = False,
            reg_tokens: int = 0,
            pre_norm: bool = False,
            fc_norm: Optional[bool] = None,
            dynamic_img_size: bool = False,
            dynamic_img_pad: bool = False,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            weight_init: Literal['skip', 'jax', 'jax_nlhb', 'moco', ''] = '',
            embed_layer: Callable = PatchEmbedPETL,
            norm_layer: Optional[LayerType] = None,
            act_layer: Optional[LayerType] = None,
            block_fn: Type[nn.Module] = BlockPETL,
            mlp_layer: Type[nn.Module] = MlpPETL,
            params=None
    ) -> None:
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of image input channels.
            num_classes: Mumber of classes for classification head.
            global_pool: Type of global pooling for final sequence (default: 'token').
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            init_values: Layer-scale init values (layer-scale enabled if not None).
            class_token: Use class token.
            no_embed_class: Don't include position embeddings for class (or reg) tokens.
            reg_tokens: Number of register tokens.
            fc_norm: Pre head norm after pool (instead of before), if None, enabled when global_pool == 'avg'.
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            weight_init: Weight initialization scheme.
            embed_layer: Patch embedding layer.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'token', 'map')
        assert class_token or global_pool != 'token'
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = get_norm_layer(norm_layer) or partial(nn.LayerNorm, eps=1e-6)
        act_layer = get_act_layer(act_layer) or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = reg_tokens
        self.has_class_token = class_token
        self.no_embed_class = no_embed_class  # don't embed prefix positions (includes reg)
        self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = False

        embed_args = {}
        if dynamic_img_size:
            # flatten deferred until after pos embed
            embed_args.update(dict(strict_img_size=False, output_fmt='NHWC'))
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            dynamic_img_pad=dynamic_img_pad,
            params=params,
            **embed_args,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, reg_tokens, embed_dim)) if reg_tokens else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        ############# Added module start #############
        self.patch_size = patch_size
        self.params = params
        if self.params.train_type in ['vpt','prompt_cam']:
            self.vpt = VPT(params, depth, patch_size, embed_dim)
        ############# Added module end #############

        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
                ############# Added module start #############
                params=params
                ############# Added module end #############
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        if global_pool == 'map':
            self.attn_pool = AttentionPoolLatent(
                self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
            )
        else:
            self.attn_pool = None
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.num_heads= num_heads

        ############# Added module start #############
        if self.params.train_type == 'vpt' or self.params.train_type == 'linear':
            self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        elif self.params.train_type == 'prompt_cam':
            self.head = nn.Linear(self.embed_dim, 1)
        ############# Added module end #############
        

        if weight_init != 'skip':
            self.init_weights(weight_init)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path: str, prefix: str = '') -> None:
        _load_weights_PETL(self, checkpoint_path, prefix)

    def forward_features(self, x: torch.Tensor, blur_head_lst=[], target_cls=-1) -> Tuple[torch.Tensor,torch.Tensor]:
        pcam_outputs = None
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            ############# Added module #############
            for idx, block in enumerate(self.blocks):
                if self.params.train_type in ['vpt','prompt_cam']:
                    prompt = self.vpt.retrieve_prompt(idx, x.shape[0])
                    if prompt is not None:
                        x = torch.cat([prompt, x], dim=1)

                # forward block
                if idx == len(self.blocks) - 1:
                    x,attn_map = block(x, idx, blur_head_lst=blur_head_lst, target_cls=target_cls)
                else:
                    x,_ = block(x, idx)

                if self.params.vpt_mode and prompt is not None:
                    x = x[:, self.params.vpt_num:, :]
                elif self.params.train_type == 'prompt_cam':
                    pcam_outputs = x
                    x = x[:, self.params.vpt_num:, :]

            if self.params.train_type == 'prompt_cam':
                x = pcam_outputs
            ############# Added module end ############# 
        x = self.norm(x)
        return x,attn_map

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        #import pdb;pdb.set_trace()
        if self.params.train_type == 'prompt_cam':
            output_feature = x[:,:self.params.vpt_num]
        else:
            if self.attn_pool is not None:
                output_feature = self.attn_pool(x)
            elif self.global_pool == 'avg':
                output_feature = x[:, self.num_prefix_tokens:].mean(dim=1)
            elif self.global_pool:
                output_feature = x[:, 0]  # class token
            else:
                output_feature = x

            output_feature = self.fc_norm(output_feature)
            output_feature = self.head_drop(output_feature)

        return output_feature if pre_logits else self.head(output_feature)

    def forward(self, x: torch.Tensor, blur_head_lst=[], target_cls=-1) -> Tuple[torch.Tensor,torch.Tensor]:
        attn_maps = None
        if self.params.vis_attn:
            x,attn_maps = self.forward_features(x,
                                                blur_head_lst=blur_head_lst,
                                                target_cls=target_cls)
        else:
            x,_ = self.forward_features(x)

        x = self.forward_head(x)
        return x, attn_maps
    
    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'avgmax', 'max', 'token', 'map')
            if global_pool == 'map' and self.attn_pool is None:
                assert False, "Cannot currently add attention pooling in reset_classifier()."
            elif global_pool != 'map' and self.attn_pool is not None:
                self.attn_pool = None  # remove attention pooling
            self.global_pool = global_pool
        ############# Added module #############    
        if self.params.train_type == 'vpt' or self.params.train_type == 'linear':
            self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        else:
            self.head = nn.Linear(self.embed_dim, 1)    
        ############# Added module end #############
        #Original
        # self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()


@torch.no_grad()
def _load_weights_PETL(model: VisionTransformerPETL, checkpoint_path: str, prefix: str = ''):
    if checkpoint_path.endswith('.npz'):
        _load_weights(model, checkpoint_path, prefix)
    elif checkpoint_path.endswith('.pth') or checkpoint_path.endswith('.bin'):
        _load_weights_pth(model, checkpoint_path, checkpoint_filter_fn)


def _load_weights_pth(model, checkpoint_path, filter_fn=checkpoint_filter_fn):
    """ Load weights from .pth checkpoints
    """
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    if filter_fn is not None:
        state_dict = filter_fn(state_dict, model)
    if 'head.weight' in state_dict:
        state_dict.pop('head.weight', None)
    if 'head.bias' in state_dict:
        state_dict.pop('head.bias', None)
    model.load_state_dict(state_dict, strict=False)


def _create_vision_transformer_petl(variant: str, pretrained: bool = False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    if 'flexi' in variant:
        #  Google FlexiViT pretrained models have a strong preference for bilinear patch / embed
        # interpolation, other pretrained models resize better w/ anti-aliased bicubic interpolation.
        _filter_fn = partial(checkpoint_filter_fn, interpolation='bilinear', antialias=False)
    else:
        _filter_fn = checkpoint_filter_fn

    #  attn pool (currently only in siglip) params removed if pool disabled, is there a better soln?
    strict = True
    if 'siglip' in variant and kwargs.get('global_pool', None) != 'map':
        strict = False

    return build_model_with_cfg(
        VisionTransformerPETL,
        variant,
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_strict=strict,
        **kwargs,
    )


@register_model
def vit_base_patch14_dinov2_petl(pretrained: bool = False, **kwargs):
    """ ViT-B/14 for DINOv2
    change img_size to 224
    """
    model_args = dict(patch_size=14, embed_dim=768, depth=12, num_heads=12, init_values=1e-5, img_size=224)
    model = _create_vision_transformer_petl(
        'vit_base_patch14_dinov2', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_base_patch16_dino_petl(pretrained: bool = False, **kwargs):
    """ ViT-B/16 for DINO
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer_petl(
        'vit_base_patch16_224.dino', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_base_patch16_224_in21k_petl(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer_petl(
        'vit_base_patch16_224_in21k', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_patch16_clip_224_petl(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """ ViT-B/16 CLIP image tower
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, pre_norm=True, norm_layer=nn.LayerNorm,
                      act_layer='quick_gelu')
    model = _create_vision_transformer_petl(
        'vit_base_patch16_clip_quickgelu_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model
