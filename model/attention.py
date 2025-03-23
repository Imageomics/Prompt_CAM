from torch.jit import Final
import torch.nn as nn
from timm.layers import use_fused_attn
import torch
import torch.nn.functional as F
from typing import Tuple


class AttentionPETL(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            params=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        ############# Added module #############
        self.params = params
        ############# Added module end #############

    def forward(self, x: torch.Tensor, block_idx, blur_head_lst=[],target_cls=-1) -> Tuple[torch.Tensor,torch.Tensor]:
        B, N, C = x.shape
        ############# Added module #############
        qkv = self.qkv(x)
        ############# Added module end #############

        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        ############# Added module #############
        if len(blur_head_lst)!=0:
            attn[:, blur_head_lst, target_cls, :] = 0

        ############# Added module end #############

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        proj = self.proj(x)
        x = self.proj_drop(proj)
        return x,attn
