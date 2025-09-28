import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import torch
import torch.fx
import torch.nn as nn
from torch import Tensor
from operator import mul



def stochastic_depth(input: Tensor, p: float, mode: str, training: bool = True):
    """
    Implements the Stochastic Depth from `"Deep Networks with Stochastic Depth"
    <https://arxiv.org/abs/1603.09382>`_ used for randomly dropping residual
    branches of residual architectures.

    Args:
        input (Tensor[N, ...]): The input tensor or arbitrary dimensions with the first one
                    being its batch i.e. a batch with ``N`` rows.
        p (float): probability of the input to be zeroed.
        mode (str): ``"batch"`` or ``"row"``.
                    ``"batch"`` randomly zeroes the entire input, ``"row"`` zeroes
                    randomly selected rows from the batch.
        training: apply stochastic depth if is ``True``. Default: ``True``

    Returns:
        Tensor[N, ...]: The randomly zeroed tensor.
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(f"drop probability has to be between 0 and 1, but got {p}")
    if mode not in ["batch", "row"]:
        raise ValueError(f"mode has to be either 'batch' or 'row', but got {mode}")
    if not training or p == 0.0:
        return input

    survival_rate = 1.0 - p
    if mode == "row":
        size = [input.shape[0]] + [1] * (input.ndim - 1)
    else:
        size = [1] * input.ndim
    noise = torch.empty(size, dtype=input.dtype, device=input.device)
    noise = noise.bernoulli_(survival_rate)
    if survival_rate > 0.0:
        noise.div_(survival_rate)
    return input * noise

torch.fx.wrap("stochastic_depth")



class StochasticDepth(nn.Module):
    """
    See :func:`stochastic_depth`.
    """

    def __init__(self, p: float, mode: str):
        super().__init__()
      
        self.p = p
        self.mode = mode

    def forward(self, input: Tensor):
        return stochastic_depth(input, self.p, self.mode, self.training)

    def __repr__(self):
        s = f"{self.__class__.__name__}(p={self.p}, mode={self.mode})"
        return s
class MLP(torch.nn.Sequential):
    """This block implements the multi-layer perceptron (MLP) module.

    Args:
        in_channels (int): Number of channels of the input
        hidden_channels (List[int]): List of the hidden channel dimensions
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the linear layer. If ``None`` this layer won't be used. Default: ``None``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the linear layer. If ``None`` this layer won't be used. Default: ``torch.nn.ReLU``
        inplace (bool, optional): Parameter for the activation layer, which can optionally do the operation in-place.
            Default is ``None``, which uses the respective default values of the ``activation_layer`` and Dropout layer.
        bias (bool): Whether to use bias in the linear layer. Default ``True``
        dropout (float): The probability for the dropout layer. Default: 0.0
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        inplace: Optional[bool] = None,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        # The addition of `norm_layer` is inspired from the implementation of TorchMultimodal:
        # https://github.com/facebookresearch/multimodal/blob/5dec8a/torchmultimodal/modules/layers/mlp.py
        params = {} if inplace is None else {"inplace": inplace}

        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer(**params))
            layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        layers.append(torch.nn.Dropout(dropout, **params))

        super().__init__(*layers)
        


class Permute(torch.nn.Module):
    """This module returns a view of the tensor input with its dimensions permuted.

    Args:
        dims (List[int]): The desired ordering of dimensions
    """

    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims

    def forward(self, x: Tensor) :
        return torch.permute(x, self.dims)



@dataclass
class MSBlockConfig:
    num_heads: int
    input_channels: int
    output_channels: int
    kernel_q: List[int]
    kernel_kv: List[int]
    stride_q: List[int]
    stride_kv: List[int]
    query_length: int
    prompt_length: int


def _prod(s: Sequence[int]):
    product = 1
    for v in s:
        product *= v
    return product


def _unsqueeze(x: torch.Tensor, target_dim: int, expand_dim: int):
    tensor_dim = x.dim()
    if tensor_dim == target_dim - 1:
        x = x.unsqueeze(expand_dim)
    elif tensor_dim != target_dim:
        raise ValueError(f"Unsupported input dimension {x.shape}")
    return x, tensor_dim


def _squeeze(x: torch.Tensor, target_dim: int, expand_dim: int, tensor_dim: int):
    if tensor_dim == target_dim - 1:
        x = x.squeeze(expand_dim)
    return x



torch.fx.wrap("_unsqueeze")
torch.fx.wrap("_squeeze")



class Pool(nn.Module):
    def __init__(
        self,
        pool: nn.Module,
        norm: Optional[nn.Module],
        activation: Optional[nn.Module] = None,
        norm_before_pool: bool = False,
    ):
        super().__init__()
        self.pool = pool
        layers = []
        if norm is not None:
            layers.append(norm)
        if activation is not None:
            layers.append(activation)
        self.norm_act = nn.Sequential(*layers) if layers else None
        self.norm_before_pool = norm_before_pool

    def forward(self, x: torch.Tensor, thw: Tuple[int, int, int],prompt_length:int):
        x, tensor_dim = _unsqueeze(x, 4, 1)

        # Separate the class token and reshape the input
        class_token, x = torch.tensor_split(x, indices=(1,), dim=2)
        prompt_tokens,x = x[:,:,:prompt_length],x[:,:,prompt_length:]
        x = x.transpose(2, 3)
        B, N, C = x.shape[:3]
        x = x.reshape((B * N, C) + thw).contiguous()

        # normalizing prior pooling is useful when we use BN which can be absorbed to speed up inference
        if self.norm_before_pool and self.norm_act is not None:
            x = self.norm_act(x)

        # apply the pool on the input and add back the token
        x = self.pool(x)
        T, H, W = x.shape[2:]
        x = x.reshape(B, N, C, -1).transpose(2, 3)
        x = torch.cat((class_token,prompt_tokens, x), dim=2)

        if not self.norm_before_pool and self.norm_act is not None:
            x = self.norm_act(x)

        x = _squeeze(x, 4, 1, tensor_dim)
        return x, (T, H, W)


def _interpolate(embedding: torch.Tensor, d: int):
    if embedding.shape[0] == d:
        return embedding

    return (
        nn.functional.interpolate(
            embedding.permute(1, 0).unsqueeze(0),
            size=d,
            mode="linear",
        )
        .squeeze(0)
        .permute(1, 0)
    )


def _add_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    q_thw: Tuple[int, int, int],
    k_thw: Tuple[int, int, int],
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    rel_pos_t: torch.Tensor,
    prompt_length = None,
):
    # Modified code from: https://github.com/facebookresearch/SlowFast/commit/1aebd71a2efad823d52b827a3deaf15a56cf4932
    q_t, q_h, q_w = q_thw
    k_t, k_h, k_w = k_thw
    dh = int(2 * max(q_h, k_h) - 1)
    dw = int(2 * max(q_w, k_w) - 1)
    dt = int(2 * max(q_t, k_t) - 1)

    # Scale up rel pos if shapes for q and k are different.
    q_h_ratio = max(k_h / q_h, 1.0)
    k_h_ratio = max(q_h / k_h, 1.0)
    dist_h = torch.arange(q_h)[:, None] * q_h_ratio - (torch.arange(k_h)[None, :] + (1.0 - k_h)) * k_h_ratio
    q_w_ratio = max(k_w / q_w, 1.0)
    k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = torch.arange(q_w)[:, None] * q_w_ratio - (torch.arange(k_w)[None, :] + (1.0 - k_w)) * k_w_ratio
    q_t_ratio = max(k_t / q_t, 1.0)
    k_t_ratio = max(q_t / k_t, 1.0)
    dist_t = torch.arange(q_t)[:, None] * q_t_ratio - (torch.arange(k_t)[None, :] + (1.0 - k_t)) * k_t_ratio

    # Interpolate rel pos if needed.
    rel_pos_h = _interpolate(rel_pos_h, dh)
    rel_pos_w = _interpolate(rel_pos_w, dw)
    rel_pos_t = _interpolate(rel_pos_t, dt)
    Rh = rel_pos_h[dist_h.long()]
    Rw = rel_pos_w[dist_w.long()]
    Rt = rel_pos_t[dist_t.long()]

    B, n_head, _, dim = q.shape
    if prompt_length is None:
        r_q = q[:, :, 1:].reshape(B, n_head, q_t, q_h, q_w, dim)
    else:
        r_q = q[:, :, 1 + prompt_length:].reshape(B, n_head, q_t, q_h, q_w, dim)
    rel_h_q = torch.einsum("bythwc,hkc->bythwk", r_q, Rh)  # [B, H, q_t, qh, qw, k_h]
    rel_w_q = torch.einsum("bythwc,wkc->bythwk", r_q, Rw)  # [B, H, q_t, qh, qw, k_w]
    # [B, H, q_t, q_h, q_w, dim] -> [q_t, B, H, q_h, q_w, dim] -> [q_t, B*H*q_h*q_w, dim]
    r_q = r_q.permute(2, 0, 1, 3, 4, 5).reshape(q_t, B * n_head * q_h * q_w, dim)
    # [q_t, B*H*q_h*q_w, dim] * [q_t, dim, k_t] = [q_t, B*H*q_h*q_w, k_t] -> [B*H*q_h*q_w, q_t, k_t]
    rel_q_t = torch.matmul(r_q, Rt.transpose(1, 2)).transpose(0, 1)
    # [B*H*q_h*q_w, q_t, k_t] -> [B, H, q_t, q_h, q_w, k_t]
    rel_q_t = rel_q_t.view(B, n_head, q_h, q_w, q_t, k_t).permute(0, 1, 4, 2, 3, 5)

    # Combine rel pos.
    rel_pos = (
        rel_h_q[:, :, :, :, :, None, :, None]
        + rel_w_q[:, :, :, :, :, None, None, :]
        + rel_q_t[:, :, :, :, :, :, None, None]
    ).reshape(B, n_head, q_t * q_h * q_w, k_t * k_h * k_w)

    if prompt_length is None:
        # Add it to attention
        attn[:, :, 1:, 1:] += rel_pos
    else:
        attn[:, :, 1+prompt_length:, 1+prompt_length:] += rel_pos

    return attn


def _add_shortcut(x: torch.Tensor, shortcut: torch.Tensor, residual_with_cls_embed: bool):
    if residual_with_cls_embed:
        x.add_(shortcut)
    else:
        x[:, :, 1:, :] += shortcut[:, :, 1:, :]
    return x


torch.fx.wrap("_add_rel_pos")
torch.fx.wrap("_add_shortcut")

class MultiscaleAttention(nn.Module):
    def __init__(
        self,
        input_size: List[int],
        embed_dim: int,
        output_dim: int,
        num_heads: int,
        kernel_q: List[int],
        kernel_kv: List[int],
        stride_q: List[int],
        stride_kv: List[int],
        residual_pool: bool,
        residual_with_cls_embed: bool,
        rel_pos_embed: bool,
        prompt_length,
        previous_prompt_length,
        query_length ,
        dropout: float = 0.0,
        is_first: bool = False,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.scaler = 1.0 / math.sqrt(self.head_dim)
        self.residual_pool = residual_pool
        self.residual_with_cls_embed = residual_with_cls_embed
        self.is_first = is_first
        self.previous_prompt_length = previous_prompt_length
        self.prompt_length = prompt_length

        self.qkv = nn.Linear(embed_dim, 3 * output_dim)
        layers: List[nn.Module] = [nn.Linear(output_dim, output_dim)]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout, inplace=True))
        self.project = nn.Sequential(*layers)

        self.pool_q: Optional[nn.Module] = None
        if _prod(kernel_q) > 1 or _prod(stride_q) > 1:
            padding_q = [int(q // 2) for q in kernel_q]
            self.pool_q = Pool(
                nn.Conv3d(
                    self.head_dim,
                    self.head_dim,
                    kernel_q,  # type: ignore[arg-type]
                    stride=stride_q,  # type: ignore[arg-type]
                    padding=padding_q,  # type: ignore[arg-type]
                    groups=self.head_dim,
                    bias=False,
                ),
                norm_layer(self.head_dim),
            )

        self.pool_k: Optional[nn.Module] = None
        self.pool_v: Optional[nn.Module] = None
        if _prod(kernel_kv) > 1 or _prod(stride_kv) > 1:
            padding_kv = [int(kv // 2) for kv in kernel_kv]
            self.pool_k = Pool(
                nn.Conv3d(
                    self.head_dim,
                    self.head_dim,
                    kernel_kv,  # type: ignore[arg-type]
                    stride=stride_kv,  # type: ignore[arg-type]
                    padding=padding_kv,  # type: ignore[arg-type]
                    groups=self.head_dim,
                    bias=False,
                ),
                norm_layer(self.head_dim),
            )
            self.pool_v = Pool(
                nn.Conv3d(
                    self.head_dim,
                    self.head_dim,
                    kernel_kv,  # type: ignore[arg-type]
                    stride=stride_kv,  # type: ignore[arg-type]
                    padding=padding_kv,  # type: ignore[arg-type]
                    groups=self.head_dim,
                    bias=False,
                ),
                norm_layer(self.head_dim),
            )

        self.rel_pos_h: Optional[nn.Parameter] = None
        self.rel_pos_w: Optional[nn.Parameter] = None
        self.rel_pos_t: Optional[nn.Parameter] = None
        if rel_pos_embed:
            size = max(input_size[1:])
            q_size = size // stride_q[1] if len(stride_q) > 0 else size
            kv_size = size // stride_kv[1] if len(stride_kv) > 0 else size
            spatial_dim = 2 * max(q_size, kv_size) - 1
            temporal_dim = 2 * input_size[0] - 1
            self.rel_pos_h = nn.Parameter(torch.zeros(spatial_dim, self.head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(spatial_dim, self.head_dim))
            self.rel_pos_t = nn.Parameter(torch.zeros(temporal_dim, self.head_dim))
            nn.init.trunc_normal_(self.rel_pos_h, std=0.02)
            nn.init.trunc_normal_(self.rel_pos_w, std=0.02)
            nn.init.trunc_normal_(self.rel_pos_t, std=0.02)

    def forward(self, x: torch.Tensor, thw: Tuple[int, int, int]) :

        B, N, C = x.shape

        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).transpose(1, 3).unbind(dim = 2)
        if self.pool_k is not None:
            k, k_thw = self.pool_k(k, thw, self.prompt_length if self.prompt_length is not None else self.previous_prompt_length)
        else:
            k_thw = thw
        if self.pool_v is not None:
            v = self.pool_v(v, thw,self.prompt_length if self.prompt_length is not None else self.previous_prompt_length)[0]
        if self.pool_q is not None:
            q, thw = self.pool_q(q, thw,self.prompt_length if self.prompt_length is not None else self.previous_prompt_length) # b,n_heads,n,dim
        

        attn = torch.matmul(self.scaler * q, k.transpose(2, 3))
        if self.rel_pos_h is not None and self.rel_pos_w is not None and self.rel_pos_t is not None:
            attn = _add_rel_pos(
                attn,
                q,
                thw,
                k_thw,
                self.rel_pos_h,
                self.rel_pos_w,
                self.rel_pos_t,
                prompt_length = self.prompt_length if self.prompt_length is not None else self.previous_prompt_length
            )
        attn = attn.softmax(dim=-1)

        x = torch.matmul(attn, v)
        if self.residual_pool:
            _add_shortcut(x, q, self.residual_with_cls_embed)
        x = x.transpose(1, 2).reshape(B, -1, self.output_dim)
        x = self.project(x)
       
       
        return x, thw,self.prompt_length if self.prompt_length is not None else self.previous_prompt_length

    
class MultiscaleBlock(nn.Module):
    def __init__(
        self,
        input_size: List[int],
        cnf: MSBlockConfig,
        residual_pool: bool,
        residual_with_cls_embed: bool,
        rel_pos_embed: bool,
        proj_after_attn: bool,
        dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        is_first: bool = False,
        previous_prompt_length: int = 200,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
    ) :
        super().__init__()
        self.proj_after_attn = proj_after_attn

        self.pool_skip: Optional[nn.Module] = None
        if _prod(cnf.stride_q) > 1:
            kernel_skip = [s + 1 if s > 1 else s for s in cnf.stride_q]
            padding_skip = [int(k // 2) for k in kernel_skip]
            self.pool_skip = Pool(
                nn.MaxPool3d(kernel_skip, stride=cnf.stride_q, padding=padding_skip), None  # type: ignore[arg-type]
            )

        attn_dim = cnf.output_channels if proj_after_attn else cnf.input_channels
        self.prompt_length = cnf.prompt_length
        self.is_first = is_first
        self.previous_prompt_length = previous_prompt_length
        self.left_prompt_embeddings = None
        self.right_prompt_embeddings = None
        self.center_prompt_embeddings = None

        if self.prompt_length is not None:
            self.left_prompt_embeddings = nn.Parameter(torch.zeros(1, self.prompt_length, cnf.input_channels))
            # xavier_uniform initialization
            nn.init.trunc_normal_(self.left_prompt_embeddings.data,std  = 0.02)

            self.right_prompt_embeddings = nn.Parameter(torch.zeros(1, self.prompt_length, cnf.input_channels))
            # xavier_uniform initialization
            nn.init.trunc_normal_(self.right_prompt_embeddings.data,  std = 0.02)

            self.center_prompt_embeddings = nn.Parameter(torch.zeros(1, self.prompt_length, cnf.input_channels))
            # xavier_uniform initialization
            nn.init.trunc_normal_(self.center_prompt_embeddings.data,  std = 0.02)

        self.norm1 = norm_layer(cnf.input_channels)
        self.norm2 = norm_layer(attn_dim)
        self.needs_transposal = isinstance(self.norm1, nn.BatchNorm1d)

        self.attn = MultiscaleAttention(
            input_size,
            cnf.input_channels,
            attn_dim,
            cnf.num_heads,
            kernel_q=cnf.kernel_q,
            kernel_kv=cnf.kernel_kv,
            stride_q=cnf.stride_q,
            stride_kv=cnf.stride_kv,
            rel_pos_embed=rel_pos_embed,
            residual_pool=residual_pool,
            residual_with_cls_embed=residual_with_cls_embed,
            dropout=dropout,
            norm_layer=norm_layer,
            query_length = cnf.query_length,
            prompt_length = cnf.prompt_length,
            previous_prompt_length = previous_prompt_length,
        )
        self.mlp = MLP(
            attn_dim,
            [4 * attn_dim, cnf.output_channels],
            activation_layer=nn.GELU,
            dropout=dropout,
            inplace=None,
        )

        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

        self.project: Optional[nn.Module] = None
        if cnf.input_channels != cnf.output_channels:
            self.project = nn.Linear(cnf.input_channels, cnf.output_channels)

    def forward(self, x: torch.Tensor, thw: Tuple[int, int, int],view:str):

        B, _,_ = x.shape
        prompt_embeddings = None
        if 'center' in view:
            prompt_embeddings = self.center_prompt_embeddings
        elif 'right' in view:
            prompt_embeddings = self.right_prompt_embeddings
        elif 'left' in view:
            prompt_embeddings = self.left_prompt_embeddings
        
        if prompt_embeddings is not None:
            prompt_embeddings = prompt_embeddings.repeat(B,1,1)
            if self.is_first:
                x = torch.cat([x[:,0:1],prompt_embeddings,x[:,1:]],dim = 1)
            else:
                x = torch.cat([x[:,0:1],prompt_embeddings,x[:,1 + self.previous_prompt_length:]],dim = 1)

        x_norm1 = self.norm1(x.transpose(1, 2)).transpose(1, 2) if self.needs_transposal else self.norm1(x)
        x_attn, thw_new, prompt_length = self.attn(x_norm1, thw)
      
        x = x if self.project is None or not self.proj_after_attn else self.project(x_norm1)
      
        x_skip = x if self.pool_skip is None else self.pool_skip(x, thw,prompt_length)[0]
     
        x = x_skip + self.stochastic_depth(x_attn)

        x_norm2 = self.norm2(x.transpose(1, 2)).transpose(1, 2) if self.needs_transposal else self.norm2(x)
        x_proj = x 
        return x_proj + self.stochastic_depth(self.mlp(x_norm2)), thw_new


class PositionalEncoding(nn.Module):
    def __init__(self, embed_size: int, spatial_size: Tuple[int, int], temporal_size: int, rel_pos_embed: bool):
        super().__init__()
        self.spatial_size = spatial_size
        self.temporal_size = temporal_size

        self.class_token = nn.Parameter(torch.zeros(embed_size))
     
    def forward(self, x: torch.Tensor):
        class_token = self.class_token.expand(x.size(0), -1).unsqueeze(1)
        x = torch.cat((class_token, x), dim=1)

        return x


def _unsqueeze(x: torch.Tensor, target_dim: int, expand_dim: int):
    tensor_dim = x.dim()
    if tensor_dim == target_dim - 1:
        x = x.unsqueeze(expand_dim)
    elif tensor_dim != target_dim:
        raise ValueError(f"Unsupported input dimension {x.shape}")
    return x, tensor_dim

class MViT_V2_Visual_Prompt_Tuning(nn.Module):
    def __init__(
        self,
        spatial_size: Tuple[int, int],
        temporal_size: int,
        block_setting: Sequence[MSBlockConfig],
        residual_pool: bool,
        residual_with_cls_embed: bool,
        rel_pos_embed: bool,
        proj_after_attn: bool,
        dropout: float = 0.5,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        num_classes: int = 400,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        patch_embed_kernel: Tuple[int, int, int] = (3, 7, 7),
        patch_embed_stride: Tuple[int, int, int] = (2, 4, 4),
        patch_embed_padding: Tuple[int, int, int] = (1, 3, 3),
        **kwargs):
       

        """
            MViT main class.

            Args:
                spatial_size (tuple of ints): The spacial size of the input as ``(H, W)``.
                temporal_size (int): The temporal size ``T`` of the input.
                block_setting (sequence of MSBlockConfig): The Network structure.
                residual_pool (bool): If True, use MViTv2 pooling residual connection.
                residual_with_cls_embed (bool): If True, the addition on the residual connection will include
                    the class embedding.
                rel_pos_embed (bool): If True, use MViTv2's relative positional embeddings.
                proj_after_attn (bool): If True, apply the projection after the attention.
                dropout (float): Dropout rate. Default: 0.0.
                attention_dropout (float): Attention dropout rate. Default: 0.0.
                stochastic_depth_prob: (float): Stochastic depth rate. Default: 0.0.
                num_classes (int): The number of classes.
                block (callable, optional): Module specifying the layer which consists of the attention and mlp.
                norm_layer (callable, optional): Module specifying the normalization layer to use.
                patch_embed_kernel (tuple of ints): The kernel of the convolution that patchifies the input.
                patch_embed_stride (tuple of ints): The stride of the convolution that patchifies the input.
                patch_embed_padding (tuple of ints): The padding of the convolution that patchifies the input.
        """

        super().__init__()
        print("Model: MViT_V2_Visual_Prompt_Tuning")
        # This implementation employs a different parameterization scheme than the one used at PyTorch Video:
        # https://github.com/facebookresearch/pytorchvideo/blob/718d0a4/pytorchvideo/models/vision_transformers.py
        # We remove any experimental configuration that didn't make it to the final variants of the models. To represent
        # the configuration of the architecture we use the simplified form suggested at Table 1 of the paper.
        total_stage_blocks = len(block_setting)
        if total_stage_blocks == 0:
            raise ValueError("The configuration parameter can't be empty.")

        if block is None:
            block = MultiscaleBlock

        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # Patch Embedding module
        self.conv_proj = nn.Conv3d(
            in_channels=3,
            out_channels=block_setting[0].input_channels,
            kernel_size=patch_embed_kernel,
            stride=patch_embed_stride,
            padding=patch_embed_padding,
        )

        input_size = [size // stride for size, stride in zip((temporal_size,) + spatial_size, self.conv_proj.stride)]

        # Spatio-Temporal Class Positional Encoding
        self.pos_encoding = PositionalEncoding(
            embed_size=block_setting[0].input_channels,
            spatial_size=(input_size[1], input_size[2]),
            temporal_size=input_size[0],
            rel_pos_embed=rel_pos_embed,
        )

        # Encoder module
        self.blocks = nn.ModuleList()

        previous_prompt_length = None
        for stage_block_id, cnf in enumerate(block_setting):
            
            # adjust stochastic depth probability based on the depth of the stage block
            sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)

            self.blocks.append(
                block(
                    input_size=input_size,
                    cnf=cnf,
                    residual_pool=residual_pool,
                    residual_with_cls_embed=residual_with_cls_embed,
                    rel_pos_embed=rel_pos_embed,
                    proj_after_attn=proj_after_attn,
                    dropout=attention_dropout,
                    stochastic_depth_prob=sd_prob,
                    norm_layer=norm_layer,
                    is_first = True if stage_block_id == 0  else False,
                    previous_prompt_length = previous_prompt_length
                )
            )
            previous_prompt_length = cnf.prompt_length if cnf.prompt_length else previous_prompt_length
           

            if len(cnf.stride_q) > 0:
                input_size = [size // stride for size, stride in zip(input_size, cnf.stride_q)]
        self.norm = norm_layer(block_setting[-1].output_channels)

        print("Num Classes",num_classes)
        self.num_classes = num_classes
        self.dropout = dropout
        self.block_setting = block_setting
        # Classifier module
        self.head = nn.Sequential(
            nn.Dropout(dropout, inplace=True),
            nn.Linear(block_setting[-1].output_channels, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, PositionalEncoding):
                for weights in m.parameters():
                    nn.init.trunc_normal_(weights, std=0.02)

        for name,params in self.named_parameters():
            if 'prompt_embeddings' not in name:
                params.requires_grad = False
        
        prompt_depth = kwargs.pop('prompt_depth',9)
        print("*"*20)
        print("Prompt_depth",prompt_depth)
        print("*"*20)


    def reset_head(self,num_classes):
        self.head = nn.Sequential(
            nn.Dropout(self.dropout, inplace=True),
            nn.Linear(self.block_setting[-1].output_channels, num_classes),
        )
        print("Reset to",num_classes)
    
    def forward_features(self,clip,view:str):
        x = clip
        # Convert if necessary (B, C, H, W) -> (B, C, 1, H, W)
        x = _unsqueeze(x, 5, 2)[0]
        # patchify and reshape: (B, C, T, H, W) -> (B, embed_channels[0], T', H', W') -> (B, THW', embed_channels[0])
        x = self.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)

        # add positional encoding
        x = self.pos_encoding(x)

        # pass patches through the encoder
        thw = (self.pos_encoding.temporal_size,) + self.pos_encoding.spatial_size
        for block in self.blocks:
            x, thw = block(x, thw,view)
        x = self.norm(x)

        # classifier "token" as used by standard language architectures
        x = x[:, 0]
        return x

    def forward(self, clip: torch.Tensor,view:str):
        x = clip
        # Convert if necessary (B, C, H, W) -> (B, C, 1, H, W)
        x = _unsqueeze(x, 5, 2)[0]
        # patchify and reshape: (B, C, T, H, W) -> (B, embed_channels[0], T', H', W') -> (B, THW', embed_channels[0])
        x = self.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)

        # add positional encoding
        x = self.pos_encoding(x)

        # pass patches through the encoder
        thw = (self.pos_encoding.temporal_size,) + self.pos_encoding.spatial_size
        for block in self.blocks:
            x, thw = block(x, thw,view)
        x = self.norm(x)

        # classifier "token" as used by standard language architectures
        x = x[:, 0]
        x = self.head(x)

        return {
            'logits':x
        }
    

def mvit_v2_s_visual_prompt_tuning(**kwargs):
    config: Dict[str, List] = {
        "num_heads": [1, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8],
        "input_channels": [96, 96, 192, 192, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 768],
        "output_channels": [96, 192, 192, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 768, 768],
        "kernel_q": [
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
        ],
        "kernel_kv": [
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
        ],
        "stride_q": [
            [1, 1, 1],
            [1, 2, 2],
            [1, 1, 1],
            [1, 2, 2],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 2, 2],
            [1, 1, 1],
        ],
        "stride_kv": [
            [1, 8, 8],
            [1, 4, 4],
            [1, 4, 4],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 1, 1],
            [1, 1, 1],
        ],
       'query_length':[25089,25089,6273,6273,1569,1569,1569,1569,1569,1569,1569,1569,1569,1569,1569,393],
    }
    prompt_depth = kwargs.pop('prompt_depth',9)
    if prompt_depth!=1:
        scale = [0.025,0.05,0.05,0.05,0.1,0.1,0.085,0.085,0.075,0.065,0.05,0.05,0.1,0.1,0.1,0.1]
    else:
        scale = [0.010]*16

    block_setting = []
   
    for i in range(len(config["num_heads"])):
        block_setting.append(
            MSBlockConfig(
                num_heads=config["num_heads"][i],
                input_channels=config["input_channels"][i],
                output_channels=config["output_channels"][i],
                kernel_q=config["kernel_q"][i],
                kernel_kv=config["kernel_kv"][i],
                stride_q=config["stride_q"][i],
                stride_kv=config["stride_kv"][i],
                prompt_length = int(scale[i]*config['query_length'][i]) if i < prompt_depth else None,
                query_length = config['query_length'][i]
            )
        )

    
    return MViT_V2_Visual_Prompt_Tuning(
        spatial_size=(224, 224),
        temporal_size=16,
        block_setting=block_setting,
        residual_pool=True,
        residual_with_cls_embed=False,
        rel_pos_embed=True,
        proj_after_attn=True,
        stochastic_depth_prob=kwargs.pop("stochastic_depth_prob", 0.2),
        num_classes = kwargs.pop('num_classes',400),
        prompt_depth = prompt_depth,
        **kwargs,
    )

