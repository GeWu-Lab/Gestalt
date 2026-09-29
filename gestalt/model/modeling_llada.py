# coding=utf-8
from __future__ import annotations

import logging
import math
import sys
from abc import abstractmethod
from functools import partial
from typing import (
    Callable,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)
from dataclasses import fields

import torch
import torch.backends.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.auto import AutoModel
from transformers.cache_utils import Cache

logger = logging.getLogger(__name__)

from gestalt.model.config import SPECIAL_TOKENS
from gestalt.model.mrope import MRoPE2DRotaryEmbedding
from gestalt.model.flex_attention import (
    _compiled_flex_attn,
    create_block_masks,
    build_dense_attn_mask,
)


MASK = SPECIAL_TOKENS.MASK
NEWLINE = SPECIAL_TOKENS.NEWLINE
BOI = SPECIAL_TOKENS.IMAGE_START
EOI = SPECIAL_TOKENS.IMAGE_END
BOS = SPECIAL_TOKENS.BOS
EOS = SPECIAL_TOKENS.EOS
BOA = SPECIAL_TOKENS.ANSWER_START
EOA = SPECIAL_TOKENS.ANSWER_END
INTERACTION_START = SPECIAL_TOKENS.INTERACTION_TOKEN_START
INTERACTION_END = SPECIAL_TOKENS.INTERACTION_TOKEN_END


from .configuration_llada import (
    LLaDAConfig,
    StrEnum,
    InitFnType,
    ActivationType,
    BlockType,
    LayerNormType,
    ModelConfig,
    ActivationCheckpointingStrategy,
)

if sys.version_info.minor > 8:
    from collections.abc import MutableMapping
elif sys.version_info.minor == 8:
    from typing import MutableMapping
else:
    raise SystemExit("This script supports Python 3.8 or higher")

__all__ =[
    "LayerNormBase",
    "LayerNorm",
    "RMSLayerNorm",
    "GemmaRMSLayerNorm",
    "RotaryEmbedding",
    "Activation",
    "GELU",
    "ReLU",
    "SwiGLU",
    "LLaDABlock",
    "LLaDASequentialBlock",
    "LLaDAModel",
    "LLaDAOutput",
    "LLaDAGenerateOutput",
]

log = logging.getLogger(__name__)


class ModuleType(StrEnum):
    in_module = "in"
    out_module = "out"
    emb = "emb"
    final_out = "final_out"


def init_weights(
    config: ModelConfig,
    module: Union[nn.Linear, nn.Embedding],
    d: Optional[int] = None,
    layer_id: Optional[int] = None,
    std_factor: float = 1.0,
    type_of_module: Optional[ModuleType] = None,
) -> None:
    d = d if d is not None else config.d_model
    if config.init_fn == InitFnType.normal:
        std = config.init_std * std_factor
        if config.init_cutoff_factor is not None:
            cutoff_value = config.init_cutoff_factor * std
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-cutoff_value, b=cutoff_value)
        else:
            nn.init.normal_(module.weight, mean=0.0, std=std)
    elif config.init_fn == InitFnType.mitchell:
        std = std_factor / math.sqrt(d)
        if layer_id is not None:
            std = std / math.sqrt(2 * (layer_id + 1))
        nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
    elif config.init_fn == InitFnType.kaiming_normal:
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
    elif config.init_fn == InitFnType.fan_in:
        std = std_factor / math.sqrt(d)
        nn.init.normal_(module.weight, mean=0.0, std=std)
    elif config.init_fn == InitFnType.full_megatron:
        if type_of_module is None:
            raise RuntimeError(f"When using the {InitFnType.full_megatron} init, every module must have a type.")

        cutoff_factor = config.init_cutoff_factor
        if cutoff_factor is None:
            cutoff_factor = 3

        if type_of_module == ModuleType.in_module:
            std = config.init_std
        elif type_of_module == ModuleType.out_module:
            std = config.init_std / math.sqrt(2.0 * config.n_layers)
        elif type_of_module == ModuleType.emb:
            std = config.init_std
        elif type_of_module == ModuleType.final_out:
            std = config.d_model**-0.5
        else:
            raise RuntimeError(f"Unknown module type '{type_of_module}'")
        nn.init.trunc_normal_(
            module.weight,
            mean=0.0,
            std=std,
            a=-cutoff_factor * std,
            b=cutoff_factor * std,
        )
    else:
        raise NotImplementedError(config.init_fn)

    if isinstance(module, nn.Linear):
        if module.bias is not None:
            nn.init.zeros_(module.bias)

        if config.init_fn == InitFnType.normal and getattr(module, "_is_residual", False):
            with torch.no_grad():
                module.weight.div_(math.sqrt(2 * config.n_layers))


def ensure_finite_(x: torch.Tensor, check_neg_inf: bool = True, check_pos_inf: bool = False):
    if check_neg_inf:
        x.masked_fill_(x == float("-inf"), torch.finfo(x.dtype).min)
    if check_pos_inf:
        x.masked_fill_(x == float("inf"), torch.finfo(x.dtype).max)


def activation_checkpoint_function(cfg: ModelConfig):
    preserve_rng_state = (
        (cfg.attention_dropout == 0.0) and (cfg.embedding_dropout == 0.0) and (cfg.residual_dropout == 0.0)
    )
    from torch.utils.checkpoint import checkpoint

    return partial(
        checkpoint,
        preserve_rng_state=preserve_rng_state,
        use_reentrant=False,
    )


class BufferCache(dict, MutableMapping[str, torch.Tensor]):
    pass


def _non_meta_init_device(config: ModelConfig) -> torch.device:
    if config.init_device is not None and config.init_device != "meta":
        return torch.device(config.init_device)
    else:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dropout(nn.Dropout):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0:
            return input
        else:
            return F.dropout(input, self.p, self.training, self.inplace)


class LayerNormBase(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = True,
        eps: float = 1e-05,
    ):
        super().__init__()
        self.config = config
        self.eps = eps
        self.normalized_shape = (size or config.d_model,)
        if elementwise_affine or (elementwise_affine is None and self.config.layer_norm_with_affine):
            self.weight = nn.Parameter(torch.ones(self.normalized_shape, device=config.init_device))
            use_bias = self.config.bias_for_layer_norm
            if use_bias is None:
                use_bias = self.config.include_bias
            if use_bias:
                self.bias = nn.Parameter(torch.zeros(self.normalized_shape, device=config.init_device))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("bias", None)
            self.register_parameter("weight", None)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def build(cls, config: ModelConfig, size: Optional[int] = None, **kwargs) -> LayerNormBase:
        if config.layer_norm_type == LayerNormType.default:
            return LayerNorm(config, size=size, low_precision=False, **kwargs)
        elif config.layer_norm_type == LayerNormType.low_precision:
            return LayerNorm(config, size=size, low_precision=True, **kwargs)
        elif config.layer_norm_type == LayerNormType.rms:
            return RMSLayerNorm(config, size=size, **kwargs)
        elif config.layer_norm_type == LayerNormType.gemma_rms:
            return GemmaRMSLayerNorm(config, size=size, **kwargs)
        else:
            raise NotImplementedError(f"Unknown LayerNorm type: '{config.layer_norm_type}'")

    def _cast_if_autocast_enabled(self, tensor: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if tensor.device.type == "cuda" and torch.is_autocast_enabled():
            return tensor.to(dtype=dtype if dtype is not None else torch.get_autocast_gpu_dtype())
        elif tensor.device.type == "cpu" and torch.is_autocast_cpu_enabled():
            return tensor.to(dtype=dtype if dtype is not None else torch.get_autocast_cpu_dtype())
        else:
            return tensor

    def reset_parameters(self):
        if self.weight is not None:
            torch.nn.init.ones_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


class LayerNorm(LayerNormBase):
    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        low_precision: bool = False,
        elementwise_affine: Optional[bool] = None,
        eps: float = 1e-05,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine, eps=eps)
        self.low_precision = low_precision

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.low_precision:
            module_device = x.device
            downcast_x = self._cast_if_autocast_enabled(x)
            downcast_weight = (
                self._cast_if_autocast_enabled(self.weight) if self.weight is not None else self.weight
            )
            downcast_bias = self._cast_if_autocast_enabled(self.bias) if self.bias is not None else self.bias
            with torch.autocast(enabled=False, device_type=module_device.type):
                return F.layer_norm(
                    downcast_x, self.normalized_shape, weight=downcast_weight, bias=downcast_bias, eps=self.eps
                )
        else:
            return F.layer_norm(x, self.normalized_shape, weight=self.weight, bias=self.bias, eps=self.eps)


class RMSLayerNorm(LayerNormBase):
    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = None,
        eps: float = 1e-5,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=x.device.type):
            og_dtype = x.dtype
            x = x.to(torch.float32)
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            x = x.to(og_dtype)

        if self.weight is not None:
            if self.bias is not None:
                return self.weight * x + self.bias
            else:
                return self.weight * x
        else:
            return x


class GemmaRMSLayerNorm(LayerNormBase):
    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = None,
        eps: float = 1e-5,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=x.device.type):
            og_dtype = x.dtype
            x = x.to(torch.float32)
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            x = x.to(og_dtype)

        if self.weight is not None:
            if self.bias is not None:
                return x * (1 + self.weight) + self.bias
            else:
                return x * (1 + self.weight)
        else:
            return x


class RotaryEmbedding(nn.Module):
    def __init__(self, config: ModelConfig, cache: BufferCache):
        super().__init__()
        self.config = config
        self.__cache = cache
        self.rope_theta = config.rope_theta
        self.get_rotary_embedding(config.max_sequence_length, _non_meta_init_device(config))

    def get_rotary_embedding(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            (pos_sin := self.__cache.get("rope_pos_sin")) is not None
            and (pos_cos := self.__cache.get("rope_pos_cos")) is not None
            and pos_sin.shape[-2] >= seq_len
            and pos_cos.shape[-2] >= seq_len
        ):
            if pos_sin.device != device:
                pos_sin = pos_sin.to(device)
                self.__cache["rope_pos_sin"] = pos_sin
            if pos_cos.device != device:
                pos_cos = pos_cos.to(device)
                self.__cache["rope_pos_cos"] = pos_cos
            return pos_sin[:, :, :seq_len, :], pos_cos[:, :, :seq_len, :]

        with torch.autocast(device.type, enabled=False):
            dim = self.config.d_model // self.config.n_heads
            inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim))
            seq = torch.arange(seq_len, device=device, dtype=torch.float)
            freqs = einsum("i , j -> i j", seq, inv_freq)
            positions = torch.cat((freqs, freqs), dim=-1)
            pos_sin, pos_cos = positions.sin()[None, None, :, :], positions.cos()[None, None, :, :]
        self.__cache["rope_pos_sin"] = pos_sin
        self.__cache["rope_pos_cos"] = pos_cos
        return pos_sin, pos_cos

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        B, nh, T, hs = x.size()
        x = x.view(B, nh, T, 2, hs // 2)
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, pos_sin: torch.Tensor, pos_cos: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return ((t * pos_cos) + (self.rotate_half(t) * pos_sin)).to(t.dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor, q_mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.config.rope_full_precision:
            q_, k_ = q.float(), k.float()
        else:
            q_, k_ = q, k

        with torch.autocast(q.device.type, enabled=False):
            query_len, key_len = q_.shape[-2], k_.shape[-2]
            pos_sin, pos_cos = self.get_rotary_embedding(key_len, q_.device)
            pos_sin = pos_sin.type_as(q_)
            pos_cos = pos_cos.type_as(q_)
            if q_mask is None:
                q_ = self.apply_rotary_pos_emb(
                    pos_sin[:, :, key_len - query_len : key_len, :],
                    pos_cos[:, :, key_len - query_len : key_len, :],
                    q_,
                )
            else:
                q_ = self.apply_rotary_pos_emb(
                    pos_sin[:, :, q_mask, :],
                    pos_cos[:, :, q_mask, :],
                    q_,
                )
            k_ = self.apply_rotary_pos_emb(pos_sin, pos_cos, k_)
        return q_.type_as(q), k_.type_as(k)


class Activation(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def output_multiplier(self) -> float:
        raise NotImplementedError

    @classmethod
    def build(cls, config: ModelConfig) -> Activation:
        if config.activation_type == ActivationType.gelu:
            return cast(Activation, GELU(approximate="none"))
        elif config.activation_type == ActivationType.relu:
            return cast(Activation, ReLU(inplace=False))
        elif config.activation_type == ActivationType.silu:
            return cast(Activation, SiLU(inplace=False))
        elif config.activation_type == ActivationType.swiglu:
            return SwiGLU(config)
        else:
            raise NotImplementedError(f"Unknown activation: '{config.activation_type}'")


class GELU(nn.GELU):
    @property
    def output_multiplier(self) -> float:
        return 1.0

class ReLU(nn.ReLU):
    @property
    def output_multiplier(self) -> float:
        return 1.0

class SiLU(nn.SiLU):
    @property
    def output_multiplier(self) -> float:
        return 1.0

class SwiGLU(Activation):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

    @property
    def output_multiplier(self) -> float:
        return 0.5


def causal_attention_bias(seq_len: int, device: torch.device) -> torch.FloatTensor:
    att_bias = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.float),
        diagonal=1,
    )
    att_bias.masked_fill_(att_bias == 1, torch.finfo(att_bias.dtype).min)
    return att_bias.view(1, 1, seq_len, seq_len)  # type: ignore


def get_causal_attention_bias(cache: BufferCache, seq_len: int, device: torch.device) -> torch.Tensor:
    if (causal_bias := cache.get("causal_attention_bias")) is not None and causal_bias.shape[-1] >= seq_len:
        if causal_bias.device != device:
            causal_bias = causal_bias.to(device)
            cache["causal_attention_bias"] = causal_bias
        return causal_bias
    with torch.autocast(device.type, enabled=False):
        causal_bias = causal_attention_bias(seq_len, device)
    cache["causal_attention_bias"] = causal_bias
    return causal_bias


def alibi_attention_bias(seq_len: int, config: ModelConfig, device: torch.device) -> torch.FloatTensor:
    alibi_bias = torch.arange(1 - seq_len, 1, dtype=torch.float, device=device).view(1, 1, 1, seq_len)
    alibi_bias = alibi_bias - torch.arange(1 - seq_len, 1, dtype=torch.float, device=device).view(1, 1, seq_len, 1)
    alibi_bias.abs_().mul_(-1)
    m = torch.arange(1, config.n_heads + 1, dtype=torch.float, device=device)
    m.mul_(config.alibi_bias_max / config.n_heads)
    return alibi_bias * (1.0 / (2 ** m.view(1, config.n_heads, 1, 1)))  # type: ignore


class LLaDABlock(nn.Module):
    """
    A base class for transformer block implementations.
    """

    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        self.hidden_size = (
            config.mlp_hidden_size if config.mlp_hidden_size is not None else config.mlp_ratio * config.d_model
        )
        self.__cache = cache
        assert config.d_model % config.n_heads == 0

        self._activation_checkpoint_fn = None
        self.dropout = Dropout(config.residual_dropout)

        self.k_norm: Optional[LayerNormBase] = None
        self.q_norm: Optional[LayerNormBase] = None
        if config.attention_layer_norm:
            self.k_norm = LayerNormBase.build(
                config,
                size=(config.d_model // config.n_heads) * config.effective_n_kv_heads,
                elementwise_affine=config.attention_layer_norm_with_affine,
            )
            self.q_norm = LayerNormBase.build(config, elementwise_affine=config.attention_layer_norm_with_affine)

        self.act = Activation.build(config)
        assert (self.act.output_multiplier * self.hidden_size) % 1 == 0

        self.attn_out = nn.Linear(
            config.d_model, config.d_model, bias=config.include_bias, device=config.init_device
        )
        self.ff_out = nn.Linear(
            int(self.act.output_multiplier * self.hidden_size),
            config.d_model,
            bias=config.include_bias,
            device=config.init_device,
        )
        self.ff_out._is_residual = True  # type: ignore

        if self.config.rope:
            self.rotary_emb = MRoPE2DRotaryEmbedding(config)

        self.flash_attn_func = None
        if config.flash_attention:
            try:
                from flash_attn import flash_attn_func  # type: ignore
                self.flash_attn_func = flash_attn_func
            except ModuleNotFoundError:
                pass

        self.use_cache = False
        self.init_cache()

    def init_cache(self):
        self.cache = {
            'k': {}, 'v': {}, 'out': {}
        }

    def caching(self, enable: bool = True):
        self.use_cache = enable
        self.init_cache()

    def reset_parameters(self):
        if self.k_norm is not None:
            self.k_norm.reset_parameters()
        if self.q_norm is not None:
            self.q_norm.reset_parameters()
            
        init_weights(
            self.config,
            self.attn_out,
            d=self.config.d_model,
            layer_id=self.layer_id,
            type_of_module=ModuleType.out_module,
        )
        init_weights(
            self.config,
            self.ff_out,
            d=self.ff_out.in_features,
            layer_id=self.layer_id,
            type_of_module=ModuleType.out_module,
        )

    def set_activation_checkpointing(self, strategy: Optional[ActivationCheckpointingStrategy]):
        if strategy == ActivationCheckpointingStrategy.fine_grained:
            self._activation_checkpoint_fn = activation_checkpoint_function(self.config)
        else:
            self._activation_checkpoint_fn = None

    @classmethod
    def _cast_attn_bias(cls, bias: torch.Tensor, input_dtype: torch.dtype) -> torch.Tensor:
        target_dtype = input_dtype
        if torch.is_autocast_enabled(bias.device.type):
            target_dtype = torch.get_autocast_dtype(bias.device.type)
        if bias.dtype != target_dtype:
            bias = bias.to(target_dtype)
            ensure_finite_(bias, check_neg_inf=True, check_pos_inf=False)
        return bias

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        block_mask=None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        **kwargs,
    ):
        num_kv_heads = k.size(1)
        num_q_heads = q.size(1)

        if block_mask is not None:
            # Training path: FlexAttention (GQA handled natively)
            return _compiled_flex_attn(q, k, v, block_mask=block_mask, enable_gqa=True), None
        else:
            # Inference path: SDPA with dense mask
            if num_q_heads != num_kv_heads:
                assert num_q_heads % num_kv_heads == 0
                k = k.repeat_interleave(num_q_heads // num_kv_heads, dim=1)
                v = v.repeat_interleave(num_q_heads // num_kv_heads, dim=1)
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=False
            ), None
    


    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
        block_mask=None,
        to_compute_mask = None,
        layer_id = None,
        vision_mask = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = q.size()
        dtype = k.dtype

        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q).to(dtype=dtype)
            k = self.k_norm(k).to(dtype=dtype)

        q = q.view(B, -1, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
        k = k.view(B, -1, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)
        v = v.view(B, -1, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)

        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        if self.config.rope:
            q, k = self.rotary_emb(q, k, position_ids=position_ids)

        if attention_bias is not None:
            attention_bias = self._cast_attn_bias(
                attention_bias, dtype
            )

        # Get the attention scores.
        # shape: (B, nh, T, hs)
        att, attn_weights = self._scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_bias,
            block_mask=block_mask,
            dropout_p=0.0 if not self.training else self.config.attention_dropout,
            is_causal=False,
        )

        att = att.transpose(1, 2).contiguous().view(B, T, C)
        return self.attn_out(att), None

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        raise NotImplementedError

    @classmethod
    def build(cls, layer_id: int, config: ModelConfig, cache: BufferCache) -> LLaDABlock:
        if config.block_type == BlockType.sequential:
            return LLaDASequentialBlock(layer_id, config, cache)
        elif config.block_type == BlockType.llama:
            return LLaDALlamaBlock(layer_id, config, cache)
        else:
            raise NotImplementedError(f"Unknown block type: '{config.block_type}'")


class LLaDASequentialBlock(LLaDABlock):
    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__(layer_id, config, cache)
        self.attn_norm = LayerNorm.build(config)
        self.ff_norm = LayerNorm.build(config)
        head_dim = config.d_model // config.n_heads
        self.fused_dims = (
            config.d_model,
            config.effective_n_kv_heads * head_dim,
            config.effective_n_kv_heads * head_dim,
        )
        self.att_proj = nn.Linear(
            config.d_model, sum(self.fused_dims), bias=config.include_bias | config.include_qkv_bias, device=config.init_device
        )
        self.ff_proj = nn.Linear(
            config.d_model, self.hidden_size, bias=config.include_bias, device=config.init_device
        )

    def reset_parameters(self):
        super().reset_parameters()
        self.attn_norm.reset_parameters()
        self.ff_norm.reset_parameters()
        init_weights(self.config, self.att_proj, d=self.config.d_model, layer_id=None, type_of_module=ModuleType.in_module)
        init_weights(self.config, self.ff_proj, d=self.config.d_model, layer_id=None, type_of_module=ModuleType.in_module)

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        if self._activation_checkpoint_fn is not None:
            q, k, v = self.att_proj(self._activation_checkpoint_fn(self.attn_norm, x)).split(
                self.fused_dims, dim=-1
            )
        else:
            q, k, v = self.att_proj(self.attn_norm(x)).split(self.fused_dims, dim=-1)

        if self._activation_checkpoint_fn is not None:
            att, cache = self._activation_checkpoint_fn(  # type: ignore
                self.attention, q, k, v, attention_bias, layer_past=layer_past, position_ids=position_ids, use_cache=use_cache
            )
        else:
            att, cache = self.attention(q, k, v, attention_bias, layer_past=layer_past, position_ids=position_ids, use_cache=use_cache)

        x = x + self.dropout(att)
        og_x = x
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.ff_norm, x)  # type: ignore
        else:
            x = self.ff_norm(x)
        x = self.ff_proj(x)
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.act, x)  # type: ignore
        else:
            x = self.act(x)
        x = self.ff_out(x)
        x = self.dropout(x)
        x = og_x + x

        return x, cache


class LLaDALlamaBlock(LLaDABlock):
    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__(layer_id, config, cache)
        self.attn_norm = LayerNorm.build(config)
        self.ff_norm = LayerNorm.build(config)
        self.__cache = cache

        head_dim = config.d_model // config.n_heads
        q_proj_out_dim = config.d_model
        k_proj_out_dim = config.effective_n_kv_heads * head_dim
        v_proj_out_dim = config.effective_n_kv_heads * head_dim
        self.q_proj = nn.Linear(
            config.d_model, q_proj_out_dim, bias=config.include_bias | config.include_qkv_bias, device=config.init_device
        )
        self.k_proj = nn.Linear(
            config.d_model, k_proj_out_dim, bias=config.include_bias | config.include_qkv_bias, device=config.init_device
        )
        self.v_proj = nn.Linear(
            config.d_model, v_proj_out_dim, bias=config.include_bias | config.include_qkv_bias, device=config.init_device
        )

        self.ff_proj = nn.Linear(
            config.d_model, self.hidden_size, bias=config.include_bias, device=config.init_device
        )
        self.up_proj = nn.Linear(
            config.d_model,
            int(self.act.output_multiplier * self.hidden_size),
            bias=config.include_bias,
            device=config.init_device,
        )

    def reset_parameters(self):
        super().reset_parameters()
        self.attn_norm.reset_parameters()
        self.ff_norm.reset_parameters()
        init_weights(self.config, self.q_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.k_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.v_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.ff_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.up_proj, d=self.config.d_model, layer_id=None)

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        cat = 'cond',
        to_compute_mask = None,
        position_ids: Optional[torch.Tensor] = None,
        block_mask=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, D = x.shape
        self.current_vision_mask = (input_ids == BOI).cumsum(-1) >= 1
        x_normed = self.attn_norm(x)
        q = self.q_proj(x_normed)
        k = self.k_proj(x_normed)
        v = self.v_proj(x_normed)

        if use_cache:
            if cat not in self.cache['k']:
                self.cache['k'][cat] = torch.zeros_like(x)
                self.cache['v'][cat] = torch.zeros_like(x)
            if to_compute_mask is not None:
                self.cache['k'][cat][to_compute_mask] = k.view(-1, D)
                self.cache['v'][cat][to_compute_mask] = v.view(-1, D)
                k = self.cache['k'][cat]
                v = self.cache['v'][cat]
            else:
                self.cache['k'][cat] = k
                self.cache['v'][cat] = v

        if self._activation_checkpoint_fn is not None:
            att, cache = self._activation_checkpoint_fn(  # type: ignore
                self.attention, q, k, v, attention_bias, layer_past=layer_past, position_ids=position_ids,
                block_mask=block_mask, use_cache=use_cache, layer_id=self.layer_id, vision_mask=self.current_vision_mask
            )
        else:
            att, cache = self.attention(q, k, v, attention_bias, layer_past=layer_past, position_ids=position_ids,
                                        block_mask=block_mask, to_compute_mask=to_compute_mask, layer_id=self.layer_id, vision_mask=self.current_vision_mask)

        x = x + self.dropout(att)

        og_x = x
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.ff_norm, x)  # type: ignore
        else:
            x = self.ff_norm(x)
        x, x_up = self.ff_proj(x), self.up_proj(x)
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.act, x)  # type: ignore
        else:
            x = self.act(x)
        x = x * x_up
        x = self.ff_out(x)
        x = self.dropout(x)
        x = og_x + x

        return x, cache


class InteractionStageOneLlamaBlock(LLaDALlamaBlock):
    """
    专门用于处理多模态融合的层。通过动态屏蔽特定的模态交互，并具有针对文本与图像的双路 MLP。
    """
    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__(layer_id, config, cache)
        # Vision-specific MLP weights
        self.vision_ff_proj = nn.Linear(
            config.d_model, self.hidden_size, bias=config.include_bias, device=config.init_device
        )
        self.vision_up_proj = nn.Linear(
            config.d_model,
            int(self.act.output_multiplier * self.hidden_size),
            bias=config.include_bias,
            device=config.init_device,
        )
        self.vision_ff_out = nn.Linear(
            int(self.act.output_multiplier * self.hidden_size),
            config.d_model,
            bias=config.include_bias,
            device=config.init_device,
        )
        self.forward_mode = "interaction_token"

    def reset_parameters(self):
        super().reset_parameters()
        init_weights(self.config, self.vision_ff_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.vision_up_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.vision_ff_out, d=self.vision_ff_out.in_features, layer_id=self.layer_id, type_of_module=ModuleType.out_module)

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        block_mask=None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, D = x.shape

        # Compute the vision mask from explicit image boundaries.
        is_boi = (input_ids == BOI)
        is_eoi = (input_ids == EOI)


        in_image_block = (is_boi.cumsum(dim=-1) - is_eoi.cumsum(dim=-1)) > 0

        is_visual_id = (input_ids >= SPECIAL_TOKENS.VISUAL_TOKEN_OFFSET) & (input_ids < SPECIAL_TOKENS.VISUAL_TOKEN_END)

        is_img_special = (input_ids == NEWLINE) | (input_ids == MASK)
        
        is_visual = (is_boi | is_eoi) | (in_image_block & (is_visual_id | is_img_special))
        
        is_interaction_token = (input_ids >= INTERACTION_START) & (input_ids < INTERACTION_END)

        x_normed = self.attn_norm(x)
        q = self.q_proj(x_normed)
        k = self.k_proj(x_normed)
        v = self.v_proj(x_normed)

        # KV cache logic
        if use_cache:
            cat = kwargs.get('cat', 'cond')
            if cat not in self.cache['k']:
                self.cache['k'][cat] = torch.zeros_like(x)
                self.cache['v'][cat] = torch.zeros_like(x)
            to_compute_mask = kwargs.get('to_compute_mask', None)
            if to_compute_mask is not None:
                self.cache['k'][cat][to_compute_mask] = k.view(-1, D)
                self.cache['v'][cat][to_compute_mask] = v.view(-1, D)
                k = self.cache['k'][cat]
                v = self.cache['v'][cat]
            else:
                self.cache['k'][cat] = k
                self.cache['v'][cat] = v

        if self._activation_checkpoint_fn is not None:
            att, cache = self._activation_checkpoint_fn(
                self.attention, q, k, v, attention_bias,
                layer_past=layer_past, position_ids=position_ids,
                block_mask=block_mask, use_cache=use_cache,
                layer_id=self.layer_id, vision_mask=is_visual)
        else:
            att, cache = self.attention(q, k, v, attention_bias,
                layer_past=layer_past, position_ids=position_ids,
                block_mask=block_mask, layer_id=self.layer_id, vision_mask=is_visual)

        x = x + self.dropout(att)
        og_x = x

        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.ff_norm, x)
        else:
            x = self.ff_norm(x)

        # Text MLP
        x_text, x_text_up = self.ff_proj(x), self.up_proj(x)
        if self._activation_checkpoint_fn is not None:
            x_text = self._activation_checkpoint_fn(self.act, x_text)
        else:
            x_text = self.act(x_text)
        x_text = x_text * x_text_up
        text_out = self.ff_out(x_text)

        # Vision MLP
        x_vis, x_vis_up = self.vision_ff_proj(x), self.vision_up_proj(x)
        if self._activation_checkpoint_fn is not None:
            x_vis = self._activation_checkpoint_fn(self.act, x_vis)
        else:
            x_vis = self.act(x_vis)
        x_vis = x_vis * x_vis_up
        vision_out = self.vision_ff_out(x_vis)

        # Route by modality
        if self.forward_mode == "interaction_token":
            inter_out = (text_out + vision_out) / 2
            
        new_hidden_states = torch.where(is_visual.unsqueeze(-1), vision_out, text_out)
        
        if self.forward_mode == "interaction_token":
            new_hidden_states = torch.where(
                is_interaction_token.unsqueeze(-1),
                inter_out,
                new_hidden_states,
            )
        
        x = og_x + self.dropout(new_hidden_states)
        return x, cache


class LLaDAOutput(NamedTuple):
    logits: torch.FloatTensor
    attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]
    hidden_states: Optional[Tuple[torch.Tensor]]


class LLaDAGenerateOutput(NamedTuple):
    token_ids: torch.LongTensor
    scores: torch.FloatTensor


class LLaDABlockGroup(nn.ModuleList):
    def __init__(self, config: ModelConfig, layer_offset: int, modules: Optional[Iterable[nn.Module]] = None):
        super().__init__(modules)
        self.config = config
        self.layer_offset = layer_offset
        self.activation_checkpointing_strategy: Optional[ActivationCheckpointingStrategy] = None
        self._activation_checkpoint_fn = activation_checkpoint_function(self.config)

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.FloatTensor] = None,
        layers_past: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        block_mask=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] =[] if use_cache else None
        for block_idx, block in enumerate(self):
            layer_past = None if layers_past is None else layers_past[block_idx]
            block_idx += self.layer_offset
            if (
                (self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.whole_layer)
                or (self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_two and block_idx % 2 == 0)
                or (self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_three and block_idx % 3 == 0)
                or (self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_four and block_idx % 4 == 0)
            ):
                x, cache = self._activation_checkpoint_fn(
                    block, x, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache,
                    input_ids=input_ids, position_ids=position_ids, block_mask=block_mask, **kwargs
                )
            else:
                x, cache = block(x, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache,
                    input_ids=input_ids, position_ids=position_ids, block_mask=block_mask, **kwargs)
            if attn_key_values is not None:
                assert cache is not None
                attn_key_values.append(cache)
        return x, attn_key_values

    def reset_parameters(self):
        for block in self:
            block.reset_parameters()

    def set_activation_checkpointing(self, strategy: Optional[ActivationCheckpointingStrategy]):
        self.activation_checkpointing_strategy = strategy
        for block in self:
            block.set_activation_checkpointing(strategy)


class LLaDAModel(nn.Module):
    def __init__(self, config: ModelConfig, init_params: bool = True):
        super().__init__()
        self.config = config
        self.__cache = BufferCache()

        if self.config.alibi and self.config.flash_attention:
            raise Exception("ALiBi is currently not supported with FlashAttention")

        if self.config.alibi and self.config.rope:
            raise Exception("ALiBi and RoPE are mutually exclusive")

        if self.config.embedding_size is not None and self.config.embedding_size != self.config.vocab_size:
            if self.config.embedding_size < self.config.vocab_size:
                raise Exception("embedding size should be at least as big as vocab size")
            elif self.config.embedding_size % 128 != 0:
                import warnings
                warnings.warn(
                    "Embedding size is not a multiple of 128! This could hurt throughput performance.", UserWarning
                )

        self.activation_checkpointing_strategy: Optional[ActivationCheckpointingStrategy] = None
        self._activation_checkpoint_fn: Callable = activation_checkpoint_function(self.config)

        if not (
            0 < self.config.block_group_size <= self.config.n_layers
            and self.config.n_layers % self.config.block_group_size == 0
        ):
            raise Exception("n layers must be divisible by block group size")

        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(
                    config.embedding_size or config.vocab_size, config.d_model, device=config.init_device
                ),
                emb_drop=Dropout(config.embedding_dropout),
                ln_f=LayerNorm.build(config),
            )
        )

        self.inter_tokens_num = 32
        self.inter_layer_num = getattr(config, "inter_layer_num", 8)
        if (
            config.block_group_size > 1
            and self.inter_layer_num % config.block_group_size != 0
        ):
            raise ValueError(
                "inter_layer_num must align with block_group_size so interaction "
                "layers receive the isolated attention mask."
            )

        blocks =[]
        for i in range(config.n_layers):
            if i < self.inter_layer_num:
                blocks.append(InteractionStageOneLlamaBlock(i, config, self.__cache))
                if getattr(config, "inter_layer_state", None) is not None:
                    logger.debug("config.inter_layer_state: %s", config.inter_layer_state)
                    blocks[-1].forward_mode = config.inter_layer_state
                
                pass  # InteractionStageOneLlamaBlock uses FlexAttention block_mask_isolated
            else:
                blocks.append(LLaDABlock.build(i, config, self.__cache))
                
        if self.config.block_group_size > 1:
            block_groups =[
                LLaDABlockGroup(config, i, blocks[i : i + config.block_group_size])
                for i in range(0, config.n_layers, config.block_group_size)
            ]
            self.transformer.update({"block_groups": nn.ModuleList(block_groups)})
        else:
            self.transformer.update({"blocks": nn.ModuleList(blocks)})

        if not (self.config.alibi or self.config.rope):
            self.transformer.update(
                {"wpe": nn.Embedding(config.max_sequence_length, config.d_model, device=config.init_device)}
            ) 
        if not config.weight_tying:
            self.transformer.update(
                {
                    "ff_out": nn.Linear(
                        config.d_model,
                        config.embedding_size or config.vocab_size,
                        bias=config.include_bias,
                        device=config.init_device,
                    )
                }
            )
            
        if init_params and self.config.init_device != "meta":
            self.reset_parameters()
        self.__num_fwd_flops: Optional[int] = None

        if self.config.alibi:
            get_causal_attention_bias(self.__cache, config.max_sequence_length, _non_meta_init_device(config))
            self.get_alibi_attention_bias(config.max_sequence_length, _non_meta_init_device(config))

        self.logit_cache = {}

    def get_interaction_tokens(self):
        return self.interaction_tokens

    def set_activation_checkpointing(self, strategy: Optional[ActivationCheckpointingStrategy]):
        self.activation_checkpointing_strategy = strategy
        if self.config.block_group_size != 1:
            for block_group in self.transformer.block_groups:
                block_group.set_activation_checkpointing(strategy)
        else:
            for block in self.transformer.blocks:
                block.set_activation_checkpointing(strategy)

    @property
    def device(self) -> torch.device:
        device: torch.device = self.transformer.wte.weight.device
        if device.type == "meta":
            return _non_meta_init_device(self.config)
        else:
            return device

    def reset_parameters(self):
        log.info("Initializing model parameters...")
        init_weights(
            self.config,
            self.transformer.wte,
            std_factor=(0.5 * math.sqrt(self.config.d_model)) if self.config.scale_logits else 1.0, 
            type_of_module=ModuleType.emb,
        )
        if hasattr(self.transformer, "wpe"):
            init_weights(self.config, self.transformer.wpe, type_of_module=ModuleType.emb)

        self.transformer.ln_f.reset_parameters()

        if hasattr(self.transformer, "ff_out"):
            init_weights(self.config, self.transformer.ff_out, type_of_module=ModuleType.final_out)

        if self.config.block_group_size == 1:
            for block in self.transformer.blocks:
                block.reset_parameters()
        else:
            for block_group in self.transformer.block_groups:
                block_group.reset_parameters()

    def get_alibi_attention_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if (alibi_bias := self.__cache.get("alibi_attention_bias")) is not None and alibi_bias.shape[-1] >= seq_len:
            if alibi_bias.device != device:
                alibi_bias = alibi_bias.to(device)
                self.__cache["alibi_attention_bias"] = alibi_bias
            return alibi_bias
        with torch.autocast(device.type, enabled=False):
            alibi_bias = alibi_attention_bias(seq_len, self.config, device)
        self.__cache["alibi_attention_bias"] = alibi_bias
        return alibi_bias

    def forward(
        self,
        input_ids: torch.LongTensor,
        input_embeddings: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        past_key_values: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
        last_logits_only: bool = False,
        output_hidden_states: Optional[bool] = None,
        use_cache = False,
        to_compute_mask = None,
        cat = '',
        position_ids: Optional[torch.Tensor] = None,
        document_ids: Optional[torch.Tensor] = None,
    ) -> LLaDAOutput:

        if use_cache and to_compute_mask is not None:
            input_ids = input_ids[to_compute_mask].view(input_ids.shape[0], -1)

        assert not self.config.alibi, "Alibi length extrapolation is not supported for MDM."
        assert self.config.rope, "Rope must be used in Llama-Encoder for MDM."

        output_hidden_states = output_hidden_states if output_hidden_states is not None else False

        if past_key_values:
            assert len(past_key_values) == self.config.n_layers

        batch_size, seq_len = input_ids.size() if input_embeddings is None else input_embeddings.size()[:2]
        if past_key_values is None:
            past_length = 0
        else:
            past_length = past_key_values[0][0].size(-2)

        x = self.transformer.wte(input_ids) if input_embeddings is None else input_embeddings

        if self.config.input_emb_norm:
            x = x * (self.config.d_model**0.5)

        if not (self.config.alibi or self.config.rope):
            pos = torch.arange(past_length, past_length + seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
            pos_emb = self.transformer.wpe(pos)
            x = pos_emb + x

        x = self.transformer.emb_drop(x)

        # Build default 1D position_ids if not provided (h_pos == w_pos)
        if position_ids is None:
            logger.debug("position_ids is None; using default 1D positions")
            pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
            position_ids = torch.stack([pos, pos])  # (2, B, L)

        if document_ids is None:
            raise ValueError(
                "document_ids must be provided explicitly. BOS/EOS tokens are "
                "text-span boundaries and are not used to infer document boundaries."
            )
        if document_ids.shape != input_ids.shape:
            raise ValueError(
                f"document_ids shape {tuple(document_ids.shape)} does not match "
                f"input_ids shape {tuple(input_ids.shape)}."
            )

        is_boi = (input_ids == BOI)
        is_eoi = (input_ids == EOI)
        
        in_image_block = (is_boi.cumsum(dim=-1) - is_eoi.cumsum(dim=-1)) > 0
        is_visual_id = (input_ids >= SPECIAL_TOKENS.VISUAL_TOKEN_OFFSET) & (input_ids < SPECIAL_TOKENS.VISUAL_TOKEN_END)

        is_img_special = (input_ids == NEWLINE) | (input_ids == MASK)
        
        is_image_token = (is_boi | is_eoi) | (in_image_block & (is_visual_id | is_img_special))
        
        is_interaction_token = (input_ids >= INTERACTION_START) & (input_ids < INTERACTION_END)
        
        is_padding = (input_ids == SPECIAL_TOKENS.PADDING)

        # CUDA training uses FlexAttention. CPU training/smoke tests and all
        # inference use the equivalent dense SDPA masks.
        if self.training and x.device.type == "cuda":
            block_mask_full, block_mask_isolated = create_block_masks(
                input_ids, document_ids, is_image_token, is_padding, is_interaction_token,
                n_heads=self.config.n_heads, device=x.device,
            )
            dense_attn_mask_full = None
            dense_attn_mask_isolated = None
        else:
            block_mask_full = None
            block_mask_isolated = None
            dense_attn_mask_full = build_dense_attn_mask(
                input_ids, document_ids, is_image_token, is_padding, is_interaction_token, isolated=False)
            dense_attn_mask_isolated = build_dense_attn_mask(
                input_ids, document_ids, is_image_token, is_padding, is_interaction_token, isolated=True)

        attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] =[] if use_cache else None
        all_hidden_states =[]

        if self.config.block_group_size == 1:
            for block_idx, block in enumerate(self.transformer.blocks):
                if output_hidden_states:
                    all_hidden_states.append(x)

                layer_past = None if past_key_values is None else past_key_values[block_idx]

                # Select mask based on block type: interaction blocks get isolated mask
                if block_idx < self.inter_layer_num:
                    cur_block_mask = block_mask_isolated
                    cur_dense_mask = dense_attn_mask_isolated
                else:
                    cur_block_mask = block_mask_full
                    cur_dense_mask = dense_attn_mask_full

                if (
                    (self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.whole_layer)
                    or (self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_two and block_idx % 2 == 0)
                    or (self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_three and block_idx % 3 == 0)
                    or (self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_four and block_idx % 4 == 0)
                ):
                    x, _ = self._activation_checkpoint_fn(
                        block, x, attention_bias=cur_dense_mask, layer_past=layer_past,
                        block_mask=cur_block_mask, position_ids=position_ids,
                        to_compute_mask=to_compute_mask, use_cache=use_cache, cat=cat, input_ids=input_ids
                    )
                else:
                    x, _ = block(x, attention_bias=cur_dense_mask, layer_past=layer_past,
                        block_mask=cur_block_mask, position_ids=position_ids,
                        to_compute_mask=to_compute_mask, use_cache=use_cache, cat=cat, input_ids=input_ids
                    )
        else:
            for group_idx, block_group in enumerate(self.transformer.block_groups):
                if output_hidden_states:
                    all_hidden_states.append(x)

                layers_past = (
                    None
                    if past_key_values is None
                    else past_key_values[
                        group_idx * self.config.block_group_size : (group_idx + 1) * self.config.block_group_size
                    ]
                )

                # Determine mask for this group based on layer range
                group_start = group_idx * self.config.block_group_size
                group_end = group_start + self.config.block_group_size
                if group_end <= self.inter_layer_num:
                    cur_block_mask = block_mask_isolated
                    cur_dense_mask = dense_attn_mask_isolated
                elif group_start >= self.inter_layer_num:
                    cur_block_mask = block_mask_full
                    cur_dense_mask = dense_attn_mask_full
                else:
                    raise RuntimeError(
                        "Interaction and standard layers cannot share one block group."
                    )

                x, _ = block_group(
                    x, attention_bias=cur_dense_mask, layers_past=layers_past,
                    block_mask=cur_block_mask, position_ids=position_ids,
                    to_compute_mask=to_compute_mask, use_cache=use_cache, cat=cat, input_ids=input_ids
                )

        if last_logits_only:
            x = x[:, -1, :].unsqueeze(1)

        x = self.transformer.ln_f(x)
        if output_hidden_states:
            all_hidden_states.append(x)

        if self.config.weight_tying:
            logits = F.linear(x, self.transformer.wte.weight, None)
        else:
            logits = self.transformer.ff_out(x)
        if self.config.scale_logits:
            logits.mul_(1 / math.sqrt(self.config.d_model))

        if use_cache:
            if cat not in self.logit_cache:
                self.logit_cache[cat] = torch.zeros_like(logits)
            if to_compute_mask is not None:
                self.logit_cache[cat][to_compute_mask] = logits.view(-1, logits.shape[-1])
                logits = self.logit_cache[cat]
            else:
                self.logit_cache[cat] = logits

        return LLaDAOutput(logits=logits, attn_key_values=attn_key_values, hidden_states=tuple(all_hidden_states) if output_hidden_states else None)
    
    def caching(self, enable: bool = True):
        for block in self.transformer.blocks:
            block.caching(enable)
        self.logit_cache = {}

    def empty_cache(self):
        for block in self.transformer.blocks:
            block.init_cache()
        self.logit_cache = {}


def create_model_config_from_pretrained_config(config: LLaDAConfig):
    kwargs = {}
    for field in fields(ModelConfig):
        kwargs[field.name] = getattr(config, field.name)
    model_config = ModelConfig(**kwargs)

    # Gestalt-specific fields are not part of the base LLaDA ModelConfig
    # dataclass, but LLaDAModel uses them when choosing interaction blocks.
    for name in ("inter_layer_num", "inter_layer_state", "inter_tokens_num"):
        if hasattr(config, name):
            setattr(model_config, name, getattr(config, name))

    return model_config


class LLaDAModelLM(PreTrainedModel):
    """
    Extremely barebones HF model wrapper.
    """

    config_class = LLaDAConfig
    base_model_prefix = "model"
    _no_split_modules =["LLaDABlock", "LLaDASequentialBlock", "LLaDALlamaBlock", "InteractionStageOneLlamaBlock"]

    def __init__(self, config: LLaDAConfig, model: Optional[LLaDAModel] = None, init_params: bool = False):
        super().__init__(config)

        if not model:
            model_config = create_model_config_from_pretrained_config(config)
            self.model = LLaDAModel(model_config, init_params=init_params)
        else:
            self.model = model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[Cache] = None,
        use_cache = False,
        to_compute_mask = None,
        cat = '',
        position_ids: Optional[torch.Tensor] = None,
        document_ids: Optional[torch.Tensor] = None,
        # conversation_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if output_attentions:
            raise ValueError("output_attentions is not yet supported in LLaDA")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model.forward(
            input_ids=input_ids,
            input_embeddings=inputs_embeds,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
            past_key_values=past_key_values,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            to_compute_mask=to_compute_mask,
            cat=cat,
            position_ids=position_ids,
            document_ids=document_ids,
        )

        logits = outputs.logits
        hidden_states = outputs.hidden_states

        loss = None
        if labels is not None:
            import warnings
            warnings.warn("Note that for LLaDA, you cannot calculate the loss here.", UserWarning)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.attn_key_values,
            hidden_states=hidden_states,
        )

    def can_generate(self) -> bool:
        return True

    def prepare_inputs_for_generation(
        self, input_ids: torch.LongTensor, past_key_values: Optional[List[Tuple]] = None, **kwargs
    ):
        if past_key_values:
            # This is because we want the model to only process the last generated token.
            input_ids = input_ids[:, -1:]
        model_inputs = {"input_ids": input_ids, "past_key_values": past_key_values}

        model_inputs.update(kwargs)
        model_inputs["use_cache"] = kwargs.pop("use_cache", self.config.use_cache)
        return model_inputs

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.model.transformer.wte

    def set_input_embeddings(self, value: torch.nn.Module):
        self.model.transformer.wte = value

    def get_output_embeddings(self):
        if self.config.weight_tying:
            return self.model.transformer.wte
        else:
            return self.model.transformer.ff_out

    def set_output_embeddings(self, value: torch.nn.Module):
        if self.config.weight_tying:
            self.model.transformer.wte = value
        else:
            self.model.transformer.ff_out = value

    def tie_weights(self):
        if self.config.weight_tying:
            self.model.transformer.ff_out = self.model.transformer.wte

    def caching(self, enable: bool = True):
        self.model.caching(enable)

    def empty_cache(self):
        self.model.empty_cache()


# Register the model so that it is available for transformer pipelines, auto-loading, etc.
AutoModel.register(LLaDAConfig, LLaDAModelLM)
