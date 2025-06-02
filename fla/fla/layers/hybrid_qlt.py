# -*- coding: utf-8 -*-
# Original code from: https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/delta_net.py
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Modififed by:
# Copyright (c) 2025 Kazuki Irie

# NB:
# - Search for "Synchrous HQLT", "Delayed-Chunk HQLT" and
#   "Delayed-Stream HQLT" to find different model variations
# - In "Delayed-Chunk HQLT", context carry-over (i.e., passing KV and FW memory
#   from one batch to another) is not properly implemented, and we do not plan
#   to implement it as "Synchrous HQLT" is a better choice regardless.
# - docstrings are not thorougly checked; some of the arguments may be missing or outdated.

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input, unpad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution, RotaryEmbedding
from fla.ops.hybrid_qlt import (
    chunk_hybrid_softmax_delta_rule, pos_chunk_hybrid_softmax_delta_rule)
from fla.ops.delta_rule import chunk_delta_rule
from fla.ops.linear_attn import chunk_linear_attn

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    warnings.warn(
        "Flash Attention is not installed. Please install it via `pip install flash-attn --no-build-isolation`",
        category=ImportWarning
    )
    flash_attn_func = None


def elu_p1(x):
    return (F.elu(x, 1., False) + 1.).to(x)


def sum_norm(x):
    return (x / x.sum(-1, keepdim=True)).to(x)

def l2_norm(x):
    return (x / x.norm(-1, keepdim=True)).to(x)


class HybridQLT(nn.Module):
    r"""
    Hybrid Quadratic-Linear Memory
    Use DeltaNet as FW-Memory, and Softmax-Attention as KV-memory
    The layer implementaion for [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484).  # noqa:
    DeltaNet was originally proposed in [Linear Transformers Are Secretly Fast Weight Programmers](https://arxiv.org/abs/2102.11174). # noqa

    Args:
        mode (str, Optional):
            Which DeltaNet kernel to use.
            Currently available: `chunk`, `fused_recurrent`, and `fused_chunk`.
            Default: `chunk`.
        hidden_size (int, Optional):
            The hidden size of the input. Default: 1024.
        expand_k (float, Optional):
            The expansion ratio for the key dim. Default: 1.0.
        expand_v (float, Optional):
            The expansion ratio for the value dim. Default: 1.0.
        num_heads (int, Optional):
            The number of heads. Default: 4.
        use_beta (bool, Optional):
            Whether to use beta. Default: `True`.
        use_gate (bool, Optional):
            Whether to use output gate. Default: `False`.
        use_short_conv (bool, Optional):
            Whether to use short convolutions. Default: `True`.
        conv_size (int, Optional):
            The kernel size of the short convolution, only used when `use_short_conv` is `True`. Default: 4.
        conv_bias (bool, Optional):
            Whether to use bias in the short convolution, only used when `use_short_conv` is `True`. Default: `False`.
        allow_neg_eigval (bool, Optional):
            Allow negative eigenvalues. Default: `False`. If set to `True`, the beta will be multiplied by 2.
            See reference: [Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues](https://arxiv.org/abs/2411.12537)
        layer_idx (int, Optional):
            The index of the layer. Default: None.
        norm_eps (float, Optional):
            The epsilon value for the layernorm/rmsnorm layer. Default: 1e-5.
        qk_activation (str, Optional):
            The activation function for the query and key. Default: `silu`.
        qk_norm (str, Optional):
            The normalization method for the query and key. Default: `l2`.
    """

    def __init__(
        self,
        mode: str = 'sync_chunk',
        d_model: int = None,
        hidden_size: int = 1024,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        num_heads: int = 4,
        use_beta: bool = True,
        use_gate: bool = False,
        use_forgetting_gate: bool = False,
        use_local_pos_encoding: bool = False,
        use_memory_gate: bool = False,
        use_memory_scaler: bool = False,
        use_memory_dynamic_scaler: bool = False,
        use_memory_gate_separate: bool = False,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        allow_neg_eigval: bool = False,
        layer_idx: int = None,
        qk_activation: str = 'identity',  # SiLU is hard coded
        qk_norm: str = 'l2',
        norm_eps: float = 1e-5,
        chunk_size: int = 64,
        **kwargs
    ) -> HybridQLT:
        super().__init__()

        self.mode = mode
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm

        assert self.qk_activation == 'identity'  # we apply silu inside FW part
        assert self.qk_norm in ['l2', 'sum']

        if d_model is not None:
            hidden_size = d_model
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.allow_neg_eigval = allow_neg_eigval
        self.chunk_size = chunk_size

        self.use_forgetting_gate = use_forgetting_gate
        self.use_local_pos_encoding = use_local_pos_encoding
        self.use_memory_gate = use_memory_gate
        self.use_memory_scaler = use_memory_scaler
        self.use_memory_gate_separate = use_memory_gate_separate
        self.use_memory_dynamic_scaler = use_memory_dynamic_scaler
        self.max_position_embeddings = kwargs.get('max_position_embeddings', None)

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        self.layer_idx = layer_idx

        if mode == 'fused_chunk':
            raise NotImplementedError("fused_chunk_delta_rule is now deprecated. Please use `chunk_delta_rule` instead.")
        assert mode in ['chunk', 'fused_recurrent', 'sync_chunk', 'delayed', 'sync_chunk_linear'], f"Not suppoerted mode `{mode}`."
        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        self.dropout_proj = nn.Dropout(0.1)
        self.dropout_out = nn.Dropout(0.1)

        if self.use_local_pos_encoding:
            self.rope_theta = 10000.  # hard coded atm
            self.rotary = RotaryEmbedding(dim=self.head_k_dim, base=self.rope_theta)

        self.use_beta = use_beta
        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
        if use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                activation='silu' if qk_activation == 'silu' else None
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                activation='silu' if qk_activation == 'silu' else None
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                activation='silu'
            )
        if use_memory_gate_separate:
            assert use_memory_gate
        if use_memory_gate:
            assert not use_memory_scaler
            self.num_gates = 1 if not self.use_memory_gate_separate else 2
            self.memory_gate_proj = nn.Linear(
                hidden_size, self.value_dim * self.num_gates, bias=False)
        if use_memory_dynamic_scaler:
            self.memory_mixer = nn.Linear(
                hidden_size, 2 * self.num_heads, bias=False)            
        if use_memory_scaler:
            assert not use_memory_gate
            self.fw_scale = nn.Parameter(
                torch.tensor([1.], dtype=self.q_proj.weight.dtype),
                requires_grad=True)
            self.kv_scale = nn.Parameter(
                torch.tensor([1.], dtype=self.q_proj.weight.dtype),
                requires_grad=True)
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)

        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs: Unpack[Dict]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.shape
        mode = self.mode
        if self.training:
            assert self.mode in ['chunk', 'sync_chunk', 'delayed', 'sync_chunk_linear'], "Only chunk mode is supported in training."

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get('cu_seqlens', None)
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states),
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens
            )
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = F.silu(self.v_proj(hidden_states))

        q = self.dropout_proj(q)
        k = self.dropout_proj(k)
        v = self.dropout_proj(v)

        q, k = map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_k_dim), (q, k))
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)
        if self.qk_activation != 'silu':
            if self.qk_activation == 'relu':
                q, k = q.relu(), k.relu()
            elif self.qk_activation == 'elu':
                q, k = elu_p1(q), elu_p1(k)
            elif self.qk_activation != 'identity':
                raise NotImplementedError

        if self.qk_norm == 'sum':
            q = sum_norm(q).to(q)
            k = sum_norm(k).to(k)

        if self.use_beta:
            beta = self.b_proj(hidden_states).sigmoid()
        else:
            beta = torch.ones_like(q[..., 0])

        if self.allow_neg_eigval:
            beta = beta * 2.

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None

        if mode == 'sync_chunk':  # NB: naming is confusing here but this is "Synchrous HQLT" with DeltaNet
            # KV part
            if self.use_local_pos_encoding:
                # apply pos enc to kv part
                seqlen_offset, max_seqlen = 0, q_len
                if past_key_values is not None:
                    seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
                    max_seqlen = q.shape[1] + seqlen_offset

                    if attention_mask is not None:
                        # to deliminate the offsets of padding tokens
                        seqlen_offset = seqlen_offset + attention_mask.sum(-1) - attention_mask.shape[-1]
                        max_seqlen = q.shape[1] + max(seqlen_offset)

                if self.max_position_embeddings is not None:
                    max_seqlen = max(max_seqlen, self.max_position_embeddings)
                q_kv, k_kv = self.rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens)
            else:
                q_kv, k_kv = q, k
            v_kv = v

            if past_key_values is not None:
                cache_has_content = past_key_values.get_seq_length(self.layer_idx) > 0
                if self.training:
                    assert not cache_has_content, 'This case (carry over context during training) is not implemented.'
                if cache_has_content:
                    k_old, _ = past_key_values[self.layer_idx]['attn_state']
                    cache_len = k_old.shape[1]

                if not self.training:
                    k_cached, v_cached = past_key_values.update(
                        attn_state=[k_kv.flatten(-2, -1), v_kv.flatten(-2, -1)],
                        layer_idx=self.layer_idx,
                        offset=q_len,
                        cache_kwargs=dict(window_size=self.chunk_size),
                        use_special=True,  # added!
                    )['attn_state']

                if cache_has_content:
                    k_kv, v_kv = k_cached, v_cached
                    k_kv = rearrange(k_kv, '... (h d) -> ... h d', d=self.head_k_dim)
                    v_kv = rearrange(v_kv, '... (h d) -> ... h d', d=self.head_v_dim)
                    # add padding q
                    q_pad = torch.zeros_like(k_kv)[:, :cache_len]
                    q_kv = torch.cat([q_pad, q_kv], dim=1)
            else:
                cache_has_content = False

            if self.training:
                if attention_mask is not None:
                    q_kv, (k_kv, v_kv), indices_q, cu_seqlens, max_seq_lens = unpad_input(q_kv, (k_kv, v_kv), attention_mask, q_len)
                    cu_seqlens_q, cu_seqlens_k = cu_seqlens
                    max_seqlen_q, max_seqlen_k = max_seq_lens
                    o_kv = flash_attn_varlen_func(
                        q_kv, k_kv, v_kv,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_k=cu_seqlens_k,
                        max_seqlen_q=max_seqlen_q,
                        max_seqlen_k=max_seqlen_k,
                        causal=True,
                        window_size=(-1, -1) if self.chunk_size is None else (self.chunk_size-1, 0)
                    )
                    o_kv = pad_input(o_kv, indices_q, batch_size, q_len)
                elif cu_seqlens is not None:
                    o_kv = flash_attn_varlen_func(
                        q_kv.squeeze(0), k_kv.squeeze(0), v_kv.squeeze(0),
                        cu_seqlens_q=cu_seqlens,
                        cu_seqlens_k=cu_seqlens,
                        max_seqlen_q=max_seqlen,
                        max_seqlen_k=max_seqlen,
                        causal=True,
                        window_size=(-1, -1) if self.chunk_size is None else (self.chunk_size-1, 0)
                    ).unsqueeze(0)
                else:
                    o_kv = flash_attn_func(
                        q_kv, k_kv, v_kv,
                        causal=True,
                        window_size=(-1, -1) if self.chunk_size is None else (self.chunk_size-1, 0)
                    )
            else:
                o_kv = flash_attn_func(
                    q_kv, k_kv, v_kv,
                    causal=True,
                    window_size=(-1, -1) if self.chunk_size is None else (self.chunk_size-1, 0)
                )
                if cache_has_content:  # only take last q_len
                    o_kv = o_kv[:, -q_len:]

            # FW part
            q, k = F.silu(q), F.silu(k)
            o_fw, recurrent_state = chunk_delta_rule(
                q=q,
                k=k,
                v=v,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True if self.qk_norm == 'l2' else False
            )
        elif mode == 'sync_chunk_linear':  # Synchrous HQLT with Linear Attention (duplicated from above)
            # KV part
            if self.use_local_pos_encoding:
                # apply pos enc to kv part
                seqlen_offset, max_seqlen = 0, q_len
                if past_key_values is not None:
                    seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
                    max_seqlen = q.shape[1] + seqlen_offset

                    if attention_mask is not None:
                        # to deliminate the offsets of padding tokens
                        seqlen_offset = seqlen_offset + attention_mask.sum(-1) - attention_mask.shape[-1]
                        max_seqlen = q.shape[1] + max(seqlen_offset)

                if self.max_position_embeddings is not None:
                    max_seqlen = max(max_seqlen, self.max_position_embeddings)
                q_kv, k_kv = self.rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens)
            else:
                q_kv, k_kv = q, k
            v_kv = v

            if past_key_values is not None:
                cache_has_content = past_key_values.get_seq_length(self.layer_idx) > 0
                if self.training:
                    assert not cache_has_content, 'This case (carry over context during training) is not implemented.'
                if cache_has_content:
                    k_old, _ = past_key_values[self.layer_idx]['attn_state']
                    cache_len = k_old.shape[1]

                if not self.training:
                    k_cached, v_cached = past_key_values.update(
                        attn_state=[k_kv.flatten(-2, -1), v_kv.flatten(-2, -1)],
                        layer_idx=self.layer_idx,
                        offset=q_len,
                        cache_kwargs=dict(window_size=self.chunk_size),
                        use_special=True,  # added!
                    )['attn_state']

                if cache_has_content:
                    k_kv, v_kv = k_cached, v_cached
                    k_kv = rearrange(k_kv, '... (h d) -> ... h d', d=self.head_k_dim)
                    v_kv = rearrange(v_kv, '... (h d) -> ... h d', d=self.head_v_dim)
                    # add padding q
                    q_pad = torch.zeros_like(k_kv)[:, :cache_len]
                    q_kv = torch.cat([q_pad, q_kv], dim=1)
            else:
                cache_has_content = False

            if self.training:
                if attention_mask is not None:
                    q_kv, (k_kv, v_kv), indices_q, cu_seqlens, max_seq_lens = unpad_input(q_kv, (k_kv, v_kv), attention_mask, q_len)
                    cu_seqlens_q, cu_seqlens_k = cu_seqlens
                    max_seqlen_q, max_seqlen_k = max_seq_lens
                    o_kv = flash_attn_varlen_func(
                        q_kv, k_kv, v_kv,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_k=cu_seqlens_k,
                        max_seqlen_q=max_seqlen_q,
                        max_seqlen_k=max_seqlen_k,
                        causal=True,
                        window_size=(-1, -1) if self.chunk_size is None else (self.chunk_size-1, 0)
                    )
                    o_kv = pad_input(o_kv, indices_q, batch_size, q_len)
                elif cu_seqlens is not None:
                    o_kv = flash_attn_varlen_func(
                        q_kv.squeeze(0), k_kv.squeeze(0), v_kv.squeeze(0),
                        cu_seqlens_q=cu_seqlens,
                        cu_seqlens_k=cu_seqlens,
                        max_seqlen_q=max_seqlen,
                        max_seqlen_k=max_seqlen,
                        causal=True,
                        window_size=(-1, -1) if self.chunk_size is None else (self.chunk_size-1, 0)
                    ).unsqueeze(0)
                else:
                    o_kv = flash_attn_func(
                        q_kv, k_kv, v_kv,
                        causal=True,
                        window_size=(-1, -1) if self.chunk_size is None else (self.chunk_size-1, 0)
                    )
            else:
                o_kv = flash_attn_func(
                    q_kv, k_kv, v_kv,
                    causal=True,
                    window_size=(-1, -1) if self.chunk_size is None else (self.chunk_size-1, 0)
                )
                if cache_has_content:  # only take last q_len
                    o_kv = o_kv[:, -q_len:]

            # FW part
            q, k = elu_p1(q), elu_p1(k)
            o_fw, recurrent_state = chunk_linear_attn(
                q=q,
                k=k,
                v=v,
                normalize=True
            )
        elif mode == 'delayed':  # NB: naming is again confusing but this is "Delayed-Stream HQLT"
            # unlike the synchronous case, FW part will also make use of the
            # kv cache to determine the delayed kv context to be fed to FWP.
            # This requires changes to exluding the positional encoding in the cache
            # (and apply pos enc just before feeding it to KVA).
            # KV part
            cache_len = 0
            if past_key_values is not None:
                cache_has_content = past_key_values.get_seq_length(self.layer_idx) > 0
                seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
                if self.training:
                    assert not cache_has_content, 'This case (carry over context during training) is not implemented.'
                if cache_has_content:
                    k_old, _ = past_key_values[self.layer_idx]['attn_state']
                    cache_len = k_old.shape[1]

                if not self.training:
                    k_cached, v_cached = past_key_values.update(
                        attn_state=[k.flatten(-2, -1), v.flatten(-2, -1)],
                        layer_idx=self.layer_idx,
                        offset=q_len,
                        cache_kwargs=dict(window_size=self.chunk_size),
                        use_special=True,  # added!
                    )['attn_state']

                if cache_has_content:
                    k_kv, v_kv = k_cached, v_cached
                    k_kv = rearrange(k_kv, '... (h d) -> ... h d', d=self.head_k_dim)
                    v_kv = rearrange(v_kv, '... (h d) -> ... h d', d=self.head_v_dim)
                    q_padd_zeros = torch.zeros(
                        [batch_size, cache_len, self.num_heads, self.head_k_dim],
                        dtype=q.dtype, device=q.device)
                    q_kv = torch.cat([q_padd_zeros, q], dim=1)
                else:
                    q_kv, k_kv, v_kv = q.clone(), k.clone(), v.clone()
            else:
                cache_has_content = False
                v_kv = v
                q_kv = q
                k_kv = k

            if self.use_local_pos_encoding:
                # apply pos enc to kv part
                seqlen_offset, max_seqlen = 0, q_len
                if past_key_values is not None:
                    seqlen_offset -= cache_len
                    max_seqlen = q_kv.shape[1] + seqlen_offset

                    if attention_mask is not None:
                        # to deliminate the offsets of padding tokens
                        seqlen_offset = seqlen_offset + attention_mask.sum(-1) - attention_mask.shape[-1]
                        max_seqlen = q.shape[1] + max(seqlen_offset)
                else:
                    seqlen_offset, max_seqlen = 0, q_len
                if self.max_position_embeddings is not None:
                    max_seqlen = max(max_seqlen, self.max_position_embeddings)
                q_kv, k_kv = self.rotary(q_kv, k_kv, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens)
            else:
                q_kv, k_kv = q, k

            if self.training:
                if attention_mask is not None:
                    q_kv, (k_kv, v_kv), indices_q, cu_seqlens, max_seq_lens = unpad_input(q_kv, (k_kv, v_kv), attention_mask, q_len)
                    cu_seqlens_q, cu_seqlens_k = cu_seqlens
                    max_seqlen_q, max_seqlen_k = max_seq_lens
                    o_kv = flash_attn_varlen_func(
                        q_kv, k_kv, v_kv,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_k=cu_seqlens_k,
                        max_seqlen_q=max_seqlen_q,
                        max_seqlen_k=max_seqlen_k,
                        causal=True,
                        window_size=(-1, -1) if self.chunk_size is None else (self.chunk_size-1, 0)
                    )
                    o_kv = pad_input(o_kv, indices_q, batch_size, q_len)
                elif cu_seqlens is not None:
                    o_kv = flash_attn_varlen_func(
                        q_kv.squeeze(0), k_kv.squeeze(0), v_kv.squeeze(0),
                        cu_seqlens_q=cu_seqlens,
                        cu_seqlens_k=cu_seqlens,
                        max_seqlen_q=max_seqlen,
                        max_seqlen_k=max_seqlen,
                        causal=True,
                        window_size=(-1, -1) if self.chunk_size is None else (self.chunk_size-1, 0)
                    ).unsqueeze(0)
                else:
                    o_kv = flash_attn_func(
                        q_kv, k_kv, v_kv,
                        causal=True,
                        window_size=(-1, -1) if self.chunk_size is None else (self.chunk_size-1, 0)
                    )
            else:
                o_kv = flash_attn_func(
                    q_kv, k_kv, v_kv,
                    causal=True,
                    window_size=(-1, -1) if self.chunk_size is None else (self.chunk_size-1, 0)
                )
                if cache_has_content:  # only take last q_len
                    o_kv = o_kv[:, -q_len:]

            # FW part
            q = F.silu(q)
            # delay q and k
            window_size = self.chunk_size
            if past_key_values is not None and cache_has_content:
                k_cached = rearrange(k_cached, '... (h d) -> ... h d', d=self.head_k_dim)
                v_cached = rearrange(v_cached, '... (h d) -> ... h d', d=self.head_v_dim)
                if cache_len >= window_size:
                    # assert k.shape[1] >= window_size
                    k_fw = F.silu(k_cached[:, :q_len])
                    v_fw = v_cached[:, :q_len]
                else:
                    pad_len = window_size-cache_len
                    k_fw = torch.zeros(
                        [batch_size, pad_len, self.num_heads, self.head_k_dim],
                        dtype=k.dtype, device=k.device)
                    v_fw = torch.zeros(
                        [batch_size, pad_len, self.num_heads, self.head_v_dim],
                        dtype=v.dtype, device=v.device)
                    k_fw = torch.cat([k_fw, F.silu(k_cached)], dim=1)[:, :q_len]
                    v_fw = torch.cat([v_fw, v_cached], dim=1)[:, :q_len]
            else:
                # append k and v zeros of length chunk
                k_padd_zeros = torch.zeros(
                    [batch_size, window_size, self.num_heads, self.head_k_dim],
                    dtype=k.dtype, device=k.device)
                v_padd_zeros = torch.zeros(
                    [batch_size, window_size, self.num_heads, self.head_v_dim],
                    dtype=v.dtype, device=v.device)
                k_fw = torch.cat([k_padd_zeros, F.silu(k[:, :-window_size])], dim=1)[:, :q_len]
                v_fw = torch.cat([v_padd_zeros, v[:, :-window_size]], dim=1)[:, :q_len]

            o_fw, recurrent_state = chunk_delta_rule(
                q=q,
                k=k_fw,
                v=v_fw,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True if self.qk_norm == 'l2' else False
            )
        elif mode == 'chunk':  # NB: naming is again confusing but this is "Delayed-Chunk HQLT"
            if self.use_local_pos_encoding:
                # apply pos enc to kv part
                seqlen_offset, max_seqlen = 0, q_len
                if past_key_values is not None:
                    seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
                    max_seqlen = q.shape[1] + seqlen_offset

                    if attention_mask is not None:
                        # to deliminate the offsets of padding tokens
                        seqlen_offset = seqlen_offset + attention_mask.sum(-1) - attention_mask.shape[-1]
                        max_seqlen = q.shape[1] + max(seqlen_offset)

                if self.max_position_embeddings is not None:
                    max_seqlen = max(max_seqlen, self.max_position_embeddings)
                q_kv, k_kv = self.rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens)
                o_fw, o_kv, recurrent_state = pos_chunk_hybrid_softmax_delta_rule(
                    q=q,
                    k=k,
                    v=v,
                    q_kv=q_kv,
                    k_kv=k_kv,
                    beta=beta,
                    initial_state=recurrent_state,
                    output_final_state=use_cache,
                    chunk_size=self.chunk_size,
                    cu_seqlens=cu_seqlens,
                    use_qk_l2norm_in_kernel=True if self.qk_norm == 'l2' else False
                )
            else:
                o_fw, o_kv, recurrent_state = chunk_hybrid_softmax_delta_rule(
                    q=q,
                    k=k,
                    v=v,
                    beta=beta,
                    initial_state=recurrent_state,
                    output_final_state=use_cache,
                    chunk_size=self.chunk_size,
                    cu_seqlens=cu_seqlens,
                    use_qk_l2norm_in_kernel=True if self.qk_norm == 'l2' else False
                )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if self.use_memory_gate:
            mg = rearrange(self.memory_gate_proj(hidden_states), '... (h d) -> ... h d', d=self.head_v_dim * self.num_gates).sigmoid()
            if self.num_gates == 1:
                o = mg * o_fw + (1 - mg) * o_kv
            else:
                assert self.num_gates == 2
                mg_fw, mg_kv = torch.split(mg, (self.head_v_dim, self.head_v_dim), -1)
                o = mg_fw * o_fw + mg_kv * o_kv
        elif self.use_memory_dynamic_scaler:
            mg = self.memory_mixer(hidden_states).sigmoid()
            mg_fw, mg_kv = torch.split(mg, (self.num_heads, self.num_heads), -1)
            o = mg_fw.unsqueeze(-1) * o_fw + mg_kv.unsqueeze(-1) * o_kv
        elif self.use_memory_scaler:
            o = self.fw_scale * o_fw + self.kv_scale * o_kv
        else:
            o = o_fw + o_kv

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=0,
            )

        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)
        o = self.dropout_out(o)

        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values
