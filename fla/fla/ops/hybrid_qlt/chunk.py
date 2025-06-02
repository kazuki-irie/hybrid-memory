# -*- coding: utf-8 -*-
# Original code from: https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Modififed by:
# Copyright (c) 2025 Kazuki Irie
# NB: 
# - docstrings are not thorougly checked; some of the arguments may be missing or outdated.
# - Parts using forgetting gates in deltanet or attention are not tested; ignore them

import warnings
from typing import Optional

import torch
from einops import rearrange

from fla.modules.l2norm import fused_silu_l2norm_fwd, fused_silu_l2norm_bwd
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu, chunk_gated_delta_rule_fwd_h
from fla.ops.common.chunk_o import (
    chunk_softmax_bwd_dqkwg, chunk_no_local_fwd_o, chunk_softmax_bwd_dqkwg)
from fla.ops.delta_rule.wy_fast import prepare_wy_repr_fwd, recompute_w_u_fwd, prepare_wy_repr_non_local_bwd
from fla.ops.attn.parallel import parallel_chunk_only_attn_fwd, parallel_chunk_only_attn_bwd
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


# version 1: separately call KW and FW kernels, and sum output
def chunk_hybrid_softmax_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    chunk_size: int = 64,
    cu_seqlens: Optional[torch.LongTensor] = None,
    use_qk_l2norm_in_kernel: Optional[bool] = True,
):
    k_kv = k.clone()
    q_kv = q.clone()
    if use_qk_l2norm_in_kernel:
        k = fused_silu_l2norm_fwd(k)
        q = fused_silu_l2norm_fwd(q)
    # obtain WY representation. u is actually the new v.
    w, u, A = prepare_wy_repr_fwd(
        k=k,
        v=v,
        beta=beta,
        cu_seqlens=cu_seqlens,
    )
    # no need for v_new
    h, _, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=None,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens
    )
    o_fw = chunk_no_local_fwd_o(
        q=q,
        v=v,  # v instead of v_new; anyway not used
        h=h,
        g=None,
        scale=scale,
        cu_seqlens=cu_seqlens
    )
    o_kv, lse = parallel_chunk_only_attn_fwd(
        q=q_kv,
        k=k_kv,
        v=v,
        g_cumsum=None,  # gating related code is not checked! do not use it
        scale=scale,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
    )
    return o_fw, A, final_state, lse, o_kv


# version 2: separately call KW and FW kernels, and sum output
def chunk_hybrid_softmax_delta_rule_fwd_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_kv: torch.Tensor,
    k_kv: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    chunk_size: int = 64,
    cu_seqlens: Optional[torch.LongTensor] = None,
    use_qk_l2norm_in_kernel: Optional[bool] = True,
):
    if use_qk_l2norm_in_kernel:
        k = fused_silu_l2norm_fwd(k)
        q = fused_silu_l2norm_fwd(q)
    # obtain WY representation. u is actually the new v.
    w, u, A = prepare_wy_repr_fwd(
        k=k,
        v=v,
        beta=beta,
        cu_seqlens=cu_seqlens,
    )
    # no need for v_new
    h, _, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=None,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens
    )
    o_fw = chunk_no_local_fwd_o(
        q=q,
        v=v,  # v instead of v_new; anyway not used
        h=h,
        g=None,
        scale=scale,
        cu_seqlens=cu_seqlens
    )
    o_kv, lse = parallel_chunk_only_attn_fwd(
        q=q_kv,
        k=k_kv,
        v=v,
        g_cumsum=None,  # Not implemented
        scale=scale,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
    )
    return o_fw, A, final_state, lse, o_kv


def chunk_hybrid_softmax_delta_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    lse: torch.Tensor,
    okv: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dokv: torch.Tensor,   # remove me
    dht: torch.Tensor,
    chunk_size: int = 64,
    cu_seqlens: Optional[torch.LongTensor] = None,
    use_qk_l2norm_in_kernel: Optional[bool] = True,
):
    k_kv = k.clone()
    q_kv = q.clone()
    if use_qk_l2norm_in_kernel:
        k = fused_silu_l2norm_fwd(k)
        q = fused_silu_l2norm_fwd(q)
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        cu_seqlens=cu_seqlens
    )
    # recompute h and v_new
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=None,
        initial_state=initial_state,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
    )
    # Compute all local/intra-chunk attention gradients
    dq_kv, dk_kv, dv, _ = parallel_chunk_only_attn_bwd(
        q=q_kv,
        k=k_kv,
        v=v,
        o=okv,
        g_cumsum=None,
        lse=lse,
        # do=do,
        do=dokv,
        scale=scale,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
    )
    dv_zeros = torch.zeros_like(dv)
    dh, dh0, du = chunk_gated_delta_rule_bwd_dhu(
        q=q,
        k=k,
        w=w,
        g=None,
        h0=initial_state,
        dht=dht,
        do=do,
        dv=dv_zeros,
        scale=scale,
        cu_seqlens=cu_seqlens
    )
    # inter chunk FWM
    dq, dk, dw, _ = chunk_softmax_bwd_dqkwg(
        q=q,
        k=k,
        v=v_new,  # v_new needed for FW part
        h=h,
        w=w,
        dv=du,  # instead of dv; only the du part is relevant
        do=do,
        dh=dh,
        g=None,
        scale=scale,
        cu_seqlens=cu_seqlens
    )
    # no change needed below, except du
    dk2, dv2, db = prepare_wy_repr_non_local_bwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        dw=dw,
        du=du,  # changed dv to du
        cu_seqlens=cu_seqlens
    )
    dv.add_(dv2)
    dk.add_(dk2)

    if use_qk_l2norm_in_kernel:
        dk = fused_silu_l2norm_bwd(k_kv, dk)
        dq = fused_silu_l2norm_bwd(q_kv, dq)
 
    dk.add_(dk_kv)
    dq.add_(dq_kv)

    return dq, dk, dv, db, dh0


def chunk_hybrid_softmax_delta_rule_bwd_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_kv: torch.Tensor,
    k_kv: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    lse: torch.Tensor,
    okv: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dokv: torch.Tensor,   # remove me
    dht: torch.Tensor,
    chunk_size: int = 64,
    cu_seqlens: Optional[torch.LongTensor] = None,
    use_qk_l2norm_in_kernel: Optional[bool] = True,
):
    q_orig = q.clone()
    k_orig = k.clone()
    if use_qk_l2norm_in_kernel:
        k = fused_silu_l2norm_fwd(k)
        q = fused_silu_l2norm_fwd(q)
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        cu_seqlens=cu_seqlens
    )
    # recompute h and v_new
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=None,
        initial_state=initial_state,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
    )
    # Compute all local/intra-chunk attention gradients
    dq_kv, dk_kv, dv, _ = parallel_chunk_only_attn_bwd(
        q=q_kv,
        k=k_kv,
        v=v,
        o=okv,
        g_cumsum=None,
        lse=lse,
        # do=do,
        do=dokv,
        scale=scale,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
    )
    dv_zeros = torch.zeros_like(dv)
    dh, dh0, du = chunk_gated_delta_rule_bwd_dhu(
        q=q,
        k=k,
        w=w,
        g=None,
        h0=initial_state,
        dht=dht,
        do=do,
        dv=dv_zeros,
        scale=scale,
        cu_seqlens=cu_seqlens
    )
    # inter chunk FWM
    dq, dk, dw, _ = chunk_softmax_bwd_dqkwg(
        q=q,
        k=k,
        v=v_new,  # v_new needed for FW part
        h=h,
        w=w,
        dv=du,  # instead of dv; only the du part is relevant
        do=do,
        dh=dh,
        g=None,
        scale=scale,
        cu_seqlens=cu_seqlens
    )
    # no change needed below, except du
    dk2, dv2, db = prepare_wy_repr_non_local_bwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        dw=dw,
        du=du,  # change dv to du
        cu_seqlens=cu_seqlens
    )
    dv.add_(dv2)  # added
    dk.add_(dk2)

    if use_qk_l2norm_in_kernel:
        dk = fused_silu_l2norm_bwd(k_orig, dk)
        dq = fused_silu_l2norm_bwd(q_orig, dq)

    return dq, dk, dv, dq_kv, dk_kv, db, dh0


class ChunkHybridSoftmaxDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        chunk_size: int = 64,
        cu_seqlens: Optional[torch.LongTensor] = None,
        use_qk_l2norm_in_kernel: bool = True
    ):
        q_orig = q
        k_orig = k

        o_fw, A, final_state, lse, o_kv = chunk_hybrid_softmax_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        ctx.save_for_backward(
            q_orig, k_orig, v, beta, A, initial_state, lse, o_kv)
        ctx.scale = scale
        ctx.chunk_size = chunk_size
        ctx.cu_seqlens = cu_seqlens
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        return o_fw.to(q.dtype), o_kv.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        do: torch.Tensor,
        do_kv: torch.Tensor,
        dht: torch.Tensor
    ):
        q, k, v, beta, A, initial_state, lse, o_kv = ctx.saved_tensors
        use_qk_l2norm_in_kernel = ctx.use_qk_l2norm_in_kernel

        dq, dk, dv, db, dh0 = chunk_hybrid_softmax_delta_rule_bwd(
            q=q,
            k=k,
            v=v,
            beta=beta,
            A=A,
            lse=lse,
            okv=o_kv,
            scale=ctx.scale,
            initial_state=initial_state,
            do=do,
            dokv=do_kv,
            dht=dht,
            chunk_size=ctx.chunk_size,
            cu_seqlens=ctx.cu_seqlens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel
        )
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), db.to(beta.dtype), None, dh0, None, None, None, None, None


class ChunkHybridSoftmaxDeltaRuleFunctionV2(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_kv: torch.Tensor,
        k_kv: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        chunk_size: int = 64,
        cu_seqlens: Optional[torch.LongTensor] = None,
        use_qk_l2norm_in_kernel: bool = True
    ):
        q_orig = q
        k_orig = k

        o_fw, A, final_state, lse, o_kv = chunk_hybrid_softmax_delta_rule_fwd_v2(
            q=q,
            k=k,
            v=v,
            q_kv=q_kv,
            k_kv=k_kv,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        ctx.save_for_backward(
            q_orig, k_orig, v, q_kv, k_kv, beta, A, initial_state, lse, o_kv)
        ctx.scale = scale
        ctx.chunk_size = chunk_size
        ctx.cu_seqlens = cu_seqlens
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        return o_fw.to(q.dtype), o_kv.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        do: torch.Tensor,
        do_kv: torch.Tensor,
        dht: torch.Tensor
    ):
        q, k, v, q_kv, k_kv, beta, A, initial_state, lse, o_kv = ctx.saved_tensors
        use_qk_l2norm_in_kernel = ctx.use_qk_l2norm_in_kernel

        dq, dk, dv, dq_kv, dk_kv, db, dh0 = chunk_hybrid_softmax_delta_rule_bwd_v2(
            q=q,
            k=k,
            v=v,
            q_kv=q_kv,
            k_kv=k_kv,
            beta=beta,
            A=A,
            lse=lse,
            okv=o_kv,
            scale=ctx.scale,
            initial_state=initial_state,
            do=do,
            dokv=do_kv,
            dht=dht,
            chunk_size=ctx.chunk_size,
            cu_seqlens=ctx.cu_seqlens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel
        )
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), dq_kv.to(q.dtype), dk_kv.to(k.dtype), db.to(beta.dtype), None, dh0, None, None, None, None, None


@torch.compiler.disable
def chunk_hybrid_softmax_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False
):
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        beta (torch.Tensor):
            betas of shape `[B, T, H]` if `head_first=False` else `[B, H, T]`.
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format, which is not supported for variable-length inputs.
            Default: `False`.
        use_qk_l2norm_in_kernel (Optional[bool]):
            Whether to use qk l2norm within the kernel for saving GPU memory.
            Default: `False`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.delta_rule import chunk_delta_rule
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> h0 = torch.randn(B, H, K, V, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = chunk_delta_rule(
            q, k, v, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = chunk_delta_rule(
            q, k, v, beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype != torch.float32, "ChunkHybridSoftmaxDeltaRuleFunction does not support float32. Please use bfloat16."
    assert len(beta.shape) == 3, "beta must be of shape (batch size, num of head, seq len)."

    if head_first:
        raise DeprecationWarning(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead."
        )
        q, k, v, beta = map(lambda x: rearrange(x, 'b h t ... -> b t h ...'), (q, k, v, beta))
    if not head_first and q.shape[1] < q.shape[2]:
        warnings.warn(
            f"Input tensor shape suggests potential format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
            "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
            "when head_first=False was specified. "
            "Please verify your input tensor format matches the expected shape [B, T, H, ...]."
        )
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    scale = k.shape[-1] ** -0.5 if scale is None else scale
    o_fw, o_kv, final_state = ChunkHybridSoftmaxDeltaRuleFunction.apply(
        q,
        k,
        v,
        beta,
        scale,
        initial_state,
        output_final_state,
        chunk_size,
        cu_seqlens,
        use_qk_l2norm_in_kernel,
    )
    if head_first:
        o_fw = rearrange(o_fw, 'b t h v -> b h t v')
        o_kv = rearrange(o_kv, 'b t h v -> b h t v')
    return o_fw, o_kv, final_state


# Use rotary embedding outside; inefficient as keeping a separate copy
@torch.compiler.disable
def pos_chunk_hybrid_softmax_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_kv: torch.Tensor,
    k_kv: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False
):
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        beta (torch.Tensor):
            betas of shape `[B, T, H]` if `head_first=False` else `[B, H, T]`.
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format, which is not supported for variable-length inputs.
            Default: `False`.
        use_qk_l2norm_in_kernel (Optional[bool]):
            Whether to use qk l2norm within the kernel for saving GPU memory.
            Default: `False`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.delta_rule import chunk_delta_rule
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> h0 = torch.randn(B, H, K, V, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = chunk_delta_rule(
            q, k, v, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = chunk_delta_rule(
            q, k, v, beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype != torch.float32, "ChunkHybridSoftmaxDeltaRuleFunction does not support float32. Please use bfloat16."
    assert len(beta.shape) == 3, "beta must be of shape (batch size, num of head, seq len)."

    if head_first:
        raise DeprecationWarning(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead."
        )
        q, k, v, beta = map(lambda x: rearrange(x, 'b h t ... -> b t h ...'), (q, k, v, beta))
    if not head_first and q.shape[1] < q.shape[2]:
        warnings.warn(
            f"Input tensor shape suggests potential format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
            "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
            "when head_first=False was specified. "
            "Please verify your input tensor format matches the expected shape [B, T, H, ...]."
        )
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    scale = k.shape[-1] ** -0.5 if scale is None else scale
    o_fw, o_kv, final_state = ChunkHybridSoftmaxDeltaRuleFunctionV2.apply(
        q,
        k,
        v,
        q_kv,
        k_kv,
        beta,
        scale,
        initial_state,
        output_final_state,
        chunk_size,
        cu_seqlens,
        use_qk_l2norm_in_kernel,
    )
    if head_first:
        o_fw = rearrange(o_fw, 'b t h v -> b h t v')
        o_kv = rearrange(o_kv, 'b t h v -> b h t v')
    return o_fw, o_kv, final_state
