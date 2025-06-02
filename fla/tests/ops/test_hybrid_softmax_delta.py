# -*- coding: utf-8 -*-

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))   # ./DIR/test/ops/
project_root = os.path.abspath(os.path.join(current_dir, "../.."))  # ./DIR/
sys.path.append(project_root)

from typing import Optional

import pytest
import torch
import torch.nn.functional as F
from einops import repeat

from fla.ops.hybrid_qlt import chunk_hybrid_softmax_delta_rule
from fla.ops.utils.testing import COMPILER_MODE, assert_close
from fla.utils import device, device_platform
from fla.ops.delta_rule import chunk_delta_rule

if COMPILER_MODE:
    test_b_list = [1]
    test_t_list = [4096]
    test_t_varlen_list = test_t_list
    test_d_list = [64, 128, 256]
else:
    test_b_list = [2]
    test_t_list = [15, 63, 300, 512]
    test_t_varlen_list = [63, 286, 300, 512]
    test_d_list = [64, 32, 100, 256]
test_h_list = [2]


def naive_local_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    chunk_size=64,
):
    _, T, HQ, D = q.shape
    H = k.shape[2]
    G = HQ // H
    NT = T // chunk_size
    last_chunk_size = T % chunk_size
    if scale is None:  # dangerous!
        scale = D ** -0.5

    # loop over chunks
    ref_list = []
    mask = torch.tril(torch.ones((chunk_size, chunk_size), dtype=torch.bool, device=device))
    for j in range(0, NT):
        q_chunk = q[:, j*chunk_size:(j+1)*chunk_size, :, :].clone()
        k_chunk = k[:, j*chunk_size:(j+1)*chunk_size, :, :].clone()
        v_chunk = v[:, j*chunk_size:(j+1)*chunk_size, :, :].clone()
        ref = torch.einsum("bqhd,bkhd->bhqk", q_chunk.float() * scale, repeat(k_chunk, "b t h d -> b t (h g) d", g=G).float())
        ref = ref.masked_fill(~mask.unsqueeze(0).unsqueeze(0), -float('inf'))
        ref_list.append(torch.einsum("bhqk,bkhd->bqhd", F.softmax(ref, dim=-1), repeat(v_chunk, "b t h d -> b t (h g) d", g=G).float()))
    # last chunk
    if last_chunk_size != 0:
        mask = torch.tril(torch.ones((last_chunk_size, last_chunk_size), dtype=torch.bool, device=device))
        q_chunk = q[:, NT*chunk_size:T, :, :].clone()
        k_chunk = k[:, NT*chunk_size:T, :, :].clone()
        v_chunk = v[:, NT*chunk_size:T, :, :].clone()
        ref = torch.einsum("bqhd,bkhd->bhqk", q_chunk.float() * scale, repeat(k_chunk, "b t h d -> b t (h g) d", g=G).float())
        ref = ref.masked_fill(~mask.unsqueeze(0).unsqueeze(0), -float('inf'))
        ref_list.append(torch.einsum("bhqk,bkhd->bqhd", F.softmax(ref, dim=-1), repeat(v_chunk, "b t h d -> b t (h g) d", g=G).float()))

    ref = torch.cat(ref_list, dim=1)
    return ref


def naive_non_local_deltanet(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    scale: Optional[float] = None,
    chunk_size=64,
    output_final_state=True,
    initial_state = None,
    use_qk_l2norm_in_kernel = False
):
    _, T, _, D = q.shape
    NT = T // chunk_size
    last_chunk_size = T % chunk_size
    if scale is None:
        scale = D ** -0.5

    # loop over chunks
    ref_list = []
    # chunk_gated_delta_rule
    recurrent_state = initial_state
    
    for j in range(0, NT):
        print(f'Loop 0, Chunk {j}')
        q_chunk = q[:, j*chunk_size:(j+1)*chunk_size, :, :].clone()
        k_chunk = k[:, j*chunk_size:(j+1)*chunk_size, :, :].clone()
        v_chunk = v[:, j*chunk_size:(j+1)*chunk_size, :, :].clone()
        beta_chunk = beta[:, j*chunk_size:(j+1)*chunk_size, :].clone()

        # query times old fw
        ref = q_chunk.transpose(1, 2).contiguous().to(recurrent_state.dtype) @ recurrent_state
        ref_list.append(ref)

        # get new fw
        _, recurrent_state = chunk_delta_rule(
            q=q_chunk,
            k=k_chunk,
            v=v_chunk,
            beta=beta_chunk,
            scale=scale,
            initial_state=recurrent_state.clone(),
            output_final_state=output_final_state,
            cu_seqlens=None,
            head_first=False,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel
        )

    # last chunk
    if last_chunk_size != 0:
        print(f'Last Chunk')
        q_chunk = q[:, NT*chunk_size:T, :, :].clone()
        k_chunk = k[:, NT*chunk_size:T, :, :].clone()
        v_chunk = v[:, NT*chunk_size:T, :, :].clone()
        beta_chunk = beta[:, NT*chunk_size:T, :].clone()

        # query times old fw
        ref = q_chunk.transpose(1, 2).contiguous().to(recurrent_state.dtype) @ recurrent_state
        ref_list.append(ref)

        # get new fw
        _, recurrent_state = chunk_delta_rule(
            q=q_chunk,
            k=k_chunk,
            v=v_chunk,
            beta=beta_chunk,
            scale=scale,
            initial_state=recurrent_state.clone(),
            output_final_state=output_final_state,
            cu_seqlens=None,
            head_first=False,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel
        )
    ref = torch.cat(ref_list, dim=2) * scale
    return ref, recurrent_state


def get_final_state_deltanet(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    scale: Optional[float] = None,
    chunk_size=64,
    output_final_state=True,
    initial_state = None,
    use_qk_l2norm_in_kernel = False
):
    _, _, _, D = q.shape
    if scale is None:
        scale = D ** -0.5

    recurrent_state = initial_state
    
    _, recurrent_state = chunk_delta_rule(
        q=q,
        k=k,
        v=v,
        beta=beta,
        scale=scale,
        initial_state=recurrent_state.clone(),
        output_final_state=output_final_state,
        cu_seqlens=None,
        head_first=False,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel
    )

    return recurrent_state


@pytest.mark.parametrize('B', test_b_list)
@pytest.mark.parametrize('T', test_t_list)
@pytest.mark.parametrize('H', test_h_list)
@pytest.mark.parametrize('D', test_d_list)
@pytest.mark.parametrize('scale', [1])
@pytest.mark.parametrize('dtype', [torch.bfloat16])
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '0',
    reason='Skipping test because TEST_CHUNK_VARLEN is enabled'
)
@pytest.mark.skipif(
    device_platform == 'intel',
    reason='Intel Triton Failure'
)
def test_chunk(
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
    scale: float,
):
    torch.manual_seed(42)
    q = torch.randn(B, T, H, D, dtype=dtype)
    k = F.normalize(torch.randn(B, T, H, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
    v = torch.randn(B, T, H, D, dtype=dtype)
    beta = torch.rand(B, T, H, dtype=dtype).sigmoid()
    h0 = torch.randn(B, H, D, D, dtype=torch.float32)
    q, k, v, beta, h0 = map(lambda x: x.to(device).requires_grad_(True), (q, k, v, beta, h0))
    do = torch.rand_like(v)
    dht = torch.rand_like(h0)

    assert scale is not None

    chunk_size = 64

    print('Forward custom kernel')
    tri_fw, tri_kv, tri_ht = chunk_hybrid_softmax_delta_rule(
        q.clone(),
        k.clone(),
        v.clone(),
        beta.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
        chunk_size=chunk_size,
        use_qk_l2norm_in_kernel=False,
    )
    tri = tri_fw + tri_kv
    print('Backward custom kernel')
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dh0 = q.grad, k.grad, v.grad, beta.grad, h0.grad
    q.grad = k.grad = v.grad = beta.grad = h0.grad = None

######################################

    print('Forward reference')
    ref_fw, ref_ht = naive_non_local_deltanet(
        q.clone(),
        k.clone(),
        v.clone(),
        beta.clone(),
        None,
        scale=scale,
        chunk_size=chunk_size,
        output_final_state=True,
        initial_state=h0.clone(),
        use_qk_l2norm_in_kernel=False
    )

    ref_kv = naive_local_attn(
        q.clone(),
        k.clone(),
        v.clone(),
        scale=scale,
        chunk_size=chunk_size,
    )
    ref_fw = ref_fw.transpose(1, 2).contiguous()
    ref = ref_fw + ref_kv
    # ref = ref_fw
    # ref = ref_kv

    ## Useful for debugging FWM
    # print('Forward reference just state')
    # ref2_state = get_final_state_deltanet(
    #     q.clone(),
    #     k.clone(),
    #     v.clone(),
    #     beta.clone(),
    #     None,
    #     scale=scale,
    #     chunk_size=chunk_size,
    #     output_final_state=True,
    #     initial_state=h0.clone(),
    #     use_qk_l2norm_in_kernel=False
    # )

    print('Backward reference')
    # ((ref * do).sum()).backward(retain_graph=True)
    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dbeta, ref_dh0 = q.grad, k.grad, v.grad, beta.grad, h0.grad

######################################

    # manual dq
    # dq_manual = scale * (do.detach().transpose(1, 2).to(h0.dtype) @ h0.detach().transpose(2, 3)).transpose(1, 2)
    # # assert_close(' dq ref1/ref2', ref_dq, dq_manual, 0.008)
    # assert_close(' dq tri/ref2', tri_dq, dq_manual, 0.008)

    # print(f"H0 norm outside: {h0.detach().flatten().square().mean().sqrt().item()}")
    # print(f"do norm outside: {do.detach().flatten().square().mean().sqrt().item()}")
    # print(f"dq norm outside: {dq_manual.detach().flatten().square().mean().sqrt().item()}")
    # print(f"dq shape outside: {dq_manual.shape}")
    # print(f"do shape outside: {do.shape}")

    # print(f"dq look: {dq_manual[0]}")

    assert_close('  o', ref, tri, 0.006)
    assert_close('ofw', ref_fw, tri_fw, 0.006)
    assert_close('okv', ref_kv, tri_kv, 0.006)
    assert_close(' ht', ref_ht, tri_ht, 0.006)
    assert_close(' dq', ref_dq, tri_dq, 0.008)
    assert_close(' dk', ref_dk, tri_dk, 0.008)
    assert_close(' dv', ref_dv, tri_dv, 0.008)
    assert_close(' db', ref_dbeta, tri_dbeta, 0.008)
    assert_close('dh0', ref_dh0, tri_dh0, 0.008)


@pytest.mark.parametrize('B', test_b_list)
@pytest.mark.parametrize('T', test_t_list)
@pytest.mark.parametrize('H', test_h_list)
@pytest.mark.parametrize('D', test_d_list)
@pytest.mark.parametrize('scale', [1])
@pytest.mark.parametrize('dtype', [torch.bfloat16])
@pytest.mark.skipif(
    os.getenv('SKIP_TEST_CHUNK_VARLEN') == '0',
    reason='Skipping test because TEST_CHUNK_VARLEN is enabled'
)
@pytest.mark.skipif(
    device_platform == 'intel',
    reason='Intel Triton Failure'
)
def test_fused_silu_l2_in_kernel(
    B: int,
    T: int,
    H: int,
    D: int,
    dtype: torch.dtype,
    scale: float,
):
    torch.manual_seed(42)
    q = torch.randn(B, T, H, D, dtype=dtype)
    k = torch.randn(B, T, H, D, dtype=dtype)
    v = torch.randn(B, T, H, D, dtype=dtype)
    beta = torch.rand(B, T, H, dtype=dtype).sigmoid()
    h0 = torch.randn(B, H, D, D, dtype=torch.float32)
    q, k, v, beta, h0 = map(lambda x: x.to(device).requires_grad_(True), (q, k, v, beta, h0))
    do = torch.rand_like(v)
    dht = torch.rand_like(h0)

    assert scale is not None

    chunk_size = 64

    print('Forward custom kernel')
    tri_fw, tri_kv, tri_ht = chunk_hybrid_softmax_delta_rule(
        q.clone(),
        k.clone(),
        v.clone(),
        beta.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
        chunk_size=chunk_size,
        use_qk_l2norm_in_kernel=True
    )
    tri = tri_fw + tri_kv
    print('Backward custom kernel')
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dh0 = q.grad, k.grad, v.grad, beta.grad, h0.grad
    q.grad = k.grad = v.grad = beta.grad = h0.grad = None

    # print('Forward Ref kernel')
    # ref_fw, ref_kv, ref_ht = chunk_hybrid_softmax_delta_rule(
    #     F.normalize(q.clone(), p=2, dim=-1).to(dtype),
    #     F.normalize(k.clone(), p=2, dim=-1).to(dtype),
    #     v.clone(),
    #     beta.clone(),
    #     scale=scale,
    #     initial_state=h0.clone(),
    #     output_final_state=True,
    #     chunk_size=chunk_size,
    #     use_qk_l2norm_in_kernel=False
    # )
    # ref = ref_fw + ref_kv
    # print('Backward Ref kernel')
    # ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    # ref_dq, ref_dk, ref_dv, ref_dbeta, ref_dh0 = q.grad, k.grad, v.grad, beta.grad, h0.grad
    # q.grad = k.grad = v.grad = beta.grad = h0.grad = None

    print('Forward reference')
    ref_fw, ref_ht = naive_non_local_deltanet(
        F.normalize(F.silu(q.clone()), p=2, dim=-1).to(dtype),
        F.normalize(F.silu(k.clone()), p=2, dim=-1).to(dtype),
        v.clone(),
        beta.clone(),
        None,
        scale=scale,
        chunk_size=chunk_size,
        output_final_state=True,
        initial_state=h0.clone(),
        use_qk_l2norm_in_kernel=False
    )

    ref_kv = naive_local_attn(
        q.clone(),
        k.clone(),
        v.clone(),
        scale=scale,
        chunk_size=chunk_size,
    )
    ref_fw = ref_fw.transpose(1, 2).contiguous()
    ref = ref_fw + ref_kv

    print('Backward reference')
    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dbeta, ref_dh0 = q.grad, k.grad, v.grad, beta.grad, h0.grad

    assert_close('  o', ref, tri, 0.006)
    assert_close('ofw', ref_fw, tri_fw, 0.006)
    assert_close('okv', ref_kv, tri_kv, 0.006)
    assert_close(' ht', ref_ht, tri_ht, 0.006)
    assert_close(' dq', ref_dq, tri_dq, 0.008)
    assert_close(' dk', ref_dk, tri_dk, 0.008)
    assert_close(' dv', ref_dv, tri_dv, 0.008)
    assert_close(' db', ref_dbeta, tri_dbeta, 0.008)
    assert_close('dh0', ref_dh0, tri_dh0, 0.008)


if __name__ == '__main__':
    torch.manual_seed(111)
    print('Test ')
    # B, T, H, D = 3, 80, 7, 23
    B, T, H, D = 3, 80, 7, 79
    # scale = D ** -.5
    scale = 0.5
    print('*** test_chunk ***')
    test_chunk(B, T, H, D, torch.bfloat16, scale)
    print('*** test_l2_in_kernel ***')
    test_fused_silu_l2_in_kernel(B, T, H, D, torch.bfloat16, scale)
    print('*** End ***')
