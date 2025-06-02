# Install the newest triton version with
# pip install "git+https://github.com/openai/triton.git#egg=triton&subdirectory=python"

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))   # ./DIR/test/ops/
project_root = os.path.abspath(os.path.join(current_dir, "../.."))  # ./DIR/
sys.path.append(project_root)

import torch
from benchmark import benchmark_backward, benchmark_combined, benchmark_forward
from torch.nn import functional as F

from fla.ops.hybrid_qlt import chunk_hybrid_softmax_delta_rule
from fla.utils import device


def time_fwd(func, *args, **kwargs):
    time_fb = benchmark_forward(func, *args, **kwargs)
    return time_fb[1].mean


def time_fwd_bwd(func, *args, **kwargs):
    time_fb = benchmark_combined(func, *args, **kwargs)
    return time_fb[1].mean


def time_bwd(func, *args, **kwargs):
    time_fb = benchmark_backward(func, *args, **kwargs)
    return time_fb[1].mean


repeats = 256
dtype = torch.bfloat16


bs_seqlen_vals = [(8, 2048), (4, 4096), (2, 8192)]
causal_vals = [True]
headdim_vals = [64, 128, 256]
dim = 2048
dropout_p = 0.0


methods = (["chunk_hybrid_softmax_delta_rule"])
time_f = {}
time_b = {}
time_f_b = {}
speed_f = {}
speed_b = {}
speed_f_b = {}
for causal in causal_vals:
    for headdim in headdim_vals:
        for B, seqlen in bs_seqlen_vals:
            config = (causal, headdim, B, seqlen)
            H = dim // headdim
            q = torch.randn(B, H, seqlen, headdim, device=device, requires_grad=True, dtype=dtype)
            k = F.normalize(torch.randn(B, H, seqlen, headdim, device=device, dtype=dtype), p=2, dim=-1).requires_grad_(True)
            v = torch.randn(B, H, seqlen, headdim, device=device, requires_grad=True, dtype=dtype)
            beta = torch.rand(B, H, seqlen, device=device, dtype=dtype).sigmoid().requires_grad_(True)
            o_fw, o_kv, _ = chunk_hybrid_softmax_delta_rule(q, k, v, beta)
            o1 = o_fw + o_kv
            o1.sum().backward(retain_graph=True)
            f_b = time_fwd_bwd(
                chunk_hybrid_softmax_delta_rule, q, k, v, beta, verbose=False
            )
            time_f_b[config, "chunk_hybrid_softmax_delta_rule"] = f_b

            print(f"### causal={causal}, headdim={headdim}, B={B}, seqlen={seqlen} ###")
            for method in methods:
                print(f"{method:>50} fwd + bwd:\t {time_f_b[config, method]*1000:>6.4f} ms ")
