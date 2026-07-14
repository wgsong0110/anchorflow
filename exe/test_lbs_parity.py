#!/usr/bin/env python
"""GPU parity test: fused CUDA LBS blend vs the torch reference (fwd + bwd grad).
Run on the instance after the CUDA ext is built:  python exe/test_lbs_parity.py"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

import torch
from lbs import lbs_blend, _lbs_blend_torch, _HAVE_CUDA


def main():
    assert torch.cuda.is_available(), "needs GPU"
    print(f"CUDA ext built: {_HAVE_CUDA}")
    dev = "cuda"
    torch.manual_seed(0)
    N, M, K = 200000, 512, 4
    x = torch.randn(N, 3, device=dev)
    idx = torch.randint(0, M, (N, K), device=dev)
    w = torch.rand(N, K, device=dev); w = w / w.sum(-1, keepdim=True)
    a_rest = torch.randn(M, 3, device=dev)
    R = torch.eye(3, device=dev).expand(M, 3, 3).contiguous()

    a1 = torch.randn(M, 3, device=dev, requires_grad=True)
    a2 = a1.detach().clone().requires_grad_(True)
    out_cuda = lbs_blend(x, w, idx, a_rest, a1, R)
    out_ref = _lbs_blend_torch(x, w, idx, a_rest, a2, R)
    fwd_err = (out_cuda - out_ref).abs().max().item()
    (out_cuda.sum()).backward(); (out_ref.sum()).backward()
    bwd_err = (a1.grad - a2.grad).abs().max().item()
    print(f"forward max_err={fwd_err:.3e}  backward(grad a_now) max_err={bwd_err:.3e}")
    ok = fwd_err < 1e-4 and bwd_err < 1e-4
    print("PARITY", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
