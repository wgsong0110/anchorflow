"""상대위치 어텐션이 (1) 주장한 대로 정확한지, (2) 실제로 싼지 확인한다.

정확성  거리 게이트는 "근사"가 아니라 항등식이라고 주장했다. softmax 가 행 상수를
        지우므로 -(p_i-p_j)^T M (p_i-p_j) 와 q/k 에 붙인 4 채널의 내적은 **같은
        어텐션 가중치**를 낳아야 한다. 명시적으로 [M,M] 을 만들어 비교한다.
불변성  평행이동시켜도 출력이 같아야 하고(절대 좌표가 새면 여기서 걸린다),
        회전시키면 달라져야 한다(방향을 못 보면 0 이 나온다).
비용    옛 경로(쌍마다 MLP -> [M,M,H] 바이어스)와 새 경로를 같은 조건에서 잰다.
        M 을 훑는 것이 중요하다 -- 옛 경로는 [M,M,32] 중간 텐서를 만들므로 M 이
        커질수록 불리해지고, 작은 M 에서는 그 GEMM 이 오히려 효율적이다.
        바이어스는 옛 경로에서도 층 전체가 한 번만 만들게 해 조건을 맞춘다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")
sys.path.insert(0, _lib)

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--M", type=int, nargs="+", default=[512, 1024, 2048])
ap.add_argument("--hidden", type=int, default=128)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--depth", type=int, default=4)
ap.add_argument("--reps", type=int, default=30)
ap.add_argument("--warmup", type=int, default=10)
ap.add_argument("--out", default=None)
a = ap.parse_args()

dev = "cuda"
torch.manual_seed(0)
from anchorflow.deform import (RelAttention, RelBlock, RelPos,   # noqa: E402
                               _rope)
from anchorflow.nextstate import (GeoAttentionBias,              # noqa: E402
                                  GeoAttentionBlock)

H, C = a.heads, a.hidden
D = C // H

# ---------------------------------------------------------------- 정확성
M0 = a.M[0]
pos0 = torch.randn(1, M0, 3, device=dev) * 0.3
x0 = torch.randn(1, M0, C, device=dev)
att = RelAttention(C, H).to(dev)
rp = RelPos(H, att.dc, lam_min=0.05, lam_max=2.0, sigma0=0.2).to(dev)
with torch.no_grad():
    cs, sn, qg, kg = rp(pos0)
    q, k, v = att.qkv(x0).chunk(3, -1)
    q, k, v = (t.view(1, M0, H, D).transpose(1, 2) for t in (q, k, v))
    qr, kr = _rope(q[..., :att.dc], cs, sn), _rope(k[..., :att.dc], cs, sn)
    qe, ke = torch.cat([qr, qg], -1), torch.cat([kr, kg], -1)
    got = (qe @ ke.transpose(-1, -2)) * att.scale
    Mh = rp.A.transpose(-1, -2) @ rp.A
    rel = pos0.unsqueeze(2) - pos0.unsqueeze(1)
    quad = torch.einsum("bijc,hcd,bijd->bhij", rel, Mh, rel)
    want = (qr @ kr.transpose(-1, -2)) * att.scale - att.scale * quad
    e_raw = float((got - want).abs().max())
    e_row = float((got - want - (got - want).mean(-1, keepdim=True)).abs().max())
    e_soft = float((got.softmax(-1) - want.softmax(-1)).abs().max())
print(f"[정확성] 거리 게이트: 원시 차이 {e_raw:.3e} (행 상수), 행 제거 후 "
      f"{e_row:.3e}, **softmax 후 {e_soft:.3e}**", flush=True)

with torch.no_grad():
    o0 = att(x0, rp(pos0))
    o1 = att(x0, rp(pos0 + torch.tensor([3.0, -1.0, 2.0], device=dev)))
    shift = float((o0 - o1).abs().max() / o0.abs().max())
    th = 0.7
    R = torch.tensor([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0],
                      [0, 0, 1]], device=dev, dtype=torch.float32)
    o2 = att(x0, rp(pos0 @ R.T))
    rot = float((o0 - o2).abs().max() / o0.abs().max())
print(f"[평행이동] 출력 상대 변화 {shift:.3e}   [회전] {rot:.3e} "
      f"(0 이면 방향을 못 본다)", flush=True)


# ---------------------------------------------------------------- 비용
def timeit(fn, warmup, reps):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.mean(ts))


old_bias = GeoAttentionBias(H).to(dev)
old_blocks = torch.nn.ModuleList([GeoAttentionBlock(C, H)
                                  for _ in range(a.depth)]).to(dev)
new_pos = RelPos(H, D - RelAttention.EXTRA, 0.05, 2.0, 0.2).to(dev)
new_blocks = torch.nn.ModuleList([RelBlock(C, H) for _ in range(a.depth)]).to(dev)

out = {}
for M in a.M:
    pos = torch.randn(1, M, 3, device=dev) * 0.3
    x = torch.randn(1, M, C, device=dev)

    def old_fwd():
        b = old_bias(pos)
        h = x
        for blk in old_blocks:
            h = blk(h, b)
        return h

    def new_fwd(sdpa):
        for blk in new_blocks:
            blk.att.USE_SDPA = sdpa
        ctx = new_pos(pos)
        h = x
        for blk in new_blocks:
            h = blk(h, ctx)
        return h

    row = {}
    for name, fn in (("옛 (쌍별 MLP 바이어스)", old_fwd),
                     ("새 (행렬곱)", lambda: new_fwd(False)),
                     ("새 (SDPA)", lambda: new_fwd(True))):
        try:
            with torch.no_grad():
                f = timeit(fn, a.warmup, a.reps)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            b = timeit(lambda: fn().square().mean().backward(), a.warmup, a.reps)
            row[name] = dict(fwd=f, fwd_bwd=b,
                             peak_mb=torch.cuda.max_memory_allocated() / 1e6)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            row[name] = dict(oom=True)
    out[M] = row
    print(f"\n[M={M}]", flush=True)
    for n, r in row.items():
        if r.get("oom"):
            print(f"  {n:<24} 메모리 부족", flush=True)
        else:
            print(f"  {n:<24} 순 {r['fwd']:6.2f} ms  순+역 {r['fwd_bwd']:7.2f} ms  "
                  f"최대 {r['peak_mb']:6.0f} MB", flush=True)

if a.out:
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(heads=H, hidden=C, depth=a.depth, exact_softmax_err=e_soft,
                   shift_err=shift, rot_change=rot,
                   timing={str(k): v for k, v in out.items()}),
              open(os.path.join(a.out, "rel_attention.json"), "w"), indent=1,
              ensure_ascii=False)
print("RELATT_OK")
