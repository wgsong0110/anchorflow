"""상대위치 어텐션이 (1) 주장한 대로 정확한지, (2) 실제로 싼지 확인한다.

두 가지를 각각 따로 검증한다.

  정확성  거리 게이트는 "근사"가 아니라 항등식이라고 주장했다. softmax 가 행 상수를
          지우므로 -(p_i-p_j)^T M (p_i-p_j) 와 q/k 에 붙인 4 채널의 내적은 **같은
          어텐션 가중치**를 낳아야 한다. 명시적으로 [M,M] 을 만들어 비교한다.
  RoPE    q_i^T R(p_j-p_i) k_j 가 되는지는, 점 구름을 통째로 평행이동시켜도 출력이
          같은지로 본다. 절대 좌표가 새면 여기서 걸린다.
  비용    옛 경로(쌍마다 MLP -> [M,M,H] 바이어스)와 새 경로를 같은 조건에서 잰다.
          순전파만이 아니라 역전파까지 재야 한다 -- 35% 를 먹던 것이 중간 텐서를
          붙잡고 있던 비용이기 때문이다.
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
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--M", type=int, default=512)
ap.add_argument("--hidden", type=int, default=128)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--depth", type=int, default=4)
ap.add_argument("--reps", type=int, default=50)
ap.add_argument("--warmup", type=int, default=10)
ap.add_argument("--out", default=None)
a = ap.parse_args()

dev = "cuda"
torch.manual_seed(0)
from anchorflow.deform import RelAttention, RelBlock          # noqa: E402
from anchorflow.nextstate import (GeoAttentionBias,           # noqa: E402
                                  GeoAttentionBlock)

M, H, C = a.M, a.heads, a.hidden
D = C // H
pos = torch.randn(1, M, 3, device=dev) * 0.3
x = torch.randn(1, M, C, device=dev)

# ---------------------------------------------------------------- 정확성
att = RelAttention(C, H, lam_min=0.05, lam_max=2.0, sigma0=0.2).to(dev)
with torch.no_grad():
    q, k, v = att.qkv(x).chunk(3, -1)
    q, k, v = (t.view(1, M, H, D).transpose(1, 2) for t in (q, k, v))
    q, k = att.rope(q[..., :att.dc], pos), att.rope(k[..., :att.dc], pos)
    Mh = att.A.transpose(-1, -2) @ att.A
    Mp = torch.einsum("hcd,bmd->bhmc", Mh, pos)
    quad = (pos.unsqueeze(1) * Mp).sum(-1, keepdim=True)
    qe = torch.cat([q, 2.0 * Mp, torch.ones_like(quad)], -1)
    ke = torch.cat([k, pos.unsqueeze(1).expand_as(Mp), -quad], -1)
    got = (qe @ ke.transpose(-1, -2)) * att.scale                # [1,H,M,M]
    # 명시적으로 만든 기준: 내용 항 + 정확한 이차형식
    rel = pos.unsqueeze(2) - pos.unsqueeze(1)                    # [1,M,M,3]
    quad_full = torch.einsum("bijc,hcd,bijd->bhij", rel, Mh, rel)
    want = (q @ k.transpose(-1, -2)) * att.scale - att.scale * quad_full
    # 행 상수만큼 다를 수 있다 -- softmax 뒤에 같아야 한다는 것이 주장이다
    e_raw = float((got - want).abs().max())
    e_row = float((got - want - (got - want).mean(-1, keepdim=True)).abs().max())
    e_soft = float((got.softmax(-1) - want.softmax(-1)).abs().max())
print(f"[정확성] 거리 게이트: 원시 차이 {e_raw:.3e} (행 상수), 행 제거 후 "
      f"{e_row:.3e}, **softmax 후 {e_soft:.3e}**", flush=True)

# 평행이동 불변
with torch.no_grad():
    o0 = att(x, pos)
    o1 = att(x, pos + torch.tensor([3.0, -1.0, 2.0], device=dev))
    shift = float((o0 - o1).abs().max() / o0.abs().max())
print(f"[평행이동] 좌표를 통째로 옮겼을 때 출력 상대 변화 {shift:.3e}", flush=True)

# 회전시키면 달라져야 정상이다 (회전 등변이 아니라 회전 '인지')
with torch.no_grad():
    th = 0.7
    R = torch.tensor([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0],
                      [0, 0, 1]], device=dev, dtype=torch.float32)
    o2 = att(x, pos @ R.T)
    rot = float((o0 - o2).abs().max() / o0.abs().max())
print(f"[회전] 좌표를 돌렸을 때 출력 상대 변화 {rot:.3e} (0 이면 방향을 못 본다)",
      flush=True)


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
old_blocks = torch.nn.ModuleList([GeoAttentionBlock(C, H) for _ in range(a.depth)]).to(dev)
new_blocks = torch.nn.ModuleList([
    RelBlock(C, H, 0.05, 2.0, 0.2) for _ in range(a.depth)]).to(dev)


def old_fwd():
    b = old_bias(pos)
    h = x
    for blk in old_blocks:
        h = blk(h, b)
    return h


def new_fwd():
    h = x
    for blk in new_blocks:
        h = blk(h, pos)
    return h


def bwd(fn):
    def go():
        fn().square().mean().backward()
    return go


res = {}
for name, fn in (("옛 경로 (쌍별 MLP 바이어스)", old_fwd),
                 ("새 경로 (RoPE-3D + 4채널 게이트)", new_fwd)):
    with torch.no_grad():
        f = timeit(fn, a.warmup, a.reps)
    torch.cuda.reset_peak_memory_stats()
    b = timeit(bwd(fn), a.warmup, a.reps)
    mem = torch.cuda.max_memory_allocated() / 1e6
    res[name] = dict(fwd=f, fwd_bwd=b, peak_mb=mem)
    print(f"[{name}] 순전파 {f:.3f} ms, 순+역 {b:.3f} ms, 최대 메모리 {mem:.0f} MB",
          flush=True)

lo = res["옛 경로 (쌍별 MLP 바이어스)"]
ne = res["새 경로 (RoPE-3D + 4채널 게이트)"]
print(f"\n[요약] M={M} depth={a.depth}: 순+역 {lo['fwd_bwd']:.2f} -> "
      f"{ne['fwd_bwd']:.2f} ms ({lo['fwd_bwd']/ne['fwd_bwd']:.2f}x), "
      f"메모리 {lo['peak_mb']:.0f} -> {ne['peak_mb']:.0f} MB", flush=True)

if a.out:
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(M=M, heads=H, hidden=C, depth=a.depth,
                   exact_softmax_err=e_soft, shift_err=shift, rot_change=rot,
                   timing=res),
              open(os.path.join(a.out, "rel_attention.json"), "w"), indent=1,
              ensure_ascii=False)
# 어떤 SDPA 백엔드가 실제로 잡히는지 (조용히 fallback 되므로 확인해야 한다)
from torch.nn.attention import SDPBackend, sdpa_kernel               # noqa: E402

qq = torch.randn(1, H, M, D, device=dev, dtype=torch.float16)
for be, nm in ((SDPBackend.FLASH_ATTENTION, "flash"),
               (SDPBackend.EFFICIENT_ATTENTION, "mem-efficient")):
    try:
        with sdpa_kernel(be):
            F.scaled_dot_product_attention(qq, qq, qq)
        print(f"[백엔드] head_dim={D} 에서 {nm} 사용 가능", flush=True)
    except Exception as e:
        print(f"[백엔드] head_dim={D} 에서 {nm} 불가: {type(e).__name__}", flush=True)

print("RELATT_OK")
