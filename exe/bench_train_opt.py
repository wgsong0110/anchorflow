"""학생 학습의 최적화 조합을 속도와 수치로 함께 잰다.

속도만 재면 안 된다 -- SDPA 는 같은 수식이지만 부동소수점 경로가 다르고, 혼합
정밀도는 위치(좌표 O(1) 대 변위 4e-3)의 유효 자릿수를 실제로 깎는다. 그래서 같은
입력·같은 가중치에서 **출력이 얼마나 달라지는지**를 함께 낸다. 기준은 옛 경로다.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_lib = os.path.join(os.path.dirname(__file__), "..", "lib")
sys.path.insert(0, _lib)

import torch

from anchorflow import nextstate as NS
from anchorflow.nextstate import NextStep

ap = argparse.ArgumentParser()
ap.add_argument("--anchors", type=int, default=1024)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--hidden", type=int, default=128)
ap.add_argument("--depth", type=int, default=4)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--chunk", type=int, default=4)
ap.add_argument("--rollout_steps", type=int, default=4)
ap.add_argument("--iters", type=int, default=30)
ap.add_argument("--warmup", type=int, default=5)
args = ap.parse_args()

dev = "cuda"
torch.manual_seed(0)
M, B = args.anchors, args.batch
DT = 0.004


def build():
    torch.manual_seed(0)
    return NextStep(args.hidden, args.depth, args.heads, 0.00206, 0.5146, 1.0,
                    use_accel=False, chunk=args.chunk).to(dev)


torch.manual_seed(1)
P0 = (torch.rand(B, M, 3, device=dev) * 2.0)          # 좌표 O(1), 실제와 같은 규모
V0 = torch.randn(B, M, 3, device=dev) * 0.5
TGT = P0 + torch.randn(B, M, 3, device=dev) * 0.002   # 변위 규모 4e-3 수준
FIXED = torch.zeros(M, dtype=torch.bool, device=dev)
FIXED[: M // 8] = True


def one(net, amp):
    """rollout_steps 만큼 굴리고 손실까지. 학습 한 스텝과 같은 모양."""
    ctx = (torch.autocast("cuda", dtype=amp) if amp is not None
           else torch.enable_grad())
    p, v, loss = P0.clone(), V0.clone(), 0.0
    with ctx:
        for _ in range(args.rollout_steps):
            for q, d in NS.apply_step(net, p, v, None, DT, FIXED):
                p, v = q, d / DT
                loss = loss + ((p - TGT) / 0.00206).pow(2).mean()
    return loss, p


def run(tag, sdpa, amp, compile_, tf32=False):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    NS.set_sdpa(sdpa)
    net = build()
    if compile_:
        net = torch.compile(net)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for i in range(args.warmup):
        loss, _ = one(net, amp)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for i in range(args.iters):
        loss, _ = one(net, amp)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / args.iters
    mem = torch.cuda.max_memory_allocated() / 2 ** 30
    # 수치 비교는 학습 전 가중치에서. 같은 입력, 같은 파라미터, 경로만 다르다.
    NS.set_sdpa(sdpa)
    ref_net = build()
    with torch.no_grad():
        _, p_out = one(ref_net, amp)
    del net, opt
    torch.cuda.empty_cache()
    return dt, mem, p_out.float()


print(f"[setup] M={M} B={B} hidden={args.hidden} depth={args.depth} "
      f"chunk={args.chunk} rollout_steps={args.rollout_steps}", flush=True)

# SDPA 는 뺐다. 기하 바이어스가 [B,H,M,M] 가산 마스크라 flash/mem-efficient
# 백엔드가 안 잡히고 math 로 떨어진다 -- 같은 행렬을 그대로 만들면서 메모리만
# 두 배 쓰고(1024 에서는 OOM) 속도 이득은 0 이었다.
COMBOS = [
    ("기준(옛 경로)",     False, None,           False, False),
    ("+TF32",             False, None,           False, True),
    ("+compile",          False, None,           True,  False),
    ("+TF32+compile",     False, None,           True,  True),
    ("+bf16",             False, torch.bfloat16, False, False),
    ("+bf16+compile",     False, torch.bfloat16, True,  False),
    ("+fp16+compile",     False, torch.float16,  True,  False),
    ("+TF32+bf16+compile", False, torch.bfloat16, True, True),
]
base_dt = base_p = None
print(f"\n{'조합':<22}{'ms/it':>9}{'배속':>7}{'최대메모리':>11}{'출력 상대차':>13}")
for tag, sdpa, amp, comp, tf32 in COMBOS:
    try:
        dt, mem, p = run(tag, sdpa, amp, comp, tf32)
    except Exception as e:
        print(f"{tag:<22}  실패: {type(e).__name__}: {str(e)[:60]}", flush=True)
        continue
    if base_dt is None:
        base_dt, base_p = dt, p
        rel = 0.0
    else:
        rel = float((p - base_p).norm() / base_p.norm().clamp(min=1e-20))
    print(f"{tag:<22}{1000*dt:>9.1f}{base_dt/dt:>7.2f}x{mem:>10.2f}G{rel:>13.2e}",
          flush=True)

print("\n출력 상대차는 학습 전 가중치·같은 입력에서 굴린 최종 위치의 상대 차이다.")
print("BENCH_DONE")
