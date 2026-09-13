"""학생 학습 한 반복이 왜 자유 F 에서 2.7 배 느린가 -- 단계별로 쪼갠다.

한 반복에 드는 것:
  1. 배치 인덱싱 (TRAJ 에서 상태·목표 뽑기) -- 채널이 3 대 12 라 데이터가 4 배
  2. 순전파 x rollout_steps
  3. 손실
  4. 역전파
두 설정을 같은 모양으로 돌려 어디서 벌어지는지 본다.
"""
from __future__ import annotations

import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--M", type=int, default=1024); ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--chunk", type=int, default=4); ap.add_argument("--rollout_steps", type=int, default=4)
ap.add_argument("--n_traj", type=int, default=250); ap.add_argument("--T", type=int, default=61)
ap.add_argument("--n", type=int, default=30)
args = ap.parse_args()
dev = "cuda"
from anchorflow.nextstate import NextStep, apply_step, apply_step_frame

def sync(): torch.cuda.synchronize()
def bench(fn, n, warm=5):
    for _ in range(warm): fn()
    sync(); t0 = time.time()
    for _ in range(n): fn()
    sync(); return (time.time() - t0) / n * 1e3

H = args.chunk * args.rollout_steps
print(f"[setup] M={args.M} batch={args.batch} chunk={args.chunk} "
      f"rollout_steps={args.rollout_steps} HORIZON={H}", flush=True)
print(f"\n{'설정':<12}{'인덱싱':>10}{'순전파':>10}{'손실':>9}{'역전파':>10}{'합계':>10}{'TRAJ GB':>10}")
for name, frame, C in (("base (p,v)", False, 3), ("자유 F", True, 12)):
    ne = C - 3
    TRAJ = torch.randn(args.n_traj, args.T, args.M, C, device=dev)
    gb = TRAJ.numel() * 4 / 1e9
    net = NextStep(128, 4, 4, 1.0, 1.0, 1.0, use_accel=False, chunk=args.chunk,
                   frame=frame, n_extra=max(ne, 1) if frame else 6).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    fixed = torch.zeros(args.M, dtype=torch.bool, device=dev)
    idx = torch.randint(args.n_traj, (args.batch,), device=dev)
    k = torch.randint(H + 2, args.T - H - 2, (args.batch,), device=dev)

    def index():
        st = TRAJ[idx, k]
        v = (st[..., :3] - TRAJ[idx, k - 1][..., :3]) / 0.004
        TG = torch.stack([TRAJ[idx, k + j + 1] for j in range(H)], 1)
        return st, v, TG
    ms_idx = bench(index, args.n)
    st, v, TG = index()
    p0 = st[..., :3].contiguous(); u0 = st[..., 3:].contiguous() if frame else None

    def fwd():
        if frame: net(p0, v, None, 0.004, u=u0, s=u0[..., :0])
        else: net(p0, v, None, 0.004)
    ms_fwd = bench(fwd, args.n) * args.rollout_steps

    def full(back):
        p, u, s = p0, u0, (u0[..., :0] if frame else None)
        vv = v; loss = 0.0; step = 0
        for j in range(args.rollout_steps):
            steps = (apply_step_frame(net, p, vv, u, s, None, 0.004, fixed) if frame
                     else apply_step(net, p, vv, None, 0.004, fixed))
            for stp in steps:
                tgt = TG[:, step]
                if frame:
                    q, d, u, s = stp
                    loss = loss + ((q - tgt[..., :3]) ** 2).mean() \
                                + ((u - tgt[..., 3:]) ** 2).mean()
                else:
                    q, d = stp
                    loss = loss + ((q - tgt) ** 2).mean()
                step += 1
            p, vv = q, d / 0.004
        if back:
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        return loss
    ms_fl = bench(lambda: full(False), args.n)
    ms_all = bench(lambda: full(True), args.n)
    ms_loss = max(ms_fl - ms_fwd, 0.0); ms_back = max(ms_all - ms_fl, 0.0)
    print(f"{name:<12}{ms_idx:>10.2f}{ms_fwd:>10.2f}{ms_loss:>9.2f}{ms_back:>10.2f}"
          f"{ms_idx+ms_all:>10.2f}{gb:>10.2f}", flush=True)
    del TRAJ, net, opt; torch.cuda.empty_cache()
print("\nSTEP_DONE")
