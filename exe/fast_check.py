"""빠른 롤아웃 경로(lib/anchorflow/fast_roll.py) 의 **같은 값 + 속도** 검사.

같은 스크립트·같은 GPU 에서 느린 경로(ref) 와 빠른 경로를 번갈아 80 프레임씩 돌려

  - 프레임별 |x_fast - x_ref| / ext 의 최대·평균 (ext = 궤적 0 프레임 물체 크기)
  - ref 를 한 번 더 돌린 ref2 와 ref 의 차이 (= 원래 코드의 실행간 잡음 바닥;
    index_add 의 atomic 순서가 실행마다 달라 원래도 비트 단위로 같지 않다)
  - GT 대비 오차 (lb_rollout 과 같은 정의: 평균 %, 정지 %, 비)
  - 프레임당 벽시계 (프레임마다 동기화), 프로파일러로 잰 스텝당 GPU 바쁜 시간과
    커널 호출 수

  python exe/fast_check.py --ckpt MAIN_best.pt --traj 0000.pt --modes ref,fast,ref2
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import numpy as np                                               # noqa: E402
import torch                                                     # noqa: E402
from anchorflow import deform_model as DM                        # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--traj", required=True)
ap.add_argument("--len", type=int, default=80)
ap.add_argument("--modes", default="ref,fast,ref2,eager")
ap.add_argument("--prof_steps", type=int, default=5)
ap.add_argument("--per_frame", action="store_true", help="프레임별 차이를 전부 출력")
a = ap.parse_args()
dev = "cuda"


def sync():
    torch.cuda.synchronize()


MODELS = {}


def model(mode):
    if mode not in MODELS:
        if mode.startswith("ref"):
            MODELS[mode] = MODELS.get("ref") or DM.EvalModel(a.ckpt, dev)
        elif mode == "eager":       # 융합 커널, 그래프 없이
            MODELS[mode] = DM.EvalModel(a.ckpt, dev, fast=True, fast_graph=False)
        elif mode == "torchg":      # torch 연산판 + 그래프
            MODELS[mode] = DM.EvalModel(a.ckpt, dev, fast=True, fast_triton=False)
        elif mode == "torche":      # torch 연산판, 그래프 없이
            MODELS[mode] = DM.EvalModel(a.ckpt, dev, fast=True, fast_graph=False,
                                        fast_triton=False)
        else:
            MODELS[mode] = DM.EvalModel(a.ckpt, dev, fast=True)
    return MODELS[mode]


d = torch.load(a.traj, map_location="cpu", weights_only=False)
for k in ("x", "v", "F"):
    if k in d and torch.is_tensor(d[k]):
        d[k] = d[k].to(dev)
T = d["x"].shape[0]
L = min(a.len, T - 1)
gsel = torch.arange(d["x"].shape[1], device=dev)
RES = {}
for mode in [m for m in a.modes.split(",") if m]:
    M = model(mode)
    with torch.no_grad():
        x, v = M.init(d, 0)
        ext = float(d["_ext"])
        xs = torch.empty(L, *x.shape, device=dev)
        wall = []
        errs, stills = [], []
        x_still = x.clone()
        for i in range(L):
            sync(); t0 = time.time()
            x, v, _J = M.frame(d, x, v)
            sync(); wall.append(time.time() - t0)
            xs[i] = x
            gt = DM.take(d["x"][i + 1], gsel, dev)
            errs.append(DM.frame_err(x, gt, ext))
            stills.append(DM.frame_err(x_still, gt, ext))
        w = DM.window_row(L, errs, stills)
        # 프로파일 (롤아웃 끝 상태에서 몇 스텝 더)
        from torch.profiler import profile, ProfilerActivity
        for _ in range(2):
            x, v, _ = M.frame(d, x, v)
        sync()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(a.prof_steps):
                x, v, _ = M.frame(d, x, v)
            sync()
        ka = prof.key_averages()
        cu = [e for e in ka if e.device_type == torch.autograd.DeviceType.CUDA]
        n_k = sum(e.count for e in cu) / a.prof_steps

        def _dt(e):
            for k in ("self_device_time_total", "self_cuda_time_total", "device_time_total"):
                if hasattr(e, k):
                    return getattr(e, k)
            return 0.0
        gpu_ms = sum(_dt(e) for e in cu) / a.prof_steps / 1e3
        # 이벤트로 잰 한 스텝 (동기화 포함 벽시계와 따로)
        ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        sync(); t0 = time.time(); ev0.record()
        for _ in range(10):
            x, v, _ = M.frame(d, x, v)
        ev1.record(); sync()
        run10 = (time.time() - t0) / 10 * 1e3
        top = sorted(cu, key=_dt, reverse=True)[:12]
    wall = np.array(wall) * 1e3
    RES[mode] = dict(xs=xs, w=w, wall=wall)
    print(f"[{mode}] 평균 {100*w['err_mean']:.3f}% (정지 {100*w['still_mean']:.3f}%, 비 {w['ratio']:.2f}) | "
          f"프레임 벽시계 첫 {wall[0]:.1f} ms, 이후 중앙 {np.median(wall[1:]):.2f} 평균 {wall[1:].mean():.2f} ms | "
          f"연속 10 스텝 {run10:.2f} ms/스텝 | GPU 바쁜 {gpu_ms:.2f} ms/스텝, 커널 {n_k:.0f} 회/스텝 | "
          f"최대 메모리 {torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)
    for e in top:
        print(f"    {_dt(e)/a.prof_steps/1e3:7.3f} ms  x{e.count/a.prof_steps:5.1f}  {e.key[:90]}")
    if getattr(M, "_fs", None) is not None:
        print(f"    용량 {M._fs.caps} 마지막 크기 {M._fs.last_stat}", flush=True)

ext = float(d["_ext"])
if "ref" in RES:
    xr = RES["ref"]["xs"]
    for mode, r in RES.items():
        if mode == "ref":
            continue
        dn = (r["xs"] - xr).norm(dim=-1) / ext                    # [L,N]
        mx = dn.max(1).values.cpu().numpy()
        mn = dn.mean(1).cpu().numpy()
        print(f"[차이 {mode} vs ref] 프레임별 최대 |dx|/ext: 첫 {mx[0]:.2e} 10번 {mx[min(9,L-1)]:.2e} "
              f"40번 {mx[min(39,L-1)]:.2e} 끝 {mx[-1]:.2e} (전체 최대 {mx.max():.2e}) | "
              f"평균 |dx|/ext: 첫 {mn[0]:.2e} 끝 {mn[-1]:.2e} (전체 최대 {mn.max():.2e}, 프레임 평균 {mn.mean():.2e})",
              flush=True)
        if a.per_frame:
            for i in range(L):
                print(f"    f{i+1:02d} max {mx[i]:.3e} mean {mn[i]:.3e}")
print("FAST_CHECK_OK")
