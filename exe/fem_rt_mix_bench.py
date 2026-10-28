"""혼합 정밀도 PCG (pcg_mix, doc/fem_realtime.md 절 I) 검산·시간: 정답 프레임 t 상태 입자 바닥 계 (대칭 반쪽 블록 ELL, 최종 판과 같은 행렬).
float64 지속 커널 (solve_coop, 최종 판) 대 float32 단독 (mode 0) 대 float32 + float64 신뢰 갱신 (mode 1, δ 여러 값).
사용: python exe/fem_rt_mix_bench.py --t 0 10 40"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import torch                                                                        # noqa: E402
from anchorflow import deform_model as DM                                           # noqa: E402
from anchorflow import fem_rt as RT                                                 # noqa: E402
from anchorflow import fem_rt_cuda as RTC                                           # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/home/dkta/work/one/L000/L000.pt")
ap.add_argument("--t", type=int, nargs="+", default=[0, 40])
a = ap.parse_args()
dev = "cuda"
d = torch.load(a.data, map_location="cpu", weights_only=False)
cfg = d["cfg"]
h = float(cfg["frame_dt"])
m = DM.traj_mass(d, dev).float()
vol = m / float(cfg["density"])
N = int(d["x"].shape[1])
g = torch.as_tensor(cfg["g"], device=dev, dtype=torch.float32)
print(f"[장치] {torch.cuda.get_device_name(0)}", flush=True)


def tm(fn, rep=3):
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(rep):
        out = fn()
    torch.cuda.synchronize()
    return out, 1e3 * (time.perf_counter() - t0) / rep


for t in a.t:
    x = d["x"][t].to(dev).float()
    v = d["v"][t].to(dev).float()
    F = DM.traj_F(d, h, dev)[t].to(dev).float().reshape(N, 3, 3)
    G = RTC.GridK(x, 0.028284)
    U, ws = RT.node_inertial(x, v, m, G, g, h)
    CF = RTC.CudaTetqK(G, x, v, F, m, vol, cfg, h, U)
    CF.warp, CF.near32, CF.sym = True, True, 1
    PC = RTC.CudaPcg(60000, dev)
    PC.sym = 1
    M = G.M
    gi = G.gi
    slots = (G.tets.unsqueeze(-1) * 15 + RT.off_cls(gi[G.tets].unsqueeze(1) - gi[G.tets].unsqueeze(2))).int().contiguous()
    PC.B.zero_()
    CF.hess_into(U, slots, PC.B, 1e-6, Ms=PC.Mmax)
    Bs = PC.B.view(15, 9, PC.Mmax)
    D = Bs[0, :, :M].t().double().reshape(M, 3, 3)
    PC.Dinv[:M * 9].copy_(torch.linalg.inv(D).reshape(-1))
    nbl = RT.BEll(G.gi, dtype=torch.float32)
    PC.nb.view(15, PC.Mmax)[:, :M] = torch.where(nbl.valid, nbl.nb, torch.full_like(nbl.nb, -1)).t().int()
    b = CF.energy_grad(U)[1].reshape(-1)
    (x0, it0, _, _), t0 = tm(lambda: PC.solve_coop(M, b, 1e-6, 20000))
    print(f"[t={t} 노드 {M}] float64 지속 (최종 판): {it0} 회 {t0:.2f} ms ({1e3 * t0 / it0:.1f} us/회) 실제 잔차 {PC.true_res(M, x0, b):.2e}", flush=True)
    for kern, cap, rows in (("pcg_mix", 2, 0), ("pcg_mix", 2, 1), ("pcg_mix_lb2", 2, 0), ("pcg_mix_lb2", 2, 1), ("pcg_mix_lb3", 3, 1),
                            ("pcg_mix_lb4", 4, 1), ("pcg_mix_lb4", 4, 0)):
        PC.mx_cap, PC.mx_kern, PC.mx_rows = cap, kern, rows
        if hasattr(PC, "mx_acc"):
            del PC.mx_acc
        for mode, dl in ((0, 0.0), (1, 0.1)):
            (xx, it, rr, neg), tt = tm(lambda: PC.solve_mix(M, b, 1e-6, 20000, mode=mode, delta=dl))
            tr = PC.true_res(M, xx, b)
            print(f"  {kern} 행 {rows} 블록 {PC.mx_nblk} {PC.mx_info} {'f32 단독' if mode == 0 else f'혼합 δ {dl:g}'}: {it} 회 갱신 {PC.last_nup} {tt:.2f} ms "
                  f"({1e3 * tt / max(it, 1):.1f} us/회) 실제 잔차 {tr:.2e} 해 차 {float((xx - x0).norm() / x0.norm()):.2e} 음곡률 {neg}", flush=True)
