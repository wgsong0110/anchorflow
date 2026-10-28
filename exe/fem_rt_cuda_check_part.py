"""입자 바닥 (--floor particle, 기존 softplus·stick 매핑) CUDA 커널 대조 (fem_rt_cuda.CudaTetq): 정답 프레임 t 상태에서
  E·기울기 vs FemFrame.tetq_energy_grad (torch), 바닥 후보 12x12 (사면체로 합) vs FemFrame.hess_near_analytic, 투영 헤시안 블록 ELL vs
  torch (hess_tetq + |λ| 투영), 시간 (torch 바닥 후보 헤시안 vs 커널).  사용: python exe/fem_rt_cuda_check_part.py --t 0,40"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
os.environ.setdefault("AF_FEM_FAST_K", "manual")
os.environ.setdefault("AF_FEM_FAST_NEAR", "analytic")
import torch                                                                        # noqa: E402
from anchorflow import deform_model as DM                                           # noqa: E402
from anchorflow import fem_fast as FF                                               # noqa: E402
from anchorflow import fem_rt as RT                                                 # noqa: E402
from anchorflow import fem_rt_cuda as RTC                                           # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/home/dkta/work/one/L000/L000.pt")
ap.add_argument("--t", default="0,40")
a = ap.parse_args()
dev = "cuda"
d = torch.load(a.data, map_location="cpu", weights_only=False)
cfg = d["cfg"]
h = float(cfg["frame_dt"])
m = DM.traj_mass(d, dev).float()
vol = m / float(cfg["density"])
N = int(d["x"].shape[1])
Fall = DM.traj_F(d, h, dev)
g = torch.as_tensor(cfg["g"], device=dev, dtype=torch.float32)


def tm(fn, rep=5):
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(rep):
        out = fn()
    torch.cuda.synchronize()
    return out, 1e3 * (time.perf_counter() - t0) / rep


for t in [int(q) for q in a.t.split(",")]:
    x = d["x"][t].to(dev).float()
    v = d["v"][t].to(dev).float()
    F = Fall[t].to(dev).float().reshape(N, 3, 3)
    G = RT.Grid(x, 0.028284, fast=True)
    U, ws = RT.node_inertial(x, v, m, G, g, h)
    fr = FF.FemFrame(x, v, F, m, vol, G.rows, G.lam, G.npos, None, cfg, h, eps_sp=0.01, eps_st=0.01, stick_xn=True, dtype=torch.float32)
    fr.tetq_setup(U)
    CF = RTC.CudaTetq(fr)
    ni = fr.tq["ni"]
    (Et, gt, _), t_et = tm(lambda: fr.energy_grad(U))
    (Ec, gc), t_ec = tm(lambda: CF.energy_grad(U))
    print(f"[t={t}] 바닥 후보 {ni.numel()} / {N}, 사면체 {CF.T} | E torch {float(Et):.10e} cuda {float(Ec):.10e} 상대차 {abs(float(Et) - float(Ec)) / abs(float(Et)):.2e} "
          f"| 기울기 상대차 {float((gt.double() - gc).norm() / gt.double().norm()):.2e} | E+기울기 torch {t_et:.2f} ms cuda {t_ec:.2f} ms", flush=True)
    _, t_e = tm(lambda: CF.energy(U))
    _, t_et2 = tm(lambda: fr.energy(U))
    print(f"  에너지만 torch {t_et2:.2f} ms cuda {t_e:.2f} ms", flush=True)
    # 바닥 후보 12x12 (사면체로 합) 대조
    Hn_t, t_nt = tm(lambda: torch.zeros(CF.T, 12, 12, device=dev, dtype=torch.float64).index_add_(0, fr.inv[ni], fr.hess_near_analytic(U, ni)))
    Hn_c, t_nc = tm(lambda: CF.near_only(U))
    print(f"  바닥 후보 헤시안 (사면체 합) 상대차 {float((Hn_c.double() - Hn_t).norm() / Hn_t.norm()):.2e} | torch {t_nt:.1f} ms cuda {t_nc:.2f} ms", flush=True)
    # 전체 투영 헤시안 블록 ELL
    def th():
        He, _ = fr.hess_tetq(U)
        Hp, _ = RT.project(He)
        Be = RT.BEll(G.gi)
        Be.add_tets(fr.uq, Hp)
        return Be
    Be, t_ht = tm(th, 2)
    M = G.M
    gi = G.gi
    tets = fr.uq
    slots = (tets.unsqueeze(-1) * 15 + RT.off_cls(gi[tets].unsqueeze(1) - gi[tets].unsqueeze(2))).int().contiguous()
    Bc = torch.zeros(M * 135, dtype=torch.float32, device=dev)

    Bt = Be.B.view(M, 15, 3, 3)
    for gt_ in (0.0, 1e-6, 1e-5):
        def ch(gt_=gt_):
            Bc.zero_()
            CF.cnt.zero_()
            CF.hess_into(U, slots, Bc, 1e-6, Ms=M, gtau=gt_)
        _, t_hc = tm(ch)
        Bcv = Bc.view(15, 9, M).permute(2, 0, 1).reshape(M, 15, 3, 3).double()
        print(f"  [판정 τ {gt_:g}] 투영 헤시안 블록 ELL 상대차 {float((Bcv - Bt).norm() / Bt.norm()):.2e} | 헤시안+투영+합 torch {t_ht:.1f} ms cuda {t_hc:.2f} ms "
              f"| 판정 통과 사면체 {int(CF.cnt[0])}/{CF.T}", flush=True)
