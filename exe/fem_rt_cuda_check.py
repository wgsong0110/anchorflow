"""fem_rt_cuda 커널 대조 (정답 프레임 t 상태, --floor node): E·기울기 vs NodeFrame.energy_grad_fast (torch),
투영 헤시안 블록 ELL vs torch (hess + project + BEll), PCG 해 vs torch PCG (같은 행렬·rtol), 커널 시간.
사용: python exe/fem_rt_cuda_check.py --t 0,40"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
os.environ.setdefault("AF_FEM_FAST_K", "manual")
import torch                                                                        # noqa: E402
from anchorflow import deform_model as DM                                           # noqa: E402
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
M_FULL = float(cfg["density"]) * float(torch.linalg.det(RT.SX.lattice(0.028284, RT.SX.HZ_CUBE * 0.028284, dtype=torch.float64).A).abs())
PC = RTC.CudaPcg(60000, dev)


def tm(fn, rep=10):
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
    fr = RT.NodeFrame(G, x, v, F, m, vol, cfg, h, dhat=0.01, k_st=10.0, m_floor=M_FULL)
    CF = RTC.CudaFem(fr)
    (Et, gt), t_t = tm(lambda: fr.energy_grad_fast(U))
    (Ec, gc), t_c = tm(lambda: CF.energy_grad(U))
    print(f"[t={t}] E torch {float(Et):.10e} cuda {float(Ec):.10e} 상대차 {abs(float(Et - Ec)) / abs(float(Et)):.2e} | 기울기 상대차 "
          f"{float((gt - gc).norm() / gt.norm()):.2e} | 시간 torch {t_t:.2f} ms cuda {t_c:.2f} ms | σ≤0.01 사면체 {int(CF.cnt[0])}", flush=True)
    _, t_e = tm(lambda: CF.energy(U))
    _, t_et = tm(lambda: fr.energy_fast(U))
    print(f"  에너지만: torch {t_et:.2f} ms cuda {t_e:.2f} ms", flush=True)
    # 헤시안
    He, Dn = fr.hess(U, f32_b=True)
    Hp, _ = RT.project(He)
    Be = RT.BEll(G.gi)
    Be.add_tets(G.tets, Hp)
    Be.add_diag(Dn)
    M = G.M
    gi = G.gi
    slots = (G.tets.unsqueeze(-1) * 15 + RT.off_cls(gi[G.tets].unsqueeze(1) - gi[G.tets].unsqueeze(2))).int().contiguous()

    def ch():
        PC.B[:M * 135].zero_()
        CF.hess_into(U, slots, PC.B, 1e-6)
    _, t_h = tm(ch)
    Bc = PC.B[:M * 135].view(M, 15, 3, 3).double().clone()
    Bc[:, 0] += Dn
    Bt = Be.B.view(M, 15, 3, 3)
    print(f"  투영 헤시안 블록 ELL 상대차 {float((Bc - Bt).norm() / Bt.norm()):.2e} (자기 블록 {float((Bc[:, 0] - Bt[:, 0]).norm() / Bt[:, 0].norm()):.2e}) "
          f"| cuda 헤시안+투영+합 {t_h:.2f} ms | 음 고유값 사면체 {int(CF.cnt[1])}/{G.T}", flush=True)

    def th():
        He_, Dn_ = fr.hess(U, f32_b=True)
        Hp_, _ = RT.project(He_)
        B_ = RT.BEll(G.gi)
        B_.add_tets(G.tets, Hp_)
        return B_
    _, t_ht = tm(th, 3)
    print(f"  torch 헤시안+투영+합 {t_ht:.2f} ms", flush=True)
    # PCG: 같은 행렬 (cuda 블록 ELL) 로 torch PCG vs cuda PCG
    PC.B[:M * 135].view(M, 15, 9)[:, 0] += Dn.reshape(M, 9).float()
    D = PC.B[:M * 135].view(M, 15, 9)[:, 0].double().reshape(M, 3, 3)
    Dinv = torch.linalg.inv(D)
    PC.Dinv[:M * 9].copy_(Dinv.reshape(-1))
    PC.nb[:M * 15].copy_(Be.nb.reshape(-1).int())
    Bd = PC.B[:M * 135].view(M, 15, 3, 3).double()
    mv = lambda q: torch.einsum("mkab,mkb->ma", Bd, q.reshape(M, 3)[Be.nb]).reshape(-1)
    b = gc.reshape(-1).double()
    (xt, it_t, rt_, _), t_pt = tm(lambda: RT.pcg(mv, lambda r: (Dinv @ r.reshape(-1, 3, 1)).reshape(-1), b, 10000, 1e-6, 1), 1)
    for ug in (False, True):
        (xc, it_c, rc_, ng), t_pc = tm(lambda: PC.solve(M, b, 1e-6, 10000, use_graph=ug), 3)
        print(f"  PCG torch {it_t} 회 {t_pt:.1f} ms | cuda{' 그래프' if ug else ''} {it_c} 회 {t_pc:.2f} ms 잔차 {rc_:.2e} | 해 상대차 {float((xt - xc).norm() / xt.norm()):.2e}", flush=True)
