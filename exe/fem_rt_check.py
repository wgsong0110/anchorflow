"""fem_rt 검산·풀이 비교 (doc/fem_realtime.md 절 H): 정답 궤적 프레임 t 상태 (x, v, F = 정답) 에서
  (1) --floor node 에너지 기울기 vs 중심 차분, 조립 헤시안·벡터 곱 vs 자동미분 헤시안-벡터 곱 (hess_mode, float64)
  (2) 블록 ELL 곱 vs CSR 곱, 멀티그리드 계층 (Galerkin 스텐실 밖으로 떨어진 항 = 0 이어야 한다)
  (3) 같은 투영 헤시안 계 A p = g 를 블록 야코비 PCG / 멀티그리드 PCG (ν, ω, f32) / PD 분해 전처리 PCG 로 풀어 반복·시간·해 차
사용: python exe/fem_rt_check.py --t 0,40 [--floor node|particle]"""
import argparse
import json
import math
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
from anchorflow import phys_resid as PR                                             # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/home/dkta/work/one/L000/L000.pt")
ap.add_argument("--t", default="0,40")
ap.add_argument("--node_h", type=float, default=0.028284)
ap.add_argument("--k_st", type=float, default=1000.0)
ap.add_argument("--floor", default="node")
ap.add_argument("--out", default="")
a = ap.parse_args()
dev = "cuda"
d = torch.load(a.data, map_location="cpu", weights_only=False)
cfg = d["cfg"]
h = float(cfg["frame_dt"])
m = DM.traj_mass(d, dev).float()
EXT = float(d["_ext"])
vol = m / float(cfg["density"])
N = int(d["x"].shape[1])
Fall = DM.traj_F(d, h, dev)
g = torch.as_tensor(cfg["g"], device=dev, dtype=torch.float32)
mu, lam = PR.lame(cfg["E"], cfg["nu"])
dhat = 0.5 * float(cfg.get("grid_lim", 2.0)) / int(cfg["n_grid"])
res = {}


def sync_time(fn, rep=3):
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(rep):
        out = fn()
    torch.cuda.synchronize()
    return out, 1e3 * (time.perf_counter() - t0) / rep


for t in [int(q) for q in a.t.split(",")]:
    R = {}
    x = d["x"][t].to(dev).float()
    v = d["v"][t].to(dev).float()
    F = Fall[t].to(dev).float().reshape(N, 3, 3)
    G = RT.Grid(x, a.node_h)
    U, ws = RT.node_inertial(x, v, m, G, g, h)
    R.update(M=G.M, T=G.T)
    if a.floor == "node":
        fr = RT.NodeFrame(G, x, v, F, m, vol, cfg, h, dhat=dhat, k_st=a.k_st)
        tets = G.tets
        # (1) 기울기·헤시안 검산 (float64)
        E0, gg = fr.energy_grad(U)
        gen = torch.Generator(device=dev).manual_seed(7 + t)
        dU = torch.randn(U.shape, device=dev, dtype=torch.float64, generator=gen) * 1e-4 * EXT
        for epsr in (1e-2, 1e-3):
            e = epsr
            Ep = float(fr.energy(U.double() + e * dU))
            Em = float(fr.energy(U.double() - e * dU))
            fd = (Ep - Em) / (2 * e)
            an = float((gg * dU).sum())
            R[f"grad_fd_rel_eps{epsr:g}"] = abs(fd - an) / max(abs(an), 1e-300)
        He, Dn = fr.hess(U, kdt=torch.float64)
        csr = RT.Csr(tets, G.M)
        A_raw = csr.build(He, Dn)
        Ud = U.double().clone().requires_grad_(True)
        with torch.enable_grad(), PR.hess_mode():
            Ead = fr.energy(Ud)
            gad, = torch.autograd.grad(Ead, Ud, create_graph=True)
            R["grad_vs_ad_rel"] = float((gad.detach() - gg).norm() / gad.detach().norm())
            for nm in ("rand", "rand2"):
                vv = torch.randn(U.shape, device=dev, dtype=torch.float64, generator=gen)
                Hv_ad, = torch.autograd.grad((gad * vv).sum(), Ud, retain_graph=True)
                Hv_as = (A_raw @ vv.reshape(-1, 1)).reshape(-1, 3)
                R[f"Hv_{nm}_rel"] = float((Hv_as - Hv_ad).norm() / Hv_ad.norm())
        # 바닥 항 상태
        nr, cn, w = fr.planes[0]
        R.update(cn_min=float(cn.min()), n_barrier_nodes=int((cn < dhat).sum()), n_stick_nodes=int((w > 0.1).sum()))
        He, Dn = fr.hess(U)
    else:
        fr = FF.FemFrame(x, v, F, m, vol, G.rows, G.lam, G.npos, None, cfg, h, eps_sp=0.01, eps_st=0.01, stick_xn=True, dtype=torch.float32)
        fr.tetq_setup(U)
        tets = fr.uq
        E0, gg, _ = fr.energy_grad(U)
        He, _ = fr.hess_tetq(U)
        Dn = None
        csr = RT.Csr(tets, G.M)
    He, negf = RT.project(He)
    A = csr.build(He, Dn)
    D = RT.block_diag(tets, He, G.M, Dn)
    Dinv = torch.linalg.inv(D)
    b = gg.reshape(-1).double()
    mv = lambda q: (A @ q.unsqueeze(-1)).squeeze(-1)
    jac = lambda r: (Dinv @ r.reshape(-1, 3, 1)).reshape(-1)
    # (2) 블록 ELL
    (Be, t_bell) = sync_time(lambda: (lambda B_: (B_.add_tets(tets, He), B_.add_diag(Dn) if Dn is not None else None, B_)[-1])(RT.BEll(G.gi)), 1)
    vv = torch.randn(b.shape, device=dev, dtype=torch.float64)
    R["bell_vs_csr_rel"] = float((Be.mv(vv) - mv(vv)).norm() / mv(vv).norm())
    _, R["t_mv_csr_ms"] = sync_time(lambda: mv(vv), 20)
    _, R["t_mv_bell_ms"] = sync_time(lambda: Be.mv(vv), 20)
    R["t_bell_build_ms"] = t_bell
    # (3) 풀이 비교
    ref, t_ref = sync_time(lambda: RT.pcg(mv, jac, b, 20000, 1e-10, 50), 1)
    xref = ref[0]
    sol = {}

    def run(nm, pre_fn, build=None, rtol=1e-6):
        t0 = time.perf_counter()
        pre = build() if build is not None else pre_fn
        torch.cuda.synchronize()
        tb = 1e3 * (time.perf_counter() - t0)
        (xs, it, rr, ng), ts = sync_time(lambda: RT.pcg(mv, pre, b, 20000, rtol, 10), 1)
        sol[nm] = dict(iters=it, build_ms=tb, solve_ms=ts, per_it_ms=ts / max(it, 1), res=math.sqrt(rr), neg=ng,
                       err_vs_ref=float((xs - xref).norm() / xref.norm()),
                       qdrop_rel=float((b * xs).sum() - 0.5 * (xs * mv(xs)).sum()) / float((b * xref).sum() - 0.5 * (xref * mv(xref)).sum()))
        print(f"  t={t} {nm}: 반복 {it} 만들기 {tb:.1f} ms 풀이 {ts:.1f} ms ({ts / max(it, 1):.3f} ms/회) 해 차 {sol[nm]['err_vs_ref']:.2e}", flush=True)

    for rt in (1e-6, 1e-4, 1e-3):
        run(f"jacobi_rtol{rt:g}", jac, rtol=rt)
    for nu, om, f32 in ((1, 0.6, False), (2, 0.6, False), (2, 0.5, False), (2, 0.7, False), (3, 0.6, False), (2, 0.6, True)):
        nm = f"mg_nu{nu}_om{om}{'_f32' if f32 else ''}"
        try:
            holder = {}

            def bld(nu=nu, om=om, f32=f32):
                Mg = RT.KuhnMG(Be, nu=nu, omega=om, dtype=torch.float32 if f32 else torch.float64)
                holder["mg"] = Mg
                return Mg
            for rt in (1e-6, 1e-3):
                run(f"{nm}_rtol{rt:g}", None, build=bld, rtol=rt)
            sol[f"{nm}_rtol1e-06"].update(levels=holder["mg"].sizes, drop=holder["mg"].drop)
        except Exception as ex:                                              # noqa: BLE001
            sol[nm] = dict(error=f"{type(ex).__name__}: {str(ex)[:300]}")
            print(f"  t={t} {nm} 실패 {sol[nm]['error']}", flush=True)
    try:
        VT = torch.zeros(G.T, device=dev, dtype=torch.float64).index_add_(0, G.pinv, vol.double())
        for pw in (0.5, 1.0, 2.0):
            cd = torch.diagonal(Dn, dim1=1, dim2=2).mean(1) if Dn is not None else None

            def bpd(pw=pw, cd=cd):
                P = RT.PdFactor()
                P.build(G, ws.double(), VT, h, pw * 2.0 * mu, cdiag=cd, reuse=False)
                return P.solve
            for rt in (1e-6, 1e-3):
                run(f"pd_w{pw:g}_rtol{rt:g}", None, build=bpd, rtol=rt)
    except Exception as ex:                                                  # noqa: BLE001
        sol["pd"] = dict(error=f"{type(ex).__name__}: {str(ex)[:300]}")
        print(f"  t={t} PD 실패 {sol['pd']['error']}", flush=True)
    R["solvers"] = sol
    R["ref"] = dict(iters=ref[1], ms=t_ref)
    res[t] = R
    print(f"[검산 t={t}] " + json.dumps({k: v_ for k, v_ in R.items() if k != "solvers"}, ensure_ascii=False), flush=True)
if a.out:
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
