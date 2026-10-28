"""멀티그리드 V 사이클 구간 시간 (fem_rt.KuhnMG): 정답 프레임 t 상태 노드 바닥 계에서 연산마다 동기화 시간 + torch.profiler 상위 커널.
사용: python exe/fem_rt_prof_mg.py --t 40"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
os.environ.setdefault("AF_FEM_FAST_K", "manual")
import torch                                                                        # noqa: E402
from anchorflow import deform_model as DM                                           # noqa: E402
from anchorflow import fem_rt as RT                                                 # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/home/dkta/work/one/L000/L000.pt")
ap.add_argument("--t", type=int, default=40)
a = ap.parse_args()
dev = "cuda"
d = torch.load(a.data, map_location="cpu", weights_only=False)
cfg = d["cfg"]
h = float(cfg["frame_dt"])
m = DM.traj_mass(d, dev).float()
vol = m / float(cfg["density"])
N = int(d["x"].shape[1])
x = d["x"][a.t].to(dev).float()
v = d["v"][a.t].to(dev).float()
F = DM.traj_F(d, h, dev)[a.t].to(dev).float().reshape(N, 3, 3)
g = torch.as_tensor(cfg["g"], device=dev, dtype=torch.float32)
G = RT.Grid(x, 0.028284, fast=True)
U, ws = RT.node_inertial(x, v, m, G, g, h)
fr = RT.NodeFrame(G, x, v, F, m, vol, cfg, h, dhat=0.01, k_st=10.0)
He, Dn = fr.hess(U)
He, _ = RT.project(He)
Be = RT.BEll(G.gi)
Be.add_tets(G.tets, He)
Be.add_diag(Dn)
csr = RT.Csr(G.tets, G.M)
A = csr.build(He, Dn)
b = fr.energy_grad_fast(U)[1].reshape(-1)


def tm(nm, fn, rep=50):
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(rep):
        fn()
    torch.cuda.synchronize()
    print(f"  {nm}: {1e3 * (time.perf_counter() - t0) / rep:.4f} ms", flush=True)


for dt in (torch.float64, torch.float32):
    for bell in (False, True):
        MG = RT.KuhnMG(Be, nu=2, omega=0.6, dtype=dt, A0csr=(None if bell else A), bell=bell)
        print(f"[MG {dt} bell={bell}] 수준 {MG.sizes}", flush=True)
        r = b.to(dt)
        lv = MG.levels[0]
        if lv["C"] is not None:
            C = lv["C"]
            tm("fine CSR 곱", lambda: (C @ r.unsqueeze(-1)).squeeze(-1))
        tm("fine BEll 곱", lambda: lv["A"].mv(r))
        tm("야코비", lambda: MG._jac(lv, r))
        pidx, w = lv["P"]
        res = r.reshape(-1, 3)
        tm("제한", lambda: torch.zeros(lv["Mc"], 3, device=dev, dtype=dt).index_add_(0, pidx.reshape(-1), (w.unsqueeze(-1) * res.unsqueeze(1)).reshape(-1, 3)))
        rc = torch.zeros(lv["Mc"], 3, device=dev, dtype=dt)
        tm("연장", lambda: (w.unsqueeze(-1) * rc[pidx]).sum(1))
        tm("가장 성긴 풀이", lambda: MG.Ki @ torch.zeros(MG.Ki.shape[0], device=dev, dtype=dt))
        tm("V 사이클 전체", lambda: MG(b))
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            out = MG(b)
        torch.cuda.current_stream().wait_stream(st)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            out = MG(b)
        tm("V 사이클 그래프 재생", lambda: gr.replay())
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CUDA]) as pf:
            for _ in range(5):
                MG(b)
            torch.cuda.synchronize()
        print(pf.key_averages().table(sort_by="cuda_time_total", row_limit=8), flush=True)
tm("CSR f64 곱 (PCG)", lambda: (A @ b.unsqueeze(-1)).squeeze(-1))
tm("PCG MG 그래프 rtol1e-6", lambda: RT.pcg_graph(lambda q: (A @ q.unsqueeze(-1)).squeeze(-1), RT.KuhnMG(Be, nu=2, omega=0.6, A0csr=A), b.double(), 10000, 1e-6), 3)
