"""PCG 전처리 비교 (실시간 판, 덤프 하나): 실행과 같은 투영 (|λ|, 바닥 1e-6·max|λ|, 고유분해 float32) 의 사면체 헤시안 -> CSR,
b = 기울기. 블록 야코비 (기준) / 2 단계 add·mult (묶음 크기·ω) / float32 SpMV 를 같은 rtol 1e-6 으로 풀어 반복수·시간·해 차이.
사용: python exe/fem_pcg_check.py <dump.pt> [--tetq] [--out json]"""
import sys, os, time, json, argparse
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from anchorflow import fem_fast, phys_resid  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("dump"); ap.add_argument("--out"); ap.add_argument("--tetq", action="store_true")
A = ap.parse_args()
dev = "cuda"
D = torch.load(A.dump, map_location="cpu", weights_only=False)
a = D["args"]
cfg = dict(E=D["E"], nu=D["nu"], g=D["g"], boundary_conditions=D["bcs"], n_grid=D["n_grid"], grid_lim=D["grid_lim"])
cfg["material"] = D["mat"]
if phys_resid.mat_name(cfg) != D["mat"]:
    cfg = dict(cfg, mat=D["mat"])
os.environ.setdefault("AF_FEM_FAST_K", "manual"); os.environ.setdefault("AF_FEM_FAST_NEAR", "analytic")
fr = fem_fast.FemFrame(D["x"].to(dev), D["v"].to(dev), D["F"].to(dev).reshape(-1, 3, 3), D["m"].to(dev),
                       D["vol"].to(dev) if torch.is_tensor(D["vol"]) else D["vol"], D["rw"].to(dev), D["lm"].to(dev),
                       D["npos"].to(dev), D["fixd"].to(dev), cfg, D["h"], eps_sp=a.get("bc_softplus", 0), eps_st=a.get("bc_stick", 0),
                       stick_xn=a.get("bc_stick_xn", False), dtype=torch.float32)
U0 = D["U0"].to(dev)
fixd = D["fixd"].to(dev)
if A.tetq:
    fr.tetq_setup(U0)
    _, g, _ = fr.tetq_energy_grad(U0)
    H, _ = fr.hess_tetq(U0)
else:
    _, g, _ = fr.energy_grad(U0)
    H, _ = fr.hess_tet(U0)
H = 0.5 * (H + H.transpose(1, 2))
ew, eV = torch.linalg.eigh(H.float())
ew, eV = ew.double(), eV.double()
He = (eV * torch.maximum(ew.abs(), 1e-6 * ew.abs().amax(1, keepdim=True)).unsqueeze(1)) @ eV.transpose(1, 2)
T = He.shape[0]; M_ = U0.shape[0]
fx3 = fixd.unsqueeze(-1).expand(-1, 3).reshape(-1)
dof = (fr.uq.unsqueeze(-1) * 3 + torch.arange(3, device=dev)).reshape(-1, 12)
ri = dof.unsqueeze(-1).expand(-1, 12, 12).reshape(-1); ci = dof.unsqueeze(1).expand(-1, 12, 12).reshape(-1)
kp = ~(fx3[ri] | fx3[ci]); fi = torch.nonzero(fx3).squeeze(1)
Acsr = torch.sparse_coo_tensor(torch.stack([torch.cat([ri[kp], fi]), torch.cat([ci[kp], fi])]),
                               torch.cat([He.reshape(-1)[kp], torch.ones(fi.numel(), device=dev, dtype=torch.float64)]), (3 * M_, 3 * M_)).coalesce().to_sparse_csr()
Acsr32 = torch.sparse_csr_tensor(Acsr.crow_indices(), Acsr.col_indices(), Acsr.values().float(), Acsr.shape)
ar = torch.arange(4, device=dev)
blk = He.reshape(T, 4, 3, 4, 3)[:, ar, :, ar, :]
Dg = torch.zeros(M_, 3, 3, device=dev, dtype=torch.float64).index_add_(0, fr.uq.t().reshape(-1), blk.reshape(-1, 3, 3))
Dg[fixd] = torch.eye(3, device=dev, dtype=torch.float64)
bad = ~(torch.diagonal(Dg, dim1=1, dim2=2) > 0).all(1); Dg[bad] = torch.eye(3, device=dev, dtype=torch.float64)
Dinv = torch.linalg.inv(Dg)
b = g.double().reshape(-1).masked_fill(fx3, 0.0)
mv64 = lambda v: (Acsr @ v.unsqueeze(-1)).squeeze(-1)
mv32 = lambda v: (Acsr32 @ v.float().unsqueeze(-1)).squeeze(-1).double()
jac = lambda r: (Dinv @ r.reshape(M_, 3, 1)).reshape(-1)
print(f"[계] 사면체 {T}, 노드 {M_}, 자유도 {3 * M_}, nnz {Acsr.values().numel()}, tetq {A.tetq}", flush=True)


def pcg(mv, pre, ncg=10000, rtol=1e-6, ck=50):
    x_ = torch.zeros_like(b); r_ = b.clone(); z_ = pre(r_); d_ = z_.clone()
    rz = (r_ * z_).sum(); thr = (b * b).sum() * rtol * rtol
    alive = torch.ones((), dtype=torch.bool, device=dev); nit = torch.zeros((), dtype=torch.long, device=dev); one = torch.ones((), dtype=b.dtype, device=dev)
    for kk in range(1, ncg + 1):
        Ad = mv(d_); dAd = (d_ * Ad).sum()
        alive = alive & (dAd > 0)
        al = torch.where(alive, rz / torch.where(dAd > 0, dAd, one), torch.zeros_like(rz))
        x_ = x_ + al * d_; r_ = r_ - al * Ad
        nit = nit + alive.long()
        alive = alive & ~((r_ * r_).sum() < thr)
        z_ = pre(r_); rz2 = (r_ * z_).sum()
        d_ = z_ + torch.where(alive, rz2 / rz, torch.zeros_like(rz)) * d_
        rz = torch.where(alive, rz2, rz)
        if kk % ck == 0 and not bool(alive):
            break
    return x_, int(nit)


def run(name, mv, prefn, build=None):
    torch.cuda.synchronize(); t0 = time.time()
    pre = build() if build else prefn
    torch.cuda.synchronize(); tb = time.time() - t0
    pcg(mv, pre, ncg=60)                                   # 데우기
    torch.cuda.synchronize(); t0 = time.time()
    x, it = pcg(mv, pre)
    torch.cuda.synchronize(); ts = time.time() - t0
    rel = float((mv64(x) - b).norm() / b.norm())
    return x, dict(name=name, it=it, ms_build=1e3 * tb, ms_solve=1e3 * ts, ms_per_it=1e3 * ts / max(it, 1), res=rel)


R = []
x0, r0 = run("block_jacobi_f64", mv64, jac); R.append(r0)
for nm, mv in (("block_jacobi_f32spmv", mv32),):
    x, r = run(nm, mv, jac); r["x_rel"] = float((x - x0).norm() / x0.norm()); R.append(r)
for ck in (50, 25):
    torch.cuda.synchronize(); t0 = time.time()
    ep = fem_fast.EllPcg(Acsr, Dinv, M_, ck=ck)
    torch.cuda.synchronize(); tb = time.time() - t0
    x, r = run(f"ell_eager_ck{ck}", ep.mv, jac) if ck == 50 else (None, None)
    if r:
        r["x_rel"] = float((x - x0).norm() / x0.norm()); r["ms_ell_build"] = 1e3 * tb; r["K"] = int(ep.ecol.shape[1]); R.append(r)
    for rep in range(2):                                  # 첫 번은 그래프 잡기 포함, 둘째는 새로 잡기 (매 걸음 다시 잡는 비용 그대로)
        torch.cuda.synchronize(); t0 = time.time()
        xg, itg, rrg, ng = ep.solve(b, 10000, 1e-6, use_graph=True)
        torch.cuda.synchronize(); tg = time.time() - t0
        R.append(dict(name=f"ell_graph_ck{ck}_rep{rep}", it=itg, ms_solve=1e3 * tg, ms_per_it=1e3 * tg / max(itg, 1),
                      res=float((mv64(xg) - b).norm() / b.norm()), x_rel=float((xg - x0).norm() / x0.norm())))
    # CSR 곱 + 그래프 (cuSPARSE 를 그래프에 잡을 수 있는지)
    try:
        cp = fem_fast.EllPcg(Acsr, Dinv, M_, ck=ck); cp.mv = mv64
        torch.cuda.synchronize(); t0 = time.time()
        xg, itg, _, _ = cp.solve(b, 10000, 1e-6, use_graph=True)
        torch.cuda.synchronize(); tg = time.time() - t0
        R.append(dict(name=f"csr_graph_ck{ck}", it=itg, ms_solve=1e3 * tg, ms_per_it=1e3 * tg / max(itg, 1), x_rel=float((xg - x0).norm() / x0.norm())))
    except Exception as e:  # noqa: BLE001
        R.append(dict(name=f"csr_graph_ck{ck}", err=repr(e)[:200]))
for r in R:
    print(json.dumps(r, ensure_ascii=False), flush=True)
R0 = len(R)
npos = D["npos"].to(dev)
nh = float(a.get("node_h", 0.028284))
for agg in (3, 4, 6):
    for mode, om in (("add", 0.0), ("mult", 0.5), ("mult", 0.7)):
        try:
            holder = {}

            def bld():
                holder["tl"] = fem_fast.TwoLevel(He, fr.uq, npos, fixd, Dinv, mv64, nh, agg=agg, mode=mode, omega=om)
                return holder["tl"]
            x, r = run(f"2lvl_{mode}_agg{agg}" + (f"_w{om}" if mode == "mult" else ""), mv64, None, build=bld)
            r["x_rel"] = float((x - x0).norm() / x0.norm()); r["n_coarse"] = holder["tl"].n6; r["coarse_f32"] = holder["tl"].f32; r["coarse_jit"] = getattr(holder["tl"], "jit", None)
            R.append(r)
            del holder
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            R.append(dict(name=f"2lvl_{mode}_agg{agg}", err=repr(e)[:300]))
for r in R[R0:]:
    print(json.dumps(r, ensure_ascii=False), flush=True)
if A.out:
    json.dump(R, open(A.out, "w"), indent=1, ensure_ascii=False)
print("FEM_PCG_CHECK_DONE")
