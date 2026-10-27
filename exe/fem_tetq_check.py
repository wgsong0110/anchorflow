"""AF_FEM_TETQ (사면체 적분점 하나, 사용자 10-27) 검산·시간 -- AF_FEM_DUMP 덤프 하나 (기존 dump40/dump/dump_t0040.pt).
  1) 근사 정도: 같은 A_T 에서 입자 적분점 vs 사면체 적분점 탄성 에너지, F 퍼짐 (tetq_diag)
  2) 기울기: tetq_energy_grad vs 중심 차분 (무작위 방향 3 개, float64)
  3) 헤시안: hess_tetq 조립 H·v vs 자동미분 헤시안-벡터 곱 (hess_mode, float64)
  4) 구간 시간 (float32 = 실행과 같은 dtype): P2G, E, E+기울기, 헤시안 (관성·K·B·바닥 후보), 기존 hess_tet 와 나란히
사용: python exe/fem_tetq_check.py <dump.pt> [--out json]"""
import sys, os, time, json, argparse
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from anchorflow import fem_fast, phys_resid  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("dump"); ap.add_argument("--out"); ap.add_argument("--reps", type=int, default=5)
A = ap.parse_args()
dev = "cuda"
D = torch.load(A.dump, map_location="cpu", weights_only=False)
a = D["args"]
cfg = dict(E=D["E"], nu=D["nu"], g=D["g"], boundary_conditions=D["bcs"], n_grid=D["n_grid"], grid_lim=D["grid_lim"])
cfg["material"] = D["mat"]
if phys_resid.mat_name(cfg) != D["mat"]:
    cfg = dict(cfg, mat=D["mat"])
os.environ.setdefault("AF_FEM_FAST_K", "manual"); os.environ.setdefault("AF_FEM_FAST_NEAR", "analytic")
R = {}


def sync():
    torch.cuda.synchronize()


def timed(fn, reps=A.reps):
    fn(); sync()
    t0 = time.time()
    for _ in range(reps):
        out = fn()
    sync()
    return out, (time.time() - t0) / reps


def frame(dt):
    return fem_fast.FemFrame(D["x"].to(dev), D["v"].to(dev), D["F"].to(dev).reshape(-1, 3, 3), D["m"].to(dev),
                             D["vol"].to(dev) if torch.is_tensor(D["vol"]) else D["vol"], D["rw"].to(dev), D["lm"].to(dev),
                             D["npos"].to(dev), D["fixd"].to(dev), cfg, D["h"], eps_sp=a.get("bc_softplus", 0), eps_st=a.get("bc_stick", 0),
                             stick_xn=a.get("bc_stick_xn", False), dtype=dt)


U0 = D["U0"].to(dev)
fixd = D["fixd"].to(dev)
free = ~fixd
# ---- 1~3: float64 ----
fr = frame(torch.float64)
U8 = U0.double()
fr.tetq_setup(U8)
R["diag_U0"] = fr.tetq_diag(U8)
print(f"[근사] {json.dumps(R['diag_U0'], ensure_ascii=False)}")
E_part = float(fr.particle_energy(U8[:, :3][fr.rw]).sum())
E_tq = float(fr.tetq_energy(U8))
R["E_total_part"], R["E_total_tetq"] = E_part, E_tq
print(f"[전체 E] 입자 적분점 {E_part:.10e} 사면체 적분점 {E_tq:.10e} 상대차 {(E_tq - E_part) / abs(E_part):.4e} (덤프 기존 {D['EJ']:.10e})")
E0, g, _ = fr.tetq_energy_grad(U8)
gen = torch.Generator(device=dev).manual_seed(7)
for j in range(3):
    v = torch.randn(U8.shape[0], 3, device=dev, dtype=torch.float64, generator=gen).masked_fill(fixd.unsqueeze(-1), 0.0)
    v = v / v.abs().max() * 1e-6
    Up, Um = U8.clone(), U8.clone()
    Up[:, :3] += v; Um[:, :3] -= v
    fd = (float(fr.tetq_energy(Up)) - float(fr.tetq_energy(Um))) / 2.0
    an = float((g.double() * v).sum())
    R[f"grad_fd{j}"] = dict(fd=fd, an=an, rel=abs(fd - an) / max(abs(fd), 1e-30))
    print(f"[기울기 차분 {j}] 차분 {fd:.6e} 해석 {an:.6e} 상대차 {R[f'grad_fd{j}']['rel']:.3e}")
H, info = fr.hess_tetq(U8)
T = fr.uq.shape[0]; M_ = U8.shape[0]
dof = (fr.uq.unsqueeze(-1) * 3 + torch.arange(3, device=dev)).reshape(-1, 12)


def Hmv(v):
    y = torch.bmm(H, v.reshape(-1)[dof].unsqueeze(-1)).reshape(-1)
    return torch.zeros(3 * M_, device=dev, dtype=torch.float64).index_add_(0, dof.reshape(-1), y).reshape(M_, 3)


Z = U8[:, :3].clone().requires_grad_(True)
with torch.enable_grad(), phys_resid.hess_mode():
    Uz = torch.cat([Z, U8[:, 3:]], 1)
    q = fr.tq
    Et = fr._tq_energy(Uz.double()[q["uq"]])
    ni = q["ni"]
    if ni.numel():
        Et = Et + fr.particle_energy(Uz[fr.rw[ni]], ni).sum()
    gZ, = torch.autograd.grad(Et, Z, create_graph=True)
    for j in range(3):
        v = torch.randn(M_, 3, device=dev, dtype=torch.float64, generator=gen).masked_fill(fixd.unsqueeze(-1), 0.0)
        hv_ad = torch.autograd.grad((gZ * v).sum(), Z, retain_graph=True)[0]
        hv_as = Hmv(v)
        rel = float((hv_as - hv_ad)[free].norm() / hv_ad[free].norm())
        R[f"Hv{j}_rel"] = rel
        print(f"[헤시안 H·v {j}] 조립 vs 자동미분 상대차 {rel:.3e} (vᵀHv {float((v * hv_as).sum()):.6e} / {float((v * hv_ad).sum()):.6e})")
    R["g_vs_autograd_hessmode"] = float((gZ.detach() - g.double())[free].norm() / gZ.detach()[free].norm())
    print(f"[기울기] tetq_energy_grad vs hess_mode 자동미분 상대차 {R['g_vs_autograd_hessmode']:.3e}")
del H, gZ, Z, Et
torch.cuda.empty_cache()
# ---- 4: float32 시간 ----
fr = frame(torch.float32)
U = U0
(_, tP) = timed(lambda: fr.tetq_setup(U))
_, tE = timed(lambda: fr.tetq_energy(U))
_, tG = timed(lambda: fr.tetq_energy_grad(U))
(Hq, info), tH = timed(lambda: fr.hess_tetq(U), reps=3)
R["t_ms"] = dict(p2g=1e3 * tP, E=1e3 * tE, Eg=1e3 * tG, hess=1e3 * tH, hess_parts=fr.last_t)
R["info"] = {k: v for k, v in info.items() if k != "t"}
fr.tq = None
(_, info0), tH0 = timed(lambda: fr.hess_tet(U), reps=3)
R["t_ms"]["hess_old_tet"] = 1e3 * tH0; R["t_ms"]["hess_old_parts"] = fr.last_t
_, tG0 = timed(lambda: fr.energy_grad(U))
R["t_ms"]["Eg_old"] = 1e3 * tG0
He = 0.5 * (Hq + Hq.transpose(1, 2))
_, tEig = timed(lambda: torch.linalg.eigh(He.float()), reps=3)
R["t_ms"]["eigh_f32"] = 1e3 * tEig
print(f"[시간 ms, float32] {json.dumps(R['t_ms'], ensure_ascii=False)} | 사면체 활성 {info['n_act']}/{T} 바닥 후보 {info['n_near']} 항등 {info['n_far']}")
if A.out:
    json.dump(R, open(A.out, "w"), indent=1, ensure_ascii=False)
print("FEM_TETQ_CHECK_DONE")
