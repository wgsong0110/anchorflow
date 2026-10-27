"""lib/anchorflow/fem_fast.py 검산·시간 (AF_FEM_DUMP 덤프 하나): 기존 경로(_fwd 파이프라인) 값과 같은지, 각 단계가 얼마나 걸리는지.
사용: python exe/fem_fast_check.py <dump.pt> [--out json]"""
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
print(f"[재질] {phys_resid.mat_name(cfg)} (덤프 {D['mat']})")
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


for dt in (torch.float32, torch.float64):
    nm = "f32" if dt == torch.float32 else "f64"
    U0 = D["U0"].to(dev)
    fr = fem_fast.FemFrame(D["x"].to(dev), D["v"].to(dev), D["F"].to(dev).reshape(-1, 3, 3), D["m"].to(dev),
                           D["vol"].to(dev) if torch.is_tensor(D["vol"]) else D["vol"], D["rw"].to(dev), D["lm"].to(dev),
                           D["npos"].to(dev), D["fixd"].to(dev), cfg, D["h"], eps_sp=a.get("bc_softplus", 0), eps_st=a.get("bc_stick", 0),
                           stick_xn=a.get("bc_stick_xn", False), dtype=dt)
    E, tE = timed(lambda: fr.energy(U0))
    (E2, g, gL), tG = timed(lambda: fr.energy_grad(U0))
    R[f"{nm}_E"] = float(E); R[f"{nm}_E_rel"] = abs(float(E) - D["EJ"]) / abs(D["EJ"])
    free = ~D["fixd"].to(dev)
    gd = D["gd"].to(dev).double()[:, :3]
    R[f"{nm}_g_rel"] = float((g.double() - gd)[free].norm() / gd[free].norm())
    R[f"{nm}_gL_rel"] = float((gL.double() - D["gL"].to(dev).double()).norm() / D["gL"].to(dev).double().norm())
    R[f"{nm}_t_energy_ms"] = 1e3 * tE; R[f"{nm}_t_grad_ms"] = 1e3 * tG
    print(f"[{nm}] E 빠른 {float(E):.10e} vs 기존 {D['EJ']:.10e} 상대차 {R[f'{nm}_E_rel']:.3e} | 기울기(자유 노드) 상대차 {R[f'{nm}_g_rel']:.3e} "
          f"입자 국소 {R[f'{nm}_gL_rel']:.3e} | 시간 E {1e3 * tE:.2f} ms, E+기울기 {1e3 * tG:.2f} ms")
    HLd = D["HL"].to(dev)
    HL, tHp = timed(lambda: fr.hess_particle(U0), reps=2)
    R[f"{nm}_HL_rel"] = float((HL - HLd).norm() / HLd.norm())
    T = fr.uq.shape[0]
    HTd = torch.zeros(T, 12, 12, device=dev, dtype=torch.float64).index_add_(0, D["inv"].to(dev), HLd)
    (HT, info), tHt = timed(lambda: fr.hess_tet(U0), reps=2)
    R[f"{nm}_HT_rel"] = float((HT - HTd).norm() / HTd.norm())
    R[f"{nm}_t_hess_particle_ms"] = 1e3 * tHp; R[f"{nm}_t_hess_tet_ms"] = 1e3 * tHt; R[f"{nm}_near"] = info
    print(f"[{nm}] 입자 12x12 vs 기존 HL 상대차 {R[f'{nm}_HL_rel']:.3e} ({1e3 * tHp:.1f} ms) | 사면체 축약 12x12 vs 기존 사면체 합 {R[f'{nm}_HT_rel']:.3e} "
          f"({1e3 * tHt:.1f} ms, 바닥 후보 {info['n_near']} / 항등 {info['n_far']})")
    # B (∇λ) vs 덤프 Bp (매핑 항등 입자)
    far = ~fr.near(U0)
    I3 = torch.eye(3, device=dev, dtype=torch.float64)
    Bf = (I3[None, :, None, None, :] * fr.gl.double()[:, None, :, :, None]).permute(0, 1, 3, 2, 4)  # [N,a,b,i,c] = δac ∇λ_i,b
    Bf = Bf.reshape(-1, 9, 12)
    Bd = D["Bp"].to(dev)
    R[f"{nm}_B_rel_far"] = float((Bf[far] - Bd[far]).norm() / Bd[far].norm())
    print(f"[{nm}] B(∇λ) vs 덤프 ∂Jd/∂U (매핑 항등 입자) 상대차 {R[f'{nm}_B_rel_far']:.3e}")
    # 투영 (eigh) 시간
    He = 0.5 * (HT + HT.transpose(1, 2))
    _, tEig = timed(lambda: torch.linalg.eigh(He), reps=2)
    R[f"{nm}_t_eigh_ms"] = 1e3 * tEig
    print(f"[{nm}] 사면체 {T} 개 12x12 eigh (float64) {1e3 * tEig:.1f} ms")
    del HL, HT, HLd, HTd, He
    torch.cuda.empty_cache()
if A.out:
    json.dump(R, open(A.out, "w"), indent=1, ensure_ascii=False)
print("FEM_FAST_CHECK_DONE")
