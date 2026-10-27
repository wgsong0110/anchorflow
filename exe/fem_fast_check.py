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
    try:
        fr.kmode = "compile"
        (HTc, _), tHc = timed(lambda: fr.hess_tet(U0), reps=3)
        R[f"{nm}_HTc_rel"] = float((HTc - HTd).norm() / HTd.norm()); R[f"{nm}_t_hess_tet_compile_ms"] = 1e3 * tHc
        print(f"[{nm}] 사면체 축약 + 컴파일 K 커널: 기존 대비 {R[f'{nm}_HTc_rel']:.3e} ({1e3 * tHc:.1f} ms)")
        del HTc
    except Exception as e:  # noqa: BLE001
        print(f"[{nm}] 컴파일 K 실패: {e!r}"[:500])
    for km in ("manual",):
        try:
            fr.kmode = km
            (HTc, _), tHc = timed(lambda: fr.hess_tet(U0), reps=3)
            R[f"{nm}_HT_{km}_rel"] = float((HTc - HTd).norm() / HTd.norm()); R[f"{nm}_t_hess_tet_{km}_ms"] = 1e3 * tHc
            print(f"[{nm}] 사면체 축약 + {km} K 커널: 기존 대비 {R[f'{nm}_HT_{km}_rel']:.3e} ({1e3 * tHc:.1f} ms)")
            del HTc
        except Exception as e:  # noqa: BLE001
            print(f"[{nm}] {km} K 실패: {e!r}"[:500])
    try:
        fr.kmode, fr.near_mode = "manual", "chain"
        (HTc, _), tHc = timed(lambda: fr.hess_tet(U0), reps=3)
        R[f"{nm}_HT_chain_rel"] = float((HTc - HTd).norm() / HTd.norm()); R[f"{nm}_t_hess_tet_chain_ms"] = 1e3 * tHc
        print(f"[{nm}] 사면체 축약 + manual K + 바닥 후보 연쇄법칙: 기존 대비 {R[f'{nm}_HT_chain_rel']:.3e} ({1e3 * tHc:.1f} ms)")
        del HTc
    except Exception as e:  # noqa: BLE001
        import traceback; traceback.print_exc()
    fr.kmode, fr.near_mode = "autograd", "autograd"
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
    (ew, eV), tEig = timed(lambda: torch.linalg.eigh(He), reps=2)
    R[f"{nm}_t_eigh_ms"] = 1e3 * tEig
    print(f"[{nm}] 사면체 {T} 개 12x12 eigh (float64) {1e3 * tEig:.1f} ms")
    if nm == "f32":
        # PCG: 기준판과 같은 투영 (|λ|, 바닥 1e-6·max|λ|) -> CSR 동기 PCG (기준판 식) vs 행렬 없는 CUDA 그래프 PCG
        Hp = (eV * torch.maximum(ew.abs(), 1e-6 * ew.abs().amax(1, keepdim=True)).unsqueeze(1)) @ eV.transpose(1, 2)
        M_ = U0.shape[0]; fixd = D["fixd"].to(dev)
        fx3 = fixd.unsqueeze(-1).expand(-1, 3).reshape(-1)
        dof = (fr.uq.unsqueeze(-1) * 3 + torch.arange(3, device=dev)).reshape(-1, 12)
        ri = dof.unsqueeze(-1).expand(-1, 12, 12).reshape(-1); ci = dof.unsqueeze(1).expand(-1, 12, 12).reshape(-1)
        kp = ~(fx3[ri] | fx3[ci]); fi = torch.nonzero(fx3).squeeze(1)
        A_ = torch.sparse_coo_tensor(torch.stack([torch.cat([ri[kp], fi]), torch.cat([ci[kp], fi])]),
                                     torch.cat([Hp.reshape(-1)[kp], torch.ones(fi.numel(), device=dev, dtype=torch.float64)]), (3 * M_, 3 * M_)).coalesce().to_sparse_csr()
        ar = torch.arange(4, device=dev)
        blk = Hp.reshape(T, 4, 3, 4, 3)[:, ar, :, ar, :]
        Dg = torch.zeros(M_, 3, 3, device=dev, dtype=torch.float64).index_add_(0, fr.uq.t().reshape(-1), blk.reshape(-1, 3, 3))
        Dg[fixd] = torch.eye(3, device=dev, dtype=torch.float64)
        bad = ~(torch.diagonal(Dg, dim1=1, dim2=2) > 0).all(1); Dg[bad] = torch.eye(3, device=dev, dtype=torch.float64)
        Dinv = torch.linalg.inv(Dg)
        bb = g.double().reshape(-1).masked_fill(fx3, 0.0)

        def pcg_sync(ncg, rtol=1e-6):
            pre = lambda r: (Dinv @ r.reshape(M_, 3, 1)).reshape(-1)
            x_ = torch.zeros_like(bb); r_ = bb.clone(); z_ = pre(r_); d_ = z_.clone(); rz = float((r_ * z_).sum()); b2 = float((bb * bb).sum())
            for kk in range(1, ncg + 1):
                Ad = (A_ @ d_.unsqueeze(-1)).squeeze(-1); dAd = float((d_ * Ad).sum())
                if not dAd > 0:
                    break
                al = rz / dAd; x_ = x_ + al * d_; r_ = r_ - al * Ad; rr = float((r_ * r_).sum())
                if rr / b2 < rtol * rtol:
                    break
                z_ = pre(r_); rz2 = float((r_ * z_).sum()); d_ = z_ + (rz2 / rz) * d_; rz = rz2
            return x_, kk
        for ncg in (10000, 500, 50):
            (xs, ks), ts = timed(lambda: pcg_sync(ncg), reps=1)
            mf = fem_fast.MFPcg(Hp, dof, fx3, Dinv, M_, ck=50)
            (xm, km, _, _), tm = timed(lambda: mf.solve(bb, ncg, 1e-6), reps=1)
            rel = float((xm - xs).norm() / xs.norm())
            R[f"pcg{ncg}"] = dict(sync_s=ts, sync_it=ks, mf_s=tm, mf_it=km, x_rel=rel)
            print(f"[PCG 상한 {ncg}] CSR 동기 {ks} 회 {1e3 * ts:.1f} ms ({1e3 * ts / max(ks, 1):.3f} ms/회) | 행렬 없는 CUDA 그래프 {km} 회 {1e3 * tm:.1f} ms "
                  f"({1e3 * tm / max(km, 1):.3f} ms/회) | 해 상대차 {rel:.3e}")
    del HL, HT, HLd, HTd, He
    torch.cuda.empty_cache()
if A.out:
    json.dump(R, open(A.out, "w"), indent=1, ensure_ascii=False)
print("FEM_FAST_CHECK_DONE")
