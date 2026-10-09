"""깃발 장면: MPMAvatar 공식 이방성 천 MPM 과 PG 공식 MPM(jelly) 을 같은 초기 조건으로 돌리고 같은 렌더러로 그린다.

깃발 = make_flag.py 의 1.5 x 1.0 m 천(xz 평면, 90 x 60 격자) + 깃대(왼쪽 변 옆). 월드 좌표는 m, z 위.
학습한 3DGS(flag_gs)를 meta.json 배율로 월드에 되돌리고, 천 가우시안은 쉬는 메쉬의 가장 가까운 삼각형에
무게중심 좌표 + 법선 거리로 묶는다 (MPMAvatar 는 메쉬로 시뮬하므로 이 결합으로 가우시안을 옮긴다).
깃대 쪽 변(맨 왼쪽 꼭짓점 열)은 고정. 물성은 MPMAvatar 저자 학습값(D 0.854, E 414.8, H 0.956) 과 코드 기본
ν 0.3, γ 500, κ 500. PG 는 같은 E·ν·밀도의 jelly, 입자 = 천 가우시안, 깃대 쪽 가우시안 고정.
초기 조건:  a  천 전체(고정 제외)에 깃발 면 법선 방향 --vel m/s (돌풍)      b  펼친 채 정지 상태에서 중력만
시뮬 좌표 = 월드 x 0.8 + 이동 (MPMAvatar 아바타 장면과 같은 크기 규모), 25 fps x --frames, 서브스텝 400.

  --stage prep    (af)   가우시안·메쉬·결합 준비            -> work/prep.pt
  --stage mpma    (mpma) MPMAvatar 시뮬                   -> work/mpma_<mode>.pt (꼭짓점 궤적)
  --stage pg      (af)   PG 시뮬                          -> work/pg_<mode>.pt (입자 궤적 + F)
  --stage render  (af)   렌더                              -> work/<solver>_<mode>.mp4
"""
import argparse
import json
import math
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--stage", choices=["prep", "mpma", "pg", "render"], required=True)
ap.add_argument("--work", required=True)
ap.add_argument("--flag", default="/home/dkta/work/flag")
ap.add_argument("--gs", default="flag_gs_clean", help="학습한 3DGS (흰 군더더기 지운 것)")
ap.add_argument("--mode", choices=["a", "b"], default="a")
ap.add_argument("--solver", choices=["mpma", "pg"], default="mpma")
ap.add_argument("--vel", type=float, default=3.0)
ap.add_argument("--frames", type=int, default=75)
ap.add_argument("--substep", type=int, default=400)
ap.add_argument("--n_grid", type=int, default=250)
ap.add_argument("--res", type=int, default=800)
ap.add_argument("--E_mul", type=float, default=1.0, help="E 배수 (두 솔버 같게). 저자 학습값 414.8 은 한쪽만 매단 깃발엔 너무 무르다")
ap.add_argument("--H", type=float, default=0.9562, help="MPMAvatar 정지 모양 높이 배율 (저자 학습값; 1 이면 끔)")
ap.add_argument("--joint_faces", action="store_true", help="MPMAvatar: 고정 꼭짓점에 닿은 삼각형도 고정 (아바타 규약)")
ap.add_argument("--trace", type=int, default=-1, help="MPMAvatar: 이 프레임부터 서브스텝마다 속도·F 를 기록해 처음 터지는 입자를 찾는다")
ap.add_argument("--trace_every", type=int, default=10, help="추적 간격 (서브스텝)")
ap.add_argument("--pin_grid", action="store_true", help="PG: 깃대 쪽 띠의 격자 속도를 0 으로 (PG 공식 cuboid 경계조건)")
a = ap.parse_args()
import numpy as np                                               # noqa: E402
import torch                                                     # noqa: E402
torch.set_grad_enabled(False)
os.makedirs(a.work, exist_ok=True)
dev = "cuda"
W_, H_, NX, NZ, POLE = 1.5, 1.0, 90, 60, 2.2
Z0 = POLE - H_ - 0.05
SC = 0.8
GLIM = 2.0
SHIFT = np.array([1.0, 1.0, 1.0]) - SC * np.array([W_ / 2, 0.0, POLE / 2])
D_, E_, Hs = 0.854, 414.8 * a.E_mul, a.H
NU, GAM, KAP = 0.3, 500.0, 500.0


def rest_mesh():
    xs, zs = np.linspace(0, W_, NX + 1), np.linspace(0, H_, NZ + 1)
    X, Z = np.meshgrid(xs, zs, indexing="xy")
    V = np.stack([X.ravel(), np.zeros(X.size), Z.ravel() + Z0], 1)
    idx = lambda i, j: j * (NX + 1) + i                         # noqa: E731
    F = []
    for j in range(NZ):
        for i in range(NX):
            F += [[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)], [idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)]]
    pin = np.zeros(len(V), bool); pin[[idx(0, j) for j in range(NZ + 1)]] = True
    return V, np.array(F), pin


def tri_frame(V, F):
    """삼각형마다 [e1 e2 n] (3x3)."""
    e1 = V[:, F[:, 1]] - V[:, F[:, 0]] if V.dim() == 3 else V[F[:, 1]] - V[F[:, 0]]
    e2 = V[:, F[:, 2]] - V[:, F[:, 0]] if V.dim() == 3 else V[F[:, 2]] - V[F[:, 0]]
    n = torch.cross(e1, e2, dim=-1); n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.stack([e1, e2, n], -1)


if a.stage == "prep":
    from plyfile import PlyData
    meta = json.load(open(f"{a.flag}/flag_ns/meta.json"))
    pl = PlyData.read(f"{a.flag}/{a.gs}/point_cloud/iteration_30000/point_cloud.ply").elements[0]
    g = lambda k: np.asarray(pl[k], np.float32)                 # noqa: E731
    xyz = np.stack([g("x"), g("y"), g("z")], 1)
    op = 1 / (1 + np.exp(-g("opacity")))
    sc = np.exp(np.stack([g(f"scale_{i}") for i in range(3)], 1))
    q = np.stack([g(f"rot_{i}") for i in range(4)], 1); q /= np.linalg.norm(q, axis=1, keepdims=True)
    dc = np.stack([g(f"f_dc_{i}") for i in range(3)], 1)
    rest_names = sorted([p.name for p in pl.properties if p.name.startswith("f_rest_")], key=lambda s: int(s.split("_")[-1]))
    fr = np.stack([g(n) for n in rest_names], 1).reshape(len(xyz), 3, -1)
    shs = np.concatenate([dc[:, :, None], fr], 2).transpose(0, 2, 1)          # [N, 16, 3]
    keep = op > 0.02
    s_ = meta["scale"]; c_ = np.array(meta["center_yup_to_zup"])
    Xw = xyz / s_ + c_
    w_, x_, y_, z_ = q.T
    R = np.stack([1 - 2 * (y_ * y_ + z_ * z_), 2 * (x_ * y_ - w_ * z_), 2 * (x_ * z_ + w_ * y_),
                  2 * (x_ * y_ + w_ * z_), 1 - 2 * (x_ * x_ + z_ * z_), 2 * (y_ * z_ - w_ * x_),
                  2 * (x_ * z_ - w_ * y_), 2 * (y_ * z_ + w_ * x_), 1 - 2 * (x_ * x_ + y_ * y_)], 1).reshape(-1, 3, 3)
    S = R * (sc / s_)[:, None, :]
    C = S @ S.transpose(0, 2, 1)
    Xw, C, shs, op = Xw[keep], C[keep], shs[keep], op[keep]
    V, F, pin = rest_mesh()
    pole = (Xw[:, 0] < -0.002) | (np.abs(Xw[:, 1]) > 0.03)
    Xc = torch.as_tensor(Xw[~pole], device=dev).double()
    Vt = torch.as_tensor(V, device=dev).double(); Ft = torch.as_tensor(F, device=dev).long()
    cen = Vt[Ft].mean(1)
    cand = torch.cdist(Xc.float(), cen.float()).topk(8, largest=False).indices
    best_d = torch.full((Xc.shape[0],), 1e9, device=dev, dtype=torch.float64)
    tri = torch.zeros(Xc.shape[0], dtype=torch.long, device=dev); bary = torch.zeros(Xc.shape[0], 3, device=dev, dtype=torch.float64)
    off = torch.zeros(Xc.shape[0], device=dev, dtype=torch.float64)
    T0 = tri_frame(Vt, Ft)
    for k in range(8):
        t = cand[:, k]; M = T0[t]
        loc = torch.linalg.solve(M, (Xc - Vt[Ft[t, 0]])[..., None])[..., 0]   # (u, v, h): Xc = v0 + u e1 + v e2 + h n
        u, v_, h = loc.unbind(-1)
        uu, vv = u.clamp(0, 1), v_.clamp(0, 1)
        sm = (uu + vv).clamp_min(1.0); uu, vv = uu / sm, vv / sm
        p = Vt[Ft[t, 0]] + uu[:, None] * M[..., 0] + vv[:, None] * M[..., 1]
        d = (Xc - p).norm(dim=1)
        better = d < best_d
        best_d = torch.where(better, d, best_d); tri = torch.where(better, t, tri)
        bary = torch.where(better[:, None], torch.stack([1 - uu - vv, uu, vv], 1), bary)
        off = torch.where(better, ((Xc - p) * M[..., 2]).sum(1), off)
    print(f"[결합] 천 가우시안 {int((~pole).sum())}, 깃대 {int(pole.sum())}, 삼각형까지 거리 중앙 {float(best_d.median()):.2e} "
          f"최대 {float(best_d.max()):.2e} m", flush=True)
    torch.save(dict(X=torch.as_tensor(Xw), C=torch.as_tensor(C), shs=torch.as_tensor(shs), op=torch.as_tensor(op),
                    pole=torch.as_tensor(pole), tri=tri.cpu(), bary=bary.float().cpu(), off=off.float().cpu(),
                    V=torch.as_tensor(V), F=torch.as_tensor(F), pin=torch.as_tensor(pin)), f"{a.work}/prep.pt")
    print(f"[저장] {a.work}/prep.pt", flush=True)

elif a.stage == "mpma":
    MP = "/home/dkta/work/MPMAvatar"; sys.path.insert(0, MP); os.chdir(MP)
    import warp as wp
    from tqdm import tqdm
    from warp_mpm.mpm_data_structure import MPMStateStruct, MPMModelStruct
    from warp_mpm.mpm_solver import MPMWARP
    P = torch.load(f"{a.work}/prep.pt")
    V, F, pin = P["V"].numpy(), P["F"].numpy(), P["pin"].numpy()
    order = np.concatenate([np.nonzero(pin)[0], np.nonzero(~pin)[0]])           # 고정 꼭짓점을 앞으로 (공식 joint 규약)
    inv = np.empty_like(order); inv[order] = np.arange(len(order))
    Vr, Fr = V[order], inv[F]
    nj = int(pin.sum())
    njf = 0
    if a.joint_faces:                                              # 고정 꼭짓점에 닿은 삼각형을 앞으로 (공식 joint 면 규약)
        jf = (Fr < nj).any(1)
        Fr = np.concatenate([Fr[jf], Fr[~jf]]); njf = int(jf.sum())
    verts = torch.as_tensor(Vr * SC + SHIFT, device=dev).float(); faces = torch.as_tensor(Fr, device=dev).long()

    def dir_vol(v, f, th=1e-5):
        d1 = v[f[:, 1]] - v[f[:, 0]]; d2 = v[f[:, 2]] - v[f[:, 0]]
        d3 = torch.cross(d1, d2, dim=1); d3 = d3 / d3.norm(dim=1, keepdim=True)
        R11 = d1.norm(dim=1); R12 = (d1 * d2).sum(1) / R11; R22 = (d2 - (R12 / R11)[:, None] * d1).norm(dim=1)
        area = 0.5 * torch.cross(d1, d2, dim=1).norm(dim=1); ev = 0.25 * th * area
        vv = torch.zeros(v.shape[0], device=dev); vv.index_add_(0, f.reshape(-1), ev[:, None].repeat(1, 3).reshape(-1))
        return torch.stack([d1, d2, d3], -1), torch.stack([R11, R12, R22], -1), ev, vv

    def rinv(v, f):
        d1 = v[f[:, 1]] - v[f[:, 0]]; d2 = v[f[:, 2]] - v[f[:, 0]]
        R11 = d1.norm(dim=1); R12 = (d1 * d2).sum(1) / R11; R22 = (d2 - (R12 / R11)[:, None] * d1).norm(dim=1)
        return torch.stack([1 / R11, -R12 / (R11 * R22), 1 / R22], -1)

    ne, nv = faces.shape[0], verts.shape[0]; n = ne + nv
    pos0 = torch.cat([verts[faces].mean(1), verts], 0)
    d0, _, ev, vv = dir_vol(verts, faces)
    if os.environ.get("WARP_KCACHE"):                          # 동시에 여러 실행이 같은 캐시를 컴파일하다 깨지는 것을 막는다
        wp.config.kernel_cache_dir = os.environ["WARP_KCACHE"]
    wp.init()
    st = MPMStateStruct(); st.init(n, ne, nv, device="cuda:0", requires_grad=False)
    pt = np.zeros(n, np.int32); pv = np.zeros(n, np.int32); pv[ne:] = 1; pe = np.zeros(n, np.int32); pe[:ne] = 1
    st.from_torch(pos0.clone(), torch.cat([ev, vv]).clone(), torch.linalg.inv(d0), rinv(verts, faces), faces.int(), pt, pv, pe,
                  torch.zeros(ne, 6, device=dev), device="cuda:0", requires_grad=False, n_grid=a.n_grid, grid_lim=2.0)
    md = MPMModelStruct(); md.init(n, device="cuda:0", requires_grad=False); md.init_other_params(n_grid=a.n_grid, grid_lim=2.0, device="cuda:0")
    sol = MPMWARP(n, ne, nv, n_grid=a.n_grid, grid_lim=2.0, num_joint_t=0, num_joint_v=nj, num_joint_f=njf, device="cuda:0")
    sol.set_parameters_dict(md, st, {"material": "sand", "g": [0.0, 0.0, -9.8], "density": 1.0, "grid_v_damping_scale": 1.1,
                                     "friction_angle": 40.0})
    one = torch.ones(n, device=dev)
    st.reset_density((one * D_).clone(), one.int(), "cuda:0", update_mass=True)
    sol.set_E_nu_from_torch(md, one * E_, one * NU, one * GAM, one * KAP, "cuda:0")
    sol.prepare_mu_lam(md, st, "cuda:0")
    sol.add_surface_collider([0.0, 0.0, 0.05], [0.0, 0.0, 1.0])
    sol.add_particle_mover(n_grid=md.n_grid)
    rv = verts.clone(); R_inv = rinv(torch.stack([rv[:, 0], rv[:, 1], rv[:, 2] * Hs], 1), faces)   # 저자 규약: 위쪽 축에 H (아바타는 y 위)
    vel = torch.zeros_like(pos0)
    if a.mode == "a":
        vel[:, 1] = a.vel * SC
        vel[ne:ne + nj] = 0.0
        vel[:njf] = 0.0
    st.reset_state(nv, pos0.clone(), d0.clone(), None, vel.clone(), tensor_R_inv=R_inv.clone(), device="cuda:0", requires_grad=False)
    st.reset_density((one * D_).clone(), None, "cuda:0", update_mass=True)
    sol.set_E_nu_from_torch(md, one * E_, one * NU, one * GAM, one * KAP, "cuda:0")
    sol.prepare_mu_lam(md, st, "cuda:0")
    sub = (1.0 / 25) / a.substep
    jv = torch.zeros(nj, 3, device=dev); jf = torch.zeros(njf, 3, device=dev)
    traj = [verts.cpu()]
    tr_ok = True; X0 = pos0.clone(); V0n = a.vel * SC

    def trace(f, s):                                               # 서브스텝 상태 요약. 처음 터진 입자의 정체를 찍는다
        x = wp.to_torch(st.particle_x); v = wp.to_torch(st.particle_v); Fm = wp.to_torch(st.particle_F)[:ne]
        dm = wp.to_torch(st.particle_d)[:ne]; sp = v.norm(dim=1)
        fn = Fm.reshape(ne, 9).norm(dim=1); dn = dm[:, :, 2].norm(dim=1); det = torch.linalg.det(Fm.double()).float()
        i = int(sp.argmax()); kind = "면" if i < ne else ("고정꼭짓점" if i < ne + nj else "꼭짓점")
        p = ((X0[i].double().cpu().numpy() - SHIFT) / SC)
        bad = (~torch.isfinite(sp)).any() or float(sp.max()) > 20 * V0n
        gm = wp.to_torch(st.grid_m); dxs = GLIM / a.n_grid

        def stencil_m(q):                                          # 입자 하나가 보는 27 격자점 질량 합 (g2p 와 같은 받침)
            b = (q / dxs - 0.5).floor().long()
            return float(gm[b[0]:b[0] + 3, b[1]:b[1] + 3, b[2]:b[2] + 3].sum())
        vi = i if i >= ne else int(torch.as_tensor(Fr[i], device=dev)[0]) + ne
        xs = x[ne:]; dd = (xs - x[vi]).norm(dim=1); rd = (X0[ne:] - X0[vi]).norm(dim=1)
        far = rd > 4 * (W_ / NX) * SC                                 # 쉬던 모양에서 네 칸 넘게 떨어진 꼭짓점
        near = float(dd[far].min()) / dxs if far.any() else float("nan")
        med = float(torch.tensor([stencil_m(xs[k]) for k in range(0, nv, max(1, nv // 200))]).median())
        print(f"   [모서리] 격자질량 {stencil_m(x[vi]):.3g} (꼭짓점 중앙 {med:.3g}), 안 이웃 최근접 {near:.2f} 칸, "
              f"이웃 꼭짓점 속도 평균 {float(sp[ne:][(rd < 1.5 * (W_ / NX) * SC)].mean()):.3g}", flush=True)
        print(f"[추적] f{f} s{s} |v|max {float(sp.max()):.3g} ({kind} {i}, 처음 위치 x{p[0]:.2f} z{p[2]:.2f}) "
              f"|F|max {float(fn.max()):.3g} det min {float(det.min()):.3g} max {float(det.max()):.3g} |d3|max {float(dn.max()):.3g} "
              f"z min {float(x[:, 2].min()):.3f} y max {float(x[:, 1].max()):.3f}", flush=True)
        if bad:
            j = int(fn.argmax()); q = ((X0[j].double().cpu().numpy() - SHIFT) / SC)
            print(f"[터짐] f{f} s{s}  |F| 최대 면 {j} 처음 위치 x{q[0]:.2f} z{q[2]:.2f}, F=\n{Fm[j].cpu().numpy()}\n d=\n{dm[j].cpu().numpy()}", flush=True)
            return None
        return True
    for f in tqdm(range(a.frames), desc=f"MPMAvatar {a.mode}"):
        for s in range(a.substep):
            sol.p2g2p(md, st, sub, joint_traditional_v=None, joint_verts_v=jv, joint_faces_v=jf, device="cuda:0")
            if 0 <= a.trace <= f and (s % a.trace_every == 0 or not tr_ok):
                tr_ok = trace(f, s)
                if tr_ok is None:
                    break
        if 0 <= a.trace <= f and tr_ok is None:
            break
        x = wp.to_torch(st.particle_x)[ne:].clone()
        if not torch.isfinite(x).all():
            print(f"[발산] 프레임 {f + 1}", flush=True); break
        traj.append(x.cpu())
    T = torch.stack(traj)[:, inv]                                                 # 원래 꼭짓점 순서로
    torch.save(dict(V=(T.double() - torch.as_tensor(SHIFT)) / SC), f"{a.work}/mpma_{a.mode}.pt")
    print(f"[MPMAvatar {a.mode}] {T.shape[0]} 프레임", flush=True)

elif a.stage == "pg":
    PG = "/home/dkta/work/i-physgaussian"; sys.path.insert(0, PG); os.chdir(PG)
    import warp as wp
    from tqdm import tqdm
    from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP
    P = torch.load(f"{a.work}/prep.pt")
    cl = ~P["pole"]
    X = (P["X"][cl].double().numpy() * SC + SHIFT)
    Cv = P["C"][cl].double().numpy() * SC * SC
    x = torch.as_tensor(X, device=dev).float().contiguous(); n = x.shape[0]; dx = 2.0 / a.n_grid
    cell = (x / dx).floor().long(); key = (cell[:, 0] * a.n_grid + cell[:, 1]) * a.n_grid + cell[:, 2]
    _, inv_, cnt = torch.unique(key, return_inverse=True, return_counts=True)
    vol = (dx ** 3 / cnt[inv_].float()).contiguous()
    c6 = torch.as_tensor(np.stack([Cv[:, 0, 0], Cv[:, 0, 1], Cv[:, 0, 2], Cv[:, 1, 1], Cv[:, 1, 2], Cv[:, 2, 2]], 1), device=dev).float()
    if os.environ.get("WARP_KCACHE"):                          # 동시에 여러 실행이 같은 캐시를 컴파일하다 깨지는 것을 막는다
        wp.config.kernel_cache_dir = os.environ["WARP_KCACHE"]
    wp.init()
    sol = MPM_Simulator_WARP(10)
    sol.load_initial_data_from_torch(x, vol, c6, n_grid=a.n_grid, grid_lim=2.0)
    sol.set_parameters_dict({"material": "jelly", "E": E_, "nu": NU, "density": D_, "g": [0.0, 0.0, -9.8],
                             "n_grid": a.n_grid, "grid_lim": 2.0, "grid_v_damping_scale": 1.1})
    sol.add_bounding_box(); sol.add_surface_collider((0.0, 0.0, 0.05), (0.0, 0.0, 1.0), "sticky", 0.0)
    if a.pin_grid:                                                 # 깃대 쪽 띠 (월드 x 0~한 칸, 천 높이 전체) 의 격자 속도 0
        lo = np.array([-0.005, -0.03, Z0 - 0.01]) * SC + SHIFT; hi = np.array([W_ / NX * 1.01, 0.03, Z0 + H_ + 0.01]) * SC + SHIFT
        sol.set_velocity_on_cuboid(tuple(((lo + hi) / 2).tolist()), tuple(((hi - lo) / 2).tolist()), (0.0, 0.0, 0.0))
    sol.finalize_mu_lam()
    pin = torch.as_tensor(P["X"][cl][:, 0].numpy() < W_ / NX * 1.01, device=dev)        # 깃대 쪽 첫 칸
    v0 = torch.zeros_like(x)
    if a.mode == "a":
        v0[:, 1] = a.vel * SC; v0[pin] = 0.0
    sol.import_particle_v_from_torch(v0.contiguous())
    pinw = wp.from_torch(pin.int().contiguous()); x0 = wp.from_torch(x.contiguous(), dtype=wp.vec3)

    @wp.kernel
    def hold(px: wp.array(dtype=wp.vec3), pv: wp.array(dtype=wp.vec3), x0: wp.array(dtype=wp.vec3), m: wp.array(dtype=int)):
        p = wp.tid()
        if m[p] == 1:
            px[p] = x0[p]; pv[p] = wp.vec3(0.0, 0.0, 0.0)

    sub = (1.0 / 25) / a.substep
    XS, FS = [x.cpu()], [torch.eye(3).expand(n, 3, 3).clone()]
    print(f"[PG {a.mode}] 입자 {n} (고정 {int(pin.sum())}), dx {dx:.4f}", flush=True)
    for f in tqdm(range(a.frames), desc=f"PG {a.mode}"):
        for s in range(a.substep):
            wp.launch(hold, dim=n, inputs=[sol.mpm_state.particle_x, sol.mpm_state.particle_v, x0, pinw])
            sol.p2g2p(f * a.substep + s, sub)
        xt = sol.export_particle_x_to_torch().clone()
        if not torch.isfinite(xt).all():
            print(f"[발산] 프레임 {f + 1}", flush=True); break
        XS.append(xt.cpu()); FS.append(sol.export_particle_F_to_torch().clone().cpu().reshape(-1, 3, 3))
    torch.save(dict(X=(torch.stack(XS).double() - torch.as_tensor(SHIFT)) / SC, F=torch.stack(FS)), f"{a.work}/pg_{a.mode}.pt")
    print(f"[PG {a.mode}] {len(XS)} 프레임", flush=True)

else:
    sys.path.insert(0, "/home/dkta/work/i-physgaussian/gaussian-splatting"); sys.path.insert(0, "/home/dkta/work/i-physgaussian")
    import imageio.v2 as imageio
    from utils.graphics_utils import getWorld2View2, getProjectionMatrix
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    P = torch.load(f"{a.work}/prep.pt")
    cl = ~P["pole"]
    Xp, Cp = P["X"][P["pole"]].float().to(dev), P["C"][P["pole"]].float().to(dev)
    C0 = P["C"][cl].float().to(dev)
    SH = torch.cat([P["shs"][cl], P["shs"][P["pole"]]]).float().to(dev)
    OP = torch.cat([P["op"][cl], P["op"][P["pole"]]]).float().to(dev)[:, None]
    Fm = P["F"].long().to(dev); tri, bary, off = P["tri"].to(dev), P["bary"].to(dev), P["off"].to(dev)
    T0 = tri_frame(P["V"].float().to(dev), Fm)
    R = torch.load(f"{a.work}/{a.solver}_{a.mode}.pt")
    # 카메라: 깃발 정면에서 비스듬히 (깃발이 +y 로 휘는 것이 보이게)
    ctr = np.array([W_ / 2, 0.0, POLE * 0.62]); eye = ctr + np.array([0.9, -3.2, 0.6])
    f_ = (ctr - eye) / np.linalg.norm(ctr - eye); up = np.array([0, 0, 1.0]); r_ = np.cross(f_, up); r_ /= np.linalg.norm(r_); u_ = np.cross(r_, f_)
    c2w = np.eye(4); c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = r_, -u_, f_, eye
    W2C = np.linalg.inv(c2w); fov = 0.69
    wv = torch.tensor(getWorld2View2(W2C[:3, :3].T, W2C[:3, 3])).transpose(0, 1).to(dev).float()
    pj = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fov, fovY=fov).transpose(0, 1).to(dev).float()
    rast = GaussianRasterizer(raster_settings=GaussianRasterizationSettings(
        image_height=a.res, image_width=a.res, tanfovx=math.tan(fov / 2), tanfovy=math.tan(fov / 2), bg=torch.ones(3, device=dev),
        scale_modifier=1.0, viewmatrix=wv, projmatrix=(wv[None] @ pj[None])[0], sh_degree=3, campos=wv.inverse()[3, :3],
        prefiltered=False, debug=False))
    nfr = R["V"].shape[0] if a.solver == "mpma" else R["X"].shape[0]
    out = f"{a.work}/{a.solver}_{a.mode}.mp4"
    WR = imageio.get_writer(out, fps=25, codec="libx264", quality=8)
    for t in range(nfr):
        if a.solver == "mpma":
            V = R["V"][t].float().to(dev)
            Tt = tri_frame(V, Fm)
            Ft = Tt @ torch.linalg.inv(T0)                                # 삼각형 변형 (면 안 두 변 + 법선)
            Xc = (bary[:, :, None] * V[Fm[tri]]).sum(1) + off[:, None] * Tt[tri][..., 2]
            Cc = Ft[tri] @ C0 @ Ft[tri].transpose(1, 2)
        else:
            Xc = R["X"][t].float().to(dev); Fg = R["F"][t].float().to(dev)
            Cc = Fg @ C0 @ Fg.transpose(1, 2)
        X = torch.cat([Xc, Xp]); C = torch.cat([Cc, Cp])
        c6 = torch.stack([C[:, 0, 0], C[:, 0, 1], C[:, 0, 2], C[:, 1, 1], C[:, 1, 2], C[:, 2, 2]], 1)
        ok = torch.isfinite(X).all(1) & torch.isfinite(c6).all(1)
        img = rast(means3D=X[ok], means2D=torch.zeros_like(X[ok]), shs=SH[ok], colors_precomp=None, opacities=OP[ok],
                   scales=None, rotations=None, cov3D_precomp=c6[ok])[0]
        WR.append_data((img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    WR.close()
    print(f"[영상] {out} ({nfr} 프레임)", flush=True)
