"""절 AK.20 (구조 변경, 사용자 허락 "같은 격자 간격의 2 차 B-스플라인 요소"): 같은 Kuhn 격자 (월드 고정 회전 정육면체 격자, 간격 node_h/√2 = 0.02 m = 정답 MPM dx) 의 노드에
2 차 B-스플라인 기저 (정답 MPM 과 같은 기저, C1) 를 둔 요소 판. 틀은 절 AK 그대로: 프레임마다 P2G 한 번 (노드 질량 Σ w m_p, 운동량 Σ w m_p v_p) -> 명시적 서브스텝 N 번
(노드 변위 U 만 움직이고, 입자 기저 값·기울기는 프레임 시작 위치로 고정 = 프레임 안 라그랑주) -> G2P 한 번 (x += Σ w U, v = Σ w v_n, F ← A_p F, A_p = I + Σ U ⊗ ∇w).
셀 탄성 = 입자 적분점 Σ_p ψ(A_p F_p) (전단 몫 가중 V_p min(1, θ V_칸 / Σ_칸 V_p), 부피 몫 V_p -- 기본 판 --ak_capmu 의 칸 = 격자 정육면체), 표준 FCR, 클램프 없음.
노드 가속도에만 국소 CFL 관성 질량 m' = max(m, (h_s/2)² k_n), k_n = Σ_p V_p (λ+2μ) |∇w_np| Σ_j |∇w_jp| (절 AG.15 와 같은 Gershgorin 상계; 지지 가장자리의 거의 0 질량 노드).
바닥 = 절 AH.13 띠 0 판과 같은 입자 판정 (s = 현재 입자 높이 < 0 인 입자의 정답 sticky 가중 w(s)), 고착 질량 Σ w_np m_p w(s), 단위 시간당 고착 (1 − a/m)^(h_s/h_ref), 비관통 깊이 npd.
torch 판 (검증용, 느림 -- 측정 아님). 출력: <out>/fem.jsonl (프레임별 오차·몸 높이·입자 탄성 우리/정답), summary.json."""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import torch                                                                         # noqa: E402
from anchorflow import deform_model as DM                                           # noqa: E402
from anchorflow import fem_rt as RT                                                 # noqa: E402
from anchorflow import phys_resid as PR                                             # noqa: E402
from anchorflow import simplex as SX                                                # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--frames", type=int, default=80)
ap.add_argument("--node_h", type=float, default=0.028284)
ap.add_argument("--N", type=int, required=True)
ap.add_argument("--floor", choices=["vel", "none"], default="vel")
ap.add_argument("--theta", type=float, default=0.5, help="전단 몫 부피 상한 θ (기본 판 --ak_capmu 와 같은 뜻; 0 이하 = 상한 없음)")
ap.add_argument("--npd", type=float, default=0.005)
ap.add_argument("--cfg_bc", action="store_true")
ap.add_argument("--region_density", action="store_true")
ap.add_argument("--pg_vdamp", action="store_true")
ap.add_argument("--no_mcfl", action="store_true", help="국소 CFL 질량 끔 (안정 걸음 확인용)")
ap.add_argument("--dump", action="store_true", help="위치 덤프 (pred.pt -- 진동 주기용)")
ap.add_argument("--tag", default="")
a = ap.parse_args()
dev = "cuda"
os.makedirs(a.out, exist_ok=True)
d = torch.load(a.data, map_location="cpu", weights_only=False)
cfg = d["cfg"]
h = float(cfg["frame_dt"])
hs = h / a.N
DM.REGION_DENSITY[0] = bool(a.region_density)
m = DM.traj_mass(d, dev).double()
EXT = float(d["_ext"])
rho = (d["_rho"].to(dev).double()) if a.region_density else None
vol = (m / rho) if a.region_density else (m / float(cfg["density"]))
mu, lam = PR.lame(cfg["E"], cfg["nu"])
g = torch.tensor([float(q) for q in cfg["g"]], device=dev, dtype=torch.float64)
Np = int(d["x"].shape[1])
L = min(a.frames, int(d["x"].shape[0]) - 1)
x = d["x"][0].to(dev).double()
v = d["v"][0].to(dev).double()
F = DM.traj_F(d, h, dev)[0].to(dev).double().reshape(Np, 3, 3)
GF = DM.traj_F(d, h, "cpu")
VD_S = float(cfg.get("grid_v_damping_scale", 1.0))
DAMP = (VD_S ** (hs / float(cfg["substep_dt"]))) if (a.pg_vdamp and VD_S < 1.0) else 1.0
lat = SX.lattice(a.node_h, SX.HZ_CUBE * a.node_h, device=dev, dtype=torch.float64)
A_, Ai = lat.A, lat.Ai
dxl = float(A_[:, 0].norm())
FLOOR = a.floor == "vel"
if FLOOR:
    pl = [b for b in cfg["boundary_conditions"] if b.get("type") == "surface_collider"][0]
    pt = torch.tensor([float(q) for q in pl["point"]], device=dev, dtype=torch.float64)
    nr = torch.tensor([float(q) for q in pl["normal"]], device=dev, dtype=torch.float64)
    nr = nr / nr.norm()
    PN = float(pt @ nr)
    DXG = float(cfg.get("grid_lim", 2.0)) / int(cfg.get("n_grid", 100))
    KEX = hs / float(cfg["substep_dt"])
BC_SEL = RT.cfg_bc_sel(cfg, x.float()) if a.cfg_bc else None
cwave = math.sqrt((lam + 2 * mu) / float((rho.min() if rho is not None else cfg["density"])))
print(f"[절 AK.20 B-스플라인 요소] 입자 {Np}, 프레임 {L}, N {a.N} (h_s {hs:.3e}), 격자 간격 {dxl:.5f} m, 전역 CFL dx/c {dxl / cwave:.3e} s (h_s 비 {hs / (dxl / cwave):.3f}), "
      f"θ {a.theta}, 국소 CFL 질량 {'끔' if a.no_mcfl else '켬'}, 바닥 {a.floor}, 감쇠 {DAMP:.6f}", flush=True)
OFF = torch.stack(torch.meshgrid(torch.arange(3), torch.arange(3), torch.arange(3), indexing="ij"), -1).reshape(27, 3).to(dev)


def wstk(s):
    u = (s + PN) / DXG
    bf = torch.floor(u - 0.5)
    fx = u - bf
    w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
    o = torch.zeros_like(s)
    for i in range(3):
        o = o + torch.where((bf + i) * DXG - PN < 0, w[i], torch.zeros_like(s))
    return o


def cof(X):
    return torch.stack([torch.cross(X[:, 1], X[:, 2], dim=-1), torch.cross(X[:, 2], X[:, 0], dim=-1), torch.cross(X[:, 0], X[:, 1], dim=-1)], 1)


def fcr_parts(X):
    U_, S_, Vh_ = torch.linalg.svd(X)
    dsg = torch.sign(torch.linalg.det(U_ @ Vh_))
    D = torch.ones(X.shape[0], 3, device=dev, dtype=X.dtype)
    D[:, 2] = dsg
    R = U_ @ torch.diag_embed(D) @ Vh_
    J = torch.linalg.det(X)
    return R, J, cof(X)


def psi(X):
    R, J, _ = fcr_parts(X)
    return mu * ((X - R) ** 2).sum((1, 2)) + 0.5 * lam * (J - 1) ** 2


x0 = x.clone()
q0 = x0 - x0.mean(0)


def bheight(xx):
    c = xx.mean(0)
    U_, S_, Vh_ = torch.linalg.svd(q0.T @ (xx - c))
    Dg = torch.eye(3, device=dev, dtype=torch.float64)
    Dg[2, 2] = torch.sign(torch.det(Vh_.T @ U_.T))
    b = (xx - c) @ (Vh_.T @ Dg @ U_.T)
    return float(b[:, 2].max() - b[:, 2].min())


flog = open(os.path.join(a.out, "fem.jsonl"), "w")
errs, recs = [], []
PRED = torch.empty(L, Np, 3) if a.dump else None
div_at = None
t_all = time.time()
for fi in range(L):
    # ---- 프레임 시작: 기저 (프레임 시작 위치 고정), 노드 번호, P2G ----
    y = x @ Ai.T / 1.0                                                               # 격자 좌표 (간격 1)
    base = torch.floor(y - 0.5).long()
    fx = y - base
    w1 = torch.stack([0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2], 1)        # [Np, 3 (i), 3 (축)]
    d1 = torch.stack([fx - 1.5, -2 * (fx - 1), fx - 0.5], 1)
    wi, wj, wk = w1[:, OFF[:, 0], 0], w1[:, OFF[:, 1], 1], w1[:, OFF[:, 2], 2]
    W = wi * wj * wk                                                                  # [Np, 27]
    gl_ = torch.stack([d1[:, OFF[:, 0], 0] * wj * wk, wi * d1[:, OFF[:, 1], 1] * wk, wi * wj * d1[:, OFF[:, 2], 2]], -1)   # 격자 좌표 기울기 [Np,27,3]
    GW = gl_ @ Ai                                                                     # 월드 기울기 (∂y/∂x = Ai) [Np,27,3]
    keys = base[:, None, :] + OFF[None]
    kmin = keys.reshape(-1, 3).min(0).values
    kk = keys - kmin
    B = int(kk.max()) + 1
    flat = ((kk[..., 0] * B + kk[..., 1]) * B + kk[..., 2]).reshape(-1)
    uq, nid = torch.unique(flat, return_inverse=True)
    M = int(uq.numel())
    nid = nid.view(Np, 27)
    mn = torch.zeros(M, device=dev, dtype=torch.float64).index_add_(0, nid.reshape(-1), (W * m[:, None]).reshape(-1))
    mv = torch.zeros(M, 3, device=dev, dtype=torch.float64).index_add_(0, nid.reshape(-1), (W[..., None] * (m[:, None] * v)[:, None, :]).reshape(-1, 3))
    vn = torch.where(mn[:, None] > 0, mv / mn.clamp_min(1e-30)[:, None], torch.zeros_like(mv))
    # 전단 몫 가중 (칸 = 격자 정육면체 floor(y))
    cell = torch.floor(y).long()
    cmx = int((cell - cell.min(0).values).max()) + 1
    cf_ = ((cell - cell.min(0).values)[:, 0] * cmx + (cell - cell.min(0).values)[:, 1]) * cmx + (cell - cell.min(0).values)[:, 2]
    cu, cinv = torch.unique(cf_, return_inverse=True)
    vc = torch.zeros(cu.numel(), device=dev, dtype=torch.float64).index_add_(0, cinv, vol)
    if a.theta > 0:
        wmu = vol * torch.clamp(a.theta * dxl ** 3 / vc[cinv], max=1.0)
    else:
        wmu = vol.clone()
    # 국소 CFL 관성 질량
    gn = GW.norm(dim=-1)                                                              # [Np,27]
    kn = torch.zeros(M, device=dev, dtype=torch.float64).index_add_(0, nid.reshape(-1), (vol[:, None] * (lam + 2 * mu) * gn * gn.sum(1, keepdim=True)).reshape(-1))
    mI = mn if a.no_mcfl else torch.maximum(mn, (hs / 2) ** 2 * kn)
    fixd = None
    if BC_SEL is not None:
        kq = torch.stack([uq // (B * B), (uq // B) % B, uq % B], -1) + kmin
        npos = kq.double() @ A_.T
        fx_, vb_ = RT.cfg_bc_nodes(cfg, [], npos.float(), torch.zeros(1, 4, dtype=torch.long, device=dev), torch.zeros(1, 4, device=dev), m[:1].float(), fi * h, h)
        if fx_ is not None:
            fixd, vbc = fx_.bool(), vb_.double()
    U = torch.zeros(M, 3, device=dev, dtype=torch.float64)
    if FLOOR:
        s0 = (x - pt) @ nr
        cand = torch.nonzero(s0 < (v.norm(dim=-1).max() * h + 0.05)).flatten()
    # ---- 서브스텝 ----
    try:
        for si in range(a.N):
            Up = U[nid]                                                               # [Np,27,3]
            Ap = torch.eye(3, device=dev, dtype=torch.float64) + torch.einsum("pka,pkb->pab", Up, GW)
            X = Ap @ F
            R, J, C = fcr_parts(X)
            P = 2 * mu * (X - R) * (wmu / vol)[:, None, None] + lam * (J - 1)[:, None, None] * C
            G = (vol[:, None, None] * P) @ F.transpose(1, 2)                           # [Np,3,3]
            fp = -torch.einsum("pab,pkb->pka", G, GW)                                 # [Np,27,3]
            f = torch.zeros(M, 3, device=dev, dtype=torch.float64).index_add_(0, nid.reshape(-1), fp.reshape(-1, 3))
            vn = torch.where(mn[:, None] > 0, (vn + hs * (f / mI.clamp_min(1e-30)[:, None] + g)) * DAMP, torch.zeros_like(vn))
            if fixd is not None:
                vn[fixd] = vbc[fixd]
            if FLOOR and cand.numel():
                xc = x[cand] + (W[cand, :, None] * Up[cand]).sum(1)
                s = (xc - pt) @ nr
                inn = s < 0
                if bool(inn.any()):
                    ci = cand[inn]
                    wst = wstk(s[inn])
                    acc = torch.zeros(M, device=dev, dtype=torch.float64).index_add_(0, nid[ci].reshape(-1), (W[ci] * (m[ci] * wst)[:, None]).reshape(-1))
                    st = acc > 0
                    k = torch.clamp(1 - acc / mn.clamp_min(1e-30), min=0) ** KEX
                    vn = torch.where(st[:, None], vn * k[:, None], vn)
                    real = s[inn] < -a.npd
                    if bool(real.any()):
                        accN = torch.zeros(M, device=dev, dtype=torch.float64).index_add_(0, nid[ci[real]].reshape(-1), torch.ones_like(W[ci[real]]).reshape(-1))
                        vnn = vn @ nr
                        fixn = (accN > 0) & (vnn < 0)
                        vn = torch.where(fixn[:, None], vn - vnn[:, None] * nr[None], vn)
            U = U + hs * vn
            if not bool(torch.isfinite(vn).all()):
                raise FloatingPointError(f"서브스텝 {si}")
    except FloatingPointError as ex_:
        div_at = fi + 1
        print(f"[발산] 프레임 {div_at} {ex_}", flush=True)
        break
    # ---- G2P ----
    Up = U[nid]
    x2 = x + (W[..., None] * Up).sum(1)
    v2 = (W[..., None] * vn[nid]).sum(1)
    Ap = torch.eye(3, device=dev, dtype=torch.float64) + torch.einsum("pka,pkb->pab", Up, GW)
    F2 = Ap @ F
    gt = d["x"][fi + 1].to(dev).double()
    e = float((x2 - gt).norm(dim=-1).mean()) / EXT
    fin = bool(torch.isfinite(x2).all())
    rec = dict(t=fi, err=e, M=M)
    if (fi + 1) % 2 == 0 or fi < 2:
        rec.update(bh=[bheight(x2), bheight(gt)], ep=[float((vol * psi(F2)).sum()), float((vol * psi(GF[fi + 1].to(dev).double().reshape(Np, 3, 3))).sum())])
    flog.write(json.dumps(rec) + "\n")
    flog.flush()
    errs.append(e)
    if PRED is not None:
        PRED[fi] = x2.float().cpu()
    print(f"  프레임 {fi + 1}/{L} 오차 {100 * e:.3f}% 노드 {M} {time.time() - t_all:.0f}s", flush=True)
    x, v, F = x2, v2, F2
    if not fin or e > 2.0:
        div_at = fi + 1
        print(f"[발산] 프레임 {div_at}", flush=True)
        break
summ = dict(args=vars(a), frames=L, frames_done=len(errs), diverged=div_at, err_mean=(sum(errs) / len(errs) if div_at is None else None),
            err_mean_done=sum(errs) / max(len(errs), 1), hs=hs, cfl_ratio=hs / (dxl / cwave), sec=time.time() - t_all)
json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1, ensure_ascii=False)
if PRED is not None and div_at is None:
    torch.save({"pred": PRED}, os.path.join(a.out, "pred.pt"))
print(f"[롤아웃] {a.tag} B-스플라인 요소 N {a.N}: {len(errs)}/{L} 프레임{' (발산 %d)' % div_at if div_at else ''}, 평균 {100 * summ['err_mean_done']:.3f}%, {summ['sec']:.0f}s", flush=True)
