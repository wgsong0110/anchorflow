"""명시적 FEM 단독 시뮬레이터 (사용자 10-28 지시, doc/fem_realtime.md 절 L) -- lib/anchorflow/fem_ex_cuda.py.

암시적 실시간 판 (exe/fem_rt_sim.py) 과 같은 모델 (사면체 적분점 F̄_T, 고정 FCR, 같은 관성·중력·감쇠, 입자 바닥 stick·softplus 매핑,
--cfg_bc·--region_density·--pg_vdamp 같은 정의) 을 뉴턴 대신 symplectic Euler 서브스텝 N 번으로 (h_s = h/N). 식은 fem_ex_cuda 머리말.

  --mode sub : 매 서브스텝 P2G·G2P (MPM 식)
  --mode lag : 프레임 동안 격자 위 라그랑주식 (U 누적, 셀 F = A_T(U) F̄_T), 바닥 후보 입자만 서브스텝마다 보간·매핑 갱신, 프레임 끝 G2P
  --opt      : 최적화 판 -- 셀 점유 격자 (호스트 동기화 프레임당 2 번), P2G·분할·F̄ 커널, 탄성 힘 float32 대수, 크기를 장치에서 읽는 커널로
               서브스텝 N 번 전체를 CUDA 그래프 하나로 (버퍼 용량 미리 할당, 넘치면 늘리고 다시 잡는다), 저장은 측정 밖
  (없으면) 기준 판 -- fem_rt 격자 (GridK) + CudaTetqK (암시적 판과 같은 P2G·분할, float64) + 같은 서브스텝 커널 float64 대수, 서브스텝마다 파이썬 루프
오차 = 프레임마다 입자 평균 |x - x_gt| / EXT 의 L 프레임 평균 (fem_rt_sim 과 같은 지표). 발산 = 값이 유한하지 않거나 바운딩박스 대각/EXT > --div 이면 그 프레임에서 멈춘다.
출력 <out>/: fem.jsonl, summary.json, roll_fem.pt (fem_rt_sim 과 같은 형식).
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import numpy as np                                                                  # noqa: E402
import torch                                                                        # noqa: E402
from anchorflow import deform_model as DM                                           # noqa: E402
from anchorflow import fem_rt as RT                                                 # noqa: E402
from anchorflow import fem_rt_cuda as RTC                                           # noqa: E402
from anchorflow import fem_ex_cuda as EX                                            # noqa: E402
from anchorflow import phys_resid as PR                                             # noqa: E402
from anchorflow import simplex as SX                                                # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/home/dkta/work/one/L000/L000.pt")
ap.add_argument("--out", required=True)
ap.add_argument("--frames", type=int, default=80)
ap.add_argument("--node_h", type=float, default=0.028284)
ap.add_argument("--floor", choices=["particle", "none"], default="particle")
ap.add_argument("--cfg_bc", action="store_true")
ap.add_argument("--region_density", action="store_true")
ap.add_argument("--pg_vdamp", action="store_true")
ap.add_argument("--tag", default="L000")
ap.add_argument("--eps_sp", type=float, default=0.01)
ap.add_argument("--eps_st", type=float, default=0.01)
ap.add_argument("--mode", choices=["sub", "lag"], default="lag")
ap.add_argument("--N", type=int, required=True, help="프레임당 서브스텝 수")
ap.add_argument("--opt", action="store_true")
ap.add_argument("--dil", type=int, default=-1, help="--opt --mode sub: 셀 팽창 (셀 수; -1 = 프레임 최대 이동으로 자동)")
ap.add_argument("--no_graph", action="store_true", help="--opt 에서 CUDA 그래프 끄기 (대조)")
ap.add_argument("--div", type=float, default=3.0)
ap.add_argument("--warm", type=int, default=2)
ap.add_argument("--no_dump", action="store_true")
ap.add_argument("--dump_f32", action="store_true")
ap.add_argument("--dump_pred", action="store_true")
ap.add_argument("--diag", action="store_true", help="첫 프레임: 노드별 국소 안정 걸음 2/ω (ω² = Σ_T V_T (λ+2μ)|∇λ_i|² / m_i) 분포, 서브스텝마다 최대 |v|")
a = ap.parse_args()

dev = "cuda"
os.makedirs(a.out, exist_ok=True)
d = torch.load(a.data, map_location="cpu", weights_only=False)
cfg = d["cfg"]
h = float(cfg["frame_dt"])
hs = h / a.N
DM.REGION_DENSITY[0] = bool(a.region_density)
m = DM.traj_mass(d, dev).float().contiguous()
EXT = float(d["_ext"])
rho = (d["_rho"].to(dev).float()) if a.region_density else None
vol = ((m / rho) if a.region_density else (m / float(cfg["density"]))).contiguous()
BC_SEL = RT.cfg_bc_sel(cfg, d["x"][0].to(dev).float()) if a.cfg_bc else None
if a.cfg_bc and any(bc.get("type") in ("enforce_particle_translation", "enforce_particle_velocity_rotation") for bc, _ in BC_SEL) and a.opt:
    raise SystemExit("--opt --cfg_bc: 입자 구속은 아직 (cuboid 만)")
VD_S = float(cfg.get("grid_v_damping_scale", 1.0))
DAMP = (VD_S ** (hs / float(cfg["substep_dt"]))) if (a.pg_vdamp and VD_S < 1.0) else 1.0   # 서브스텝당 (프레임 합 = fem_rt_sim 의 VDAMP)
Np = int(d["x"].shape[1])
L = min(int(a.frames), int(d["x"].shape[0]) - 1)
x = d["x"][0].to(dev).float().contiguous()
v = d["v"][0].to(dev).float().contiguous()
F = DM.traj_F(d, h, dev)[0].to(dev).float().reshape(Np, 3, 3).contiguous()
GT = d["x"][1:L + 1].to(dev).float()
gv = [float(q) for q in cfg["g"]]
mu, lam = PR.lame(cfg["E"], cfg["nu"])
planes = [b for b in (cfg.get("boundary_conditions") or []) if b.get("type") == "surface_collider"]
FLOOR = a.floor == "particle" and bool(planes)
if a.floor == "particle" and not planes:
    raise SystemExit("--floor particle 인데 surface_collider 가 없다")
if FLOOR:
    pt_ = [float(q) for q in planes[0]["point"]]
    nr_ = np.array(planes[0]["normal"], dtype=np.float64)
    nr_ = (nr_ / np.linalg.norm(nr_)).tolist()
    PL = [np.float64(q) for q in pt_ + nr_]
    PLf = [np.float32(q) for q in pt_ + nr_]
else:
    PL = [np.float64(0)] * 6
    PLf = [np.float32(0)] * 6
XN = 1.0 if a.eps_st > 0 else 0.0
# CFL: 파속 √((λ+2μ)/ρ) (가장 가벼운 밀도), 길이 = Kuhn 사면체 최소 높이 (첫 격자에서)
rho_min = float(rho.min()) if rho is not None else float(cfg["density"])
cwave = math.sqrt((lam + 2.0 * mu) / rho_min)
G0 = RTC.GridK(x, a.node_h)
lmin = EX.kuhn_min_altitude(G0.gl8)
dt_cfl = lmin / cwave
del G0
print(f"[명시적 FEM] 입자 {Np}, 프레임 {L}, h {h:.6f}, 서브스텝 {a.N} (h_s {hs:.3e}), 판 {a.mode} {'최적화' if a.opt else '기준'}, E {cfg['E']:g} ν {cfg['nu']:g} "
      f"(μ {mu:.4g} λ {lam:.4g}), EXT {EXT:.6f}, 바닥 {'입자 (stick·softplus)' if FLOOR else '없음'}, 감쇠 {DAMP:.8f}/서브스텝", flush=True)
print(f"[CFL] 파속 √((λ+2μ)/ρ_min) = {cwave:.2f} m/s (ρ_min {rho_min:g}), Kuhn 최소 높이 {lmin:.5f} (node_h {a.node_h}), "
      f"Δt_CFL = {dt_cfl:.3e} s -> N_CFL = h/Δt_CFL = {h / dt_cfl:.1f} (지금 h_s/Δt_CFL = {hs / dt_cfl:.3f})", flush=True)
print(f"[명령] {' '.join(sys.argv)}", flush=True)

I3 = torch.eye(3, device=dev)


def fixd_of(npos, rows_, lam_, now):
    if BC_SEL is None:
        return None, None
    fx, vb = RT.cfg_bc_nodes(cfg, BC_SEL, npos, rows_, lam_, m, now, h)
    if fx is None:
        return None, None
    return fx.to(torch.uint8).contiguous(), vb.float().contiguous()


def node_mv(G, x_, v_):
    w = G.lam * m.unsqueeze(-1)
    ws = torch.zeros(G.M, device=dev).index_add_(0, G.rows.reshape(-1), w.reshape(-1))
    vs = torch.zeros(G.M, 3, device=dev).index_add_(0, G.rows.reshape(-1), (w.unsqueeze(-1) * v_.unsqueeze(1)).reshape(-1, 3))
    return ws.contiguous(), (vs / ws.clamp_min(1e-30).unsqueeze(-1)).contiguous()


def near_args(n_near):
    return PL + [np.float64(a.eps_st), np.float64(a.eps_sp), np.float64(XN)]


# ---------------------------------------------------------------- 기준 판 (fem_rt 격자 + CudaTetqK, float64)
def frame_base(i, x, v, F):
    st = dict()
    if a.mode == "lag":
        G = RTC.GridK(x, a.node_h)
        mn, vn = node_mv(G, x, v)
        U0 = h * vn + (h * h) * torch.tensor(gv, device=dev)
        CF = RTC.CudaTetqK(G, x, v, F, m, vol, cfg, h, U0, eps_sp=a.eps_sp, eps_st=a.eps_st, stick_xn=True, floor=FLOOR)
        fixd, vbc = fixd_of(G.npos, G.rows, G.lam, i * h)
        ni, nn = CF.ni, CF.nn
        xs = x[ni].contiguous()
        Fn = F.reshape(Np, 9)[ni].contiguous()
        vnear = torch.zeros(max(nn, 1), 3, device=dev)
        nslot = torch.full((Np,), -1, dtype=torch.int32, device=dev)
        nslot[ni] = torch.arange(nn, device=dev, dtype=torch.int32)
        Uc = torch.zeros(G.M, 3, device=dev)
        Vn = vn.clone()
        if a.diag and i == 0:
            kd = torch.zeros(G.M, device=dev, dtype=torch.float64).index_add_(
                0, G.tets.reshape(-1), (CF.VT8[:, None] * (lam + 2 * mu) * (CF.gl8.view(-1, 4, 3) ** 2).sum(-1)).reshape(-1))
            om = (kd / mn.double().clamp_min(1e-30)).sqrt()
            dtl = 2.0 / om.clamp_min(1e-30)
            ok = kd > 0
            qs = torch.quantile(dtl[ok].float().cpu(), torch.tensor([0.0, 1e-4, 1e-3, 0.01, 0.1, 0.5]))
            mfull = float(cfg["density"]) * float(torch.linalg.det(SX.lattice(a.node_h, SX.HZ_CUBE * a.node_h, dtype=torch.float64).A).abs())
            print(f"[진단] 노드 {G.M} (힘 받는 {int(ok.sum())}), 국소 안정 걸음 2/ω 분위 (0, 1e-4, 1e-3, 1e-2, 0.1, 0.5): "
                  + ", ".join(f"{float(q):.2e}" for q in qs) + f" s -> 필요 N (최소) {h / float(qs[0]):.0f}, (1e-3 분위) {h / float(qs[2]):.0f}; "
                  f"노드 질량 / 꽉 찬 노드 질량 분위 " + ", ".join(f"{float(q):.2e}" for q in torch.quantile((mn[ok] / mfull).float().cpu(), torch.tensor([0.0, 1e-3, 0.01, 0.1, 0.5]))), flush=True)
            lo_ = dtl < hs
            print(f"[진단] h_s {hs:.2e} 보다 국소 안정 걸음이 짧은 노드 {int((lo_ & ok).sum())}", flush=True)
        for j in range(a.N):
            f = torch.zeros(G.M, 3, device=dev)
            EX.launch("ex_tet_force_dd", G.T, (EX.P(None), np.int32(G.T), EX.P(G.tets), EX.P(Uc), EX.P(CF.gl8), EX.P(CF.FT8), EX.P(CF.VT8),
                                               np.float64(mu), np.float64(lam), EX.P(f)))
            dU = torch.empty(G.M, 3, device=dev)
            EX.launch("ex_node_step", G.M, (EX.P(None), np.int32(G.M), EX.P(mn), EX.P(f), EX.P(Vn), EX.P(dU), np.float32(gv[0]), np.float32(gv[1]),
                                            np.float32(gv[2]), np.float32(hs), np.float32(DAMP), EX.P(fixd), EX.P(vbc)))
            gn = torch.zeros(G.M, 3, device=dev)
            if nn:
                EX.launch("ex_near_force", nn, (EX.P(None), np.int32(nn), EX.P(ni), EX.P(xs), EX.P(Fn), EX.P(dU), EX.P(G.rows), EX.P(CF.lamb),
                                                EX.P(CF.gl4), EX.P(CF.tinv), EX.P(m), EX.P(vol), np.float64(hs), *near_args(nn),
                                                np.float64(mu), np.float64(lam), EX.P(gn)))
            EX.launch("ex_node_corr", G.M, (EX.P(None), np.int32(G.M), EX.P(mn), EX.P(gn), EX.P(Vn), EX.P(dU), EX.P(Uc), np.float32(hs), EX.P(fixd),
                                            np.int32(1 if nn else 0)))
            if a.diag and i == 0 and (j < 8 or j % 8 == 0):
                print(f"    [진단] 서브스텝 {j}: 최대 |v_n| {float(Vn.norm(dim=-1).max()):.3e}, 최대 |f/m| {float((f.norm(dim=-1) / mn.clamp_min(1e-30)).max()):.3e}, "
                      f"최대 |바닥 g/m| {float((gn.norm(dim=-1) / mn.clamp_min(1e-30)).max()):.3e}, 비유한 {int((~torch.isfinite(Vn)).sum())}", flush=True)
            if nn:
                EX.launch("ex_near_upd", nn, (EX.P(None), np.int32(nn), EX.P(ni), EX.P(xs), EX.P(Fn), EX.P(dU), EX.P(G.rows), EX.P(CF.lamb),
                                              EX.P(CF.gl4), EX.P(CF.tinv), *near_args(nn), np.float64(hs), EX.P(vnear)))
        x2 = torch.empty_like(x)
        v2 = torch.empty_like(x)
        F2 = torch.empty(Np, 9, device=dev)
        EX.launch("ex_g2p_lag", Np, (np.int32(Np), EX.P(x), EX.P(Uc), EX.P(Vn), EX.P(G.rows), EX.P(CF.lamb), EX.P(CF.gl4), EX.P(CF.tinv),
                                     EX.P(F.reshape(Np, 9).contiguous()), EX.P(nslot), EX.P(xs), EX.P(vnear), EX.P(Fn), EX.P(x2), EX.P(v2), EX.P(F2)), b=256)
        st.update(M=G.M, T=G.T, n_near=nn)
        return x2, v2, F2.view(Np, 3, 3), st
    # sub: 매 서브스텝 격자·P2G·G2P
    nns = []
    for j in range(a.N):
        G = RTC.GridK(x, a.node_h)
        mn, vn = node_mv(G, x, v)
        U0 = hs * vn + (hs * hs) * torch.tensor(gv, device=dev)
        CF = RTC.CudaTetqK(G, x, v, F, m, vol, cfg, hs, U0, eps_sp=a.eps_sp, eps_st=a.eps_st, stick_xn=True, floor=FLOOR)
        fixd, vbc = fixd_of(G.npos, G.rows, G.lam, i * h + j * hs)
        ni, nn = CF.ni, CF.nn
        nns.append(nn)
        f = torch.zeros(G.M, 3, device=dev)
        EX.launch("ex_tet_force_dd", G.T, (EX.P(None), np.int32(G.T), EX.P(G.tets), EX.P(None), EX.P(CF.gl8), EX.P(CF.FT8), EX.P(CF.VT8),
                                           np.float64(mu), np.float64(lam), EX.P(f)))
        dU = torch.empty(G.M, 3, device=dev)
        Vn = vn.clone()
        EX.launch("ex_node_step", G.M, (EX.P(None), np.int32(G.M), EX.P(mn), EX.P(f), EX.P(Vn), EX.P(dU), np.float32(gv[0]), np.float32(gv[1]),
                                        np.float32(gv[2]), np.float32(hs), np.float32(1.0), EX.P(fixd), EX.P(vbc)))
        gn = torch.zeros(G.M, 3, device=dev)
        if nn:
            xs = x[ni].contiguous()
            Fn = F.reshape(Np, 9)[ni].contiguous()
            EX.launch("ex_near_force", nn, (EX.P(None), np.int32(nn), EX.P(ni), EX.P(xs), EX.P(Fn), EX.P(dU), EX.P(G.rows), EX.P(CF.lamb),
                                            EX.P(CF.gl4), EX.P(CF.tinv), EX.P(m), EX.P(vol), np.float64(hs), *near_args(nn),
                                            np.float64(mu), np.float64(lam), EX.P(gn)))
            EX.launch("ex_node_corr", G.M, (EX.P(None), np.int32(G.M), EX.P(mn), EX.P(gn), EX.P(Vn), EX.P(dU), EX.P(None), np.float32(hs), EX.P(fixd),
                                            np.int32(1)))
        x2 = torch.empty_like(x)
        v2 = torch.empty_like(x)
        F2 = torch.empty(Np, 9, device=dev)
        EX.launch("ex_g2p_sub", Np, (np.int32(Np), EX.P(x), EX.P(dU), EX.P(G.rows), EX.P(CF.lamb), EX.P(CF.gl4), EX.P(CF.tinv),
                                     EX.P(F.reshape(Np, 9).contiguous()), np.int32(1 if FLOOR else 0), *PL, np.float64(a.eps_st), np.float64(a.eps_sp),
                                     np.float64(XN), np.float64(hs), np.float64(DAMP), EX.P(x2), EX.P(v2), EX.P(F2)), b=256)
        x, v, F = x2, v2, F2.view(Np, 3, 3)
    st.update(M=G.M, T=G.T, n_near=int(sum(nns) / len(nns)))
    return x, v, F, st


# ---------------------------------------------------------------- 최적화 판
class Opt:
    def __init__(self):
        self.lat = SX.lattice(float(a.node_h), SX.HZ_CUBE * float(a.node_h), device=dev, dtype=torch.float32)
        self.Ai = self.lat.Ai.float().contiguous()
        self.gpar = torch.zeros(12, device=dev)
        self.ipar = torch.zeros(4, dtype=torch.int32, device=dev)
        self.sz = torch.zeros(4, dtype=torch.int32, device=dev)
        self.Mcap = self.Tcap = self.Pcap = 0
        z = lambda *s, dt=torch.float32: torch.zeros(*s, device=dev, dtype=dt)
        self.rows = z(Np, 4, dt=torch.long)
        self.lamb = z(Np, 4)
        self.pinv = z(Np, dt=torch.long)
        self.nslot = z(Np, dt=torch.int32)
        self.nidx = z(Np, dt=torch.long)
        self.xs, self.Fn, self.vnear = z(Np, 3), z(Np, 9), z(Np, 3)
        self.X = [z(Np, 3), z(Np, 3)]
        self.V = [z(Np, 3), z(Np, 3)]
        self.FF = [z(Np, 9), z(Np, 9)]
        self.graph = None
        self.n_cap = 0
        self.fixd = None

    def _alloc(self, M, T, P):
        new = False
        if M > self.Mcap:
            self.Mcap = int(M * 1.3) + 64
            z = lambda *s: torch.zeros(*s, device=dev)
            self.mn, self.mv, self.vn, self.f, self.gn, self.dU, self.Uc = z(self.Mcap), z(self.Mcap, 3), z(self.Mcap, 3), z(self.Mcap, 3), \
                z(self.Mcap, 3), z(self.Mcap, 3), z(self.Mcap, 3)
            self.fx = torch.zeros(self.Mcap, dtype=torch.uint8, device=dev)
            self.vbc = z(self.Mcap, 3)
            new = True
        if T > self.Tcap:
            self.Tcap = int(T * 1.3) + 64
            self.VTs = torch.zeros(self.Tcap, device=dev)
            self.FTs = torch.zeros(self.Tcap, 9, device=dev)
            self.FT = torch.zeros(self.Tcap, 9, device=dev)
            self.gl = torch.zeros(self.Tcap, 12, device=dev)
            self.tets = torch.zeros(self.Tcap, 4, dtype=torch.long, device=dev)
            new = True
        if P > self.Pcap:
            self.Pcap = int(P * 1.3) + 64
            self.cm0 = torch.zeros(self.Pcap, dtype=torch.uint8, device=dev)
            self.cm1 = torch.zeros(self.Pcap, dtype=torch.uint8, device=dev)
            self.nmark = torch.zeros(self.Pcap, dtype=torch.uint8, device=dev)
            self.tmark = torch.zeros(self.Pcap * 9, dtype=torch.uint8, device=dev)
            self.ncm = torch.zeros(self.Pcap, dtype=torch.long, device=dev)
            self.tcm = torch.zeros(self.Pcap * 9, dtype=torch.long, device=dev)
            new = True
        if new:
            self.graph = None

    def grid(self, x, v, now):
        """셀 점유 격자 (월드 고정 Kuhn 격자, GridK 와 같은 격자점·같은 사면체 꼭짓점 순서). 동기화 2 번 (범위, 개수)."""
        lat = self.lat
        lo_ = x.min(0).values
        hi_ = x.max(0).values
        vmax = v.norm(dim=-1).max()
        ext = (hi_ - lo_).max().clamp_min(1e-12)
        dil = a.dil
        pad = 0.05 * ext
        if a.mode == "sub":
            pad = pad + vmax * h * 1.5 + 2.0 * a.node_h
        b0, b1 = lo_ - pad, hi_ + pad
        cor = torch.stack([torch.stack([b0[0] if (mm & 1) else b1[0], b0[1] if (mm & 2) else b1[1], b0[2] if (mm & 4) else b1[2]]) for mm in range(8)], 0)
        y = cor @ lat.Ai.T
        imin = torch.floor(y.min(0).values).long() - 1
        imax = torch.ceil(y.max(0).values).long() + 1
        ii = torch.cat([imin, imax, (vmax * h / lmin).ceil().long().view(1)]).tolist()                       # 동기화 1
        nnl = [max(ii[3 + k] - ii[k] + 1, 2) for k in range(3)]
        if a.mode == "sub" and dil < 0:
            dil = int(ii[6]) + 2                                                       # 프레임 최대 이동 (셀) + 2
        if a.mode == "lag":
            dil = 0
        n0, n1, n2 = nnl
        Pn = n0 * n1 * n2
        lo = (lat.A @ imin.to(lat.A.dtype))
        self.gpar[:3] = lo
        self.gpar[3:] = self.Ai.reshape(-1)
        self.ipar.copy_(torch.tensor([n0, n1, n2, 0], dtype=torch.int32), non_blocking=True)
        self.dims = (n0, n1, n2)
        self._alloc(0, 0, Pn)
        cm0, cm1 = self.cm0[:Pn], self.cm1[:Pn]
        cm0.zero_()
        EX.launch("ex_mark_cells", Np, (np.int32(Np), EX.P(x), EX.P(self.gpar), EX.P(self.ipar), EX.P(cm0)), b=256)
        if dil > 0:
            EX.launch("ex_dilate", Pn, (EX.P(self.ipar), np.int32(dil), EX.P(cm0), EX.P(cm1)), b=256)
            cm = cm1
        else:
            cm = cm0
        nm, tm = self.nmark[:Pn], self.tmark[:Pn * 9]
        nm.zero_()
        tm.zero_()
        EX.launch("ex_mark_nt", Pn, (EX.P(self.ipar), EX.P(cm), EX.P(nm), EX.P(tm)), b=256)
        ncs = torch.cumsum(nm, 0, dtype=torch.int64)
        tcs = torch.cumsum(tm, 0, dtype=torch.int64)
        MT = torch.stack([ncs[-1], tcs[-1]]).tolist()                                 # 동기화 2
        M, T = int(MT[0]), int(MT[1])
        self._alloc(M, T, Pn)
        nm, tm = self.nmark[:Pn], self.tmark[:Pn * 9]
        self.ncm[:Pn] = torch.where(nm.bool(), ncs - 1, torch.full_like(ncs, -1))
        self.tcm[:Pn * 9] = tcs - 1
        EX.launch("ex_build_tets", Pn * 9, (EX.P(self.ipar), EX.P(tm), EX.P(self.tcm), EX.P(self.ncm), EX.P(self.tets)), b=256)
        ar = torch.arange(Pn, device=dev)
        uniq = torch.empty(M + 1, dtype=torch.long, device=dev).scatter_(0, torch.where(nm.bool(), ncs - 1, torch.full_like(ncs, M)), ar)[:M]
        k2 = uniq % n2
        k1 = (uniq // n2) % n1
        k0 = uniq // (n1 * n2)
        npos = (torch.stack([k0, k1, k2], -1).to(lat.A.dtype) @ lat.A.T + lo).float().contiguous()
        self.sz.copy_(torch.tensor([M, T, 0, 0], dtype=torch.int32), non_blocking=True)
        EX.launch("ex_tet_geo32", T, (EX.P(self.sz), EX.P(self.tets), EX.P(npos), EX.P(self.gl)))
        self.M, self.T = M, T
        self.fixd = None
        if BC_SEL is not None:
            # cuboid 만 (입자 구속 없음 -- 위에서 막았다): train_deform.cfg_bc_nodes 의 cuboid 판정 그대로 (노드 위치)
            fx, vb = RT.cfg_bc_nodes(cfg, [], npos, torch.zeros(1, 4, dtype=torch.long, device=dev), torch.zeros(1, 4, device=dev), m[:1], now, h)
            if fx is not None:
                self.fx[:M] = fx.to(torch.uint8)
                self.vbc[:M] = vb.float()
                self.fixd = True
        return M, T

    # ---- 커널 묶음 (그래프에 들어가는 부분은 크기를 self.sz 에서 읽는다) ----
    def _p2g(self, x, v, F, H):
        Mc, Tc = self.Mcap, self.Tcap
        self.mn.zero_()
        self.mv.zero_()
        self.VTs.zero_()
        self.FTs.zero_()
        self.sz[2:3].zero_()
        EX.launch("ex_locate", Np, (np.int32(Np), EX.P(x), EX.P(self.gpar), EX.P(self.ipar), EX.P(self.ncm), EX.P(self.tcm), EX.P(self.tmark),
                                    EX.P(self.rows), EX.P(self.lamb), EX.P(self.pinv), EX.P(self.sz)), b=256)
        EX.launch("ex_p2g_mv", Np, (np.int32(Np), EX.P(self.rows), EX.P(self.lamb), EX.P(m), EX.P(v), EX.P(self.mn), EX.P(self.mv)), b=256)
        EX.launch("ex_vfin", Mc, (EX.P(self.sz), np.int32(0), EX.P(self.mn), EX.P(self.mv), EX.P(self.vn)), b=256)
        EX.launch("ex_split", Np, (np.int32(Np), EX.P(x), EX.P(self.rows), EX.P(self.lamb), EX.P(self.pinv), EX.P(self.vn), np.float32(H),
                                   np.float32(gv[0]), np.float32(gv[1]), np.float32(gv[2]), np.int32(1 if FLOOR else 0), *PLf,
                                   np.float32(a.eps_sp), np.float32(a.eps_st), EX.P(vol), EX.P(F), EX.P(self.VTs), EX.P(self.FTs),
                                   EX.P(self.nslot), EX.P(self.nidx), EX.P(self.sz)), b=256)
        EX.launch("ex_ffin", Tc, (EX.P(self.sz), np.int32(0), EX.P(self.VTs), EX.P(self.FTs), EX.P(self.FT)), b=256)

    def _force_step(self, Ucum):
        Mc, Tc = self.Mcap, self.Tcap
        self.f.zero_()
        EX.launch("ex_tet_force_ff", Tc, (EX.P(self.sz), np.int32(0), EX.P(self.tets), EX.P(Ucum), EX.P(self.gl), EX.P(self.FT), EX.P(self.VTs),
                                          np.float32(mu), np.float32(lam), EX.P(self.f)))
        EX.launch("ex_node_step", Mc, (EX.P(self.sz), np.int32(0), EX.P(self.mn), EX.P(self.f), EX.P(self.vn), EX.P(self.dU), np.float32(gv[0]),
                                       np.float32(gv[1]), np.float32(gv[2]), np.float32(hs),
                                       np.float32(DAMP if a.mode == "lag" else 1.0), EX.P(self.fx if self.fixd else None), EX.P(self.vbc)), b=256)

    def _near(self, Ucum, upd):
        Mc = self.Mcap
        if FLOOR:
            self.gn.zero_()
            EX.launch("ex_near_force", Np, (EX.P(self.sz), np.int32(0), EX.P(self.nidx), EX.P(self.xs), EX.P(self.Fn), EX.P(self.dU), EX.P(self.rows),
                                            EX.P(self.lamb), EX.P(self.gl), EX.P(self.pinv), EX.P(m), EX.P(vol), np.float64(hs), *near_args(0),
                                            np.float64(mu), np.float64(lam), EX.P(self.gn)), b=64)
        EX.launch("ex_node_corr", Mc, (EX.P(self.sz), np.int32(0), EX.P(self.mn), EX.P(self.gn), EX.P(self.vn), EX.P(self.dU), EX.P(Ucum),
                                       np.float32(hs), EX.P(self.fx if self.fixd else None), np.int32(1 if FLOOR else 0)), b=256)
        if FLOOR and upd:
            EX.launch("ex_near_upd", Np, (EX.P(self.sz), np.int32(0), EX.P(self.nidx), EX.P(self.xs), EX.P(self.Fn), EX.P(self.dU), EX.P(self.rows),
                                          EX.P(self.lamb), EX.P(self.gl), EX.P(self.pinv), *near_args(0), np.float64(hs), EX.P(self.vnear)), b=64)

    def _substeps(self):
        if a.mode == "lag":
            for j in range(a.N):
                self._force_step(self.Uc)
                self._near(self.Uc, True)
        else:
            for j in range(a.N):
                xa, va, Fa = self.X[j % 2], self.V[j % 2], self.FF[j % 2]
                xb, vb, Fb = self.X[(j + 1) % 2], self.V[(j + 1) % 2], self.FF[(j + 1) % 2]
                self._p2g(xa, va, Fa, hs)
                self._force_step(None)
                if FLOOR:
                    EX.launch("ex_gather", Np, (EX.P(self.sz), np.int32(0), EX.P(self.nidx), EX.P(xa), EX.P(Fa), EX.P(self.xs), EX.P(self.Fn)), b=256)
                self._near(None, False)
                EX.launch("ex_g2p_sub", Np, (np.int32(Np), EX.P(xa), EX.P(self.dU), EX.P(self.rows), EX.P(self.lamb), EX.P(self.gl), EX.P(self.pinv),
                                             EX.P(Fa), np.int32(1 if FLOOR else 0), *PL, np.float64(a.eps_st), np.float64(a.eps_sp), np.float64(XN),
                                             np.float64(hs), np.float64(DAMP), EX.P(xb), EX.P(vb), EX.P(Fb)), b=256)

    def run_substeps(self):
        if a.no_graph:
            self._substeps()
            return
        if self.graph is None:
            self._substeps()                                                       # 데우기 (컴파일), 결과는 버린다 -> 아래에서 상태를 되돌린다
            raise _Recapture()
        self.graph.replay()

    def capture(self):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        g = torch.cuda.CUDAGraph()
        with torch.cuda.stream(s):
            g.capture_begin()
            self._substeps()
            g.capture_end()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = g
        self.n_cap += 1

    def frame(self, i, x, v, F):
        M, T = self.grid(x, v, i * h)
        Fv = F.reshape(Np, 9)
        if a.mode == "lag":
            self._p2g(x, v, Fv, h)
            self.Uc.zero_()
            self.vnear.zero_()
            if FLOOR:
                EX.launch("ex_gather", Np, (EX.P(self.sz), np.int32(0), EX.P(self.nidx), EX.P(x), EX.P(Fv), EX.P(self.xs), EX.P(self.Fn)), b=256)
            snap = (self.vn.clone(), self.xs.clone(), self.Fn.clone()) if (self.graph is None and not a.no_graph) else None
            try:
                self.run_substeps()
            except _Recapture:
                self.vn.copy_(snap[0])
                self.xs.copy_(snap[1])
                self.Fn.copy_(snap[2])
                self.Uc.zero_()
                self.vnear.zero_()
                self.capture()
                self.graph.replay()
            x2, v2, F2 = self.X[1], self.V[1], self.FF[1]
            EX.launch("ex_g2p_lag", Np, (np.int32(Np), EX.P(x), EX.P(self.Uc), EX.P(self.vn), EX.P(self.rows), EX.P(self.lamb), EX.P(self.gl),
                                         EX.P(self.pinv), EX.P(Fv), EX.P(self.nslot if FLOOR else None), EX.P(self.xs), EX.P(self.vnear), EX.P(self.Fn),
                                         EX.P(x2), EX.P(v2), EX.P(F2)), b=256)
            return x2, v2, F2.view(Np, 3, 3)
        self.X[0].copy_(x)
        self.V[0].copy_(v)
        self.FF[0].copy_(Fv)
        try:
            self.run_substeps()
        except _Recapture:
            self.X[0].copy_(x)
            self.V[0].copy_(v)
            self.FF[0].copy_(Fv)
            self.capture()
            self.graph.replay()
        k = a.N % 2
        return self.X[k], self.V[k], self.FF[k].view(Np, 3, 3)


class _Recapture(Exception):
    pass


OPT = Opt() if a.opt else None
if OPT is not None:
    RTC.RAWPTR[0] = True
PRED = torch.empty(L, Np, 3, device=dev) if (not a.no_dump or a.dump_pred) else None
VD = torch.empty(L, Np, 3, device=dev, dtype=torch.float16) if not a.no_dump else None
FD = torch.empty(L, Np, 3, 3, device=dev, dtype=(torch.float32 if a.dump_f32 else torch.float16)) if not a.no_dump else None
errs, recs = [], []
flog = open(os.path.join(a.out, "fem.jsonl"), "w")
div_at = None
for i in range(L):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if OPT is None:
        x2, v2, F2, st = frame_base(i, x, v, F)
    else:
        ncap0 = OPT.n_cap
        x2, v2, F2 = OPT.frame(i, x, v, F)
        st = dict(M=OPT.M, T=OPT.T, recap=OPT.n_cap - ncap0)
    torch.cuda.synchronize()
    t_frame = time.perf_counter() - t0
    # ---- 측정 밖 ----
    if OPT is not None:
        szh = OPT.sz.tolist()
        st.update(n_near=int(szh[2]), overflow=int(szh[3]))
        if szh[3]:
            print(f"  [경고] 프레임 {i + 1}: 팽창 영역 밖 입자 {szh[3]} (격자 넘침)", flush=True)
            OPT.sz[3:4].zero_()
    e = float((x2 - GT[i]).norm(dim=-1).mean()) / EXT
    spread = float((x2.max(0).values - x2.min(0).values).norm()) / EXT
    fin = bool(torch.isfinite(x2).all()) and math.isfinite(e)
    smin = nbelow = None
    if FLOOR:
        s = ((x2 - torch.tensor(pt_, device=dev)) * torch.tensor(nr_, device=dev)).sum(-1)
        smin, nbelow = float(s.min()), int((s < 0).sum())
    errs.append(e if fin else float("nan"))
    rec = dict(t=i, sec=t_frame, err=e, spread=spread, s_min=smin, n_below=nbelow, **st)
    flog.write(json.dumps(rec, ensure_ascii=False) + "\n")
    flog.flush()
    recs.append(rec)
    print(f"  프레임 {i + 1}/{L} {1e3 * t_frame:.2f} ms 오차 {100 * e:.3f}% (누적 평균 {100 * np.nanmean(errs):.3f}%) 퍼짐 {spread:.3f} "
          f"노드 {st.get('M')} 사면체 {st.get('T')} 바닥 후보 {st.get('n_near')}" + (f" 최저 입자 {smin:+.5f} 아래 {nbelow}" if FLOOR else ""), flush=True)
    if PRED is not None:
        PRED[i] = x2
    if VD is not None:
        VD[i] = v2.half()
        FD[i] = F2.to(FD.dtype)
    x, v, F = x2.clone(), v2.clone(), F2.clone()
    if not fin or spread > a.div:
        div_at = i + 1
        print(f"[발산] 프레임 {div_at}: {'값 비유한' if not fin else '퍼짐 %.2f > %.1f' % (spread, a.div)} -- 멈춤", flush=True)
        break

flog.close()
W = a.warm
sec = [r["sec"] for r in recs[W:]]
med = lambda xs: sorted(xs)[len(xs) // 2] if xs else float("nan")
done = len(recs)
summ = dict(out=a.out, args=vars(a), frames=L, frames_done=done, diverged=div_at, err_mean=(sum(errs) / len(errs) if div_at is None else None),
            err_mean_done=float(np.nanmean(errs)), ms_mean=1e3 * sum(sec) / max(len(sec), 1), ms_med=1e3 * med(sec), fps=len(sec) / max(sum(sec), 1e-9),
            N=a.N, hs=hs, dt_cfl=dt_cfl, N_cfl=h / dt_cfl, cwave=cwave, lmin=lmin, mode=a.mode, opt=a.opt,
            s_min=(min(r["s_min"] for r in recs) if FLOOR else None), n_below_max=(max(r["n_below"] for r in recs) if FLOOR else None),
            spread_last=recs[-1]["spread"], n_near_med=med([r.get("n_near") or 0 for r in recs]),
            overflow_frames=sum(1 for r in recs if r.get("overflow")), gpu=torch.cuda.get_device_name(0), tag=a.tag, damp_sub=DAMP)
json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), ensure_ascii=False, indent=1)
print(f"[롤아웃] {a.tag} 서브스텝 {a.N} {a.mode}{' 최적화' if a.opt else ''}: {done}/{L} 프레임{' (발산 %d)' % div_at if div_at else ''}, 평균 {100 * summ['err_mean_done']:.3f}%  "
      f"프레임당 {summ['ms_mean']:.2f} ms (중앙 {summ['ms_med']:.2f}, {summ['fps']:.2f} fps, 앞 {W} 프레임 제외) 퍼짐 {summ['spread_last']:.3f} "
      f"최저 입자 {summ['s_min']} 아래 최대 {summ['n_below_max']}", flush=True)
if PRED is not None and div_at is None:
    if a.dump_pred:
        torch.save({"pred": PRED.cpu(), "EXT": EXT}, os.path.join(a.out, "pred.pt"))
    if not a.no_dump:
        torch.save({"pred": PRED.cpu(), "gt": GT.cpu(), "x0": d["x"][0].float(), "EXT": EXT, "t0": 0, "tag": a.tag,
                    "F": FD.float().cpu(), "v": VD.float().cpu(), "ctrl_pos": d.get("ctrl_pos"), "ctrl_R": d.get("ctrl_R"), "ctrl_id": d.get("ctrl_id"),
                    "nodes": None, "fscal": None, "gt_fscal": None}, os.path.join(a.out, "roll_fem.pt"))
        print(f"[롤아웃 덤프] {os.path.join(a.out, 'roll_fem.pt')}", flush=True)
