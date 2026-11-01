"""절 AK.22: 2 차 B-스플라인 요소 판 CUDA (lib/anchorflow/fem_bsel_cuda.py) -- exe/fem_bsel_sim.py (torch 검증판) 와 같은 식.
프레임: (torch) 격자 좌표·칸 번호·입자 칸 정렬, 노드 질량·운동량·국소 CFL 질량 (P2G), 바닥 후보 -> 서브스텝 N 번 CUDA 그래프 (힘·접촉·노드 걸음) -> G2P 커널.
출력 <out>/fem.jsonl·summary.json (오차·프레임 시간 ms -- 단독 측정은 fem_ab_run 처럼 GPU0 + gpu0.lock 으로 부를 때만 의미)."""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import numpy as np                                                                    # noqa: E402
import torch                                                                         # noqa: E402
from anchorflow import deform_model as DM                                           # noqa: E402
from anchorflow import fem_bsel_cuda as BK                                          # noqa: E402
from anchorflow import fem_rt as RT                                                 # noqa: E402
from anchorflow import fem_rt_cuda as RTC                                           # noqa: E402
from anchorflow import phys_resid as PR                                             # noqa: E402
from anchorflow import simplex as SX                                                # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--frames", type=int, default=80)
ap.add_argument("--node_h", type=float, default=0.028284)
ap.add_argument("--N", type=int, required=True)
ap.add_argument("--floor", choices=["vel", "none"], default="vel")
ap.add_argument("--theta", type=float, default=0.5)
ap.add_argument("--npd", type=float, default=0.005)
ap.add_argument("--cfg_bc", action="store_true")
ap.add_argument("--region_density", action="store_true")
ap.add_argument("--pg_vdamp", action="store_true")
ap.add_argument("--no_graph", action="store_true")
ap.add_argument("--dump", action="store_true")
ap.add_argument("--tag", default="")
ap.add_argument("--warm", type=int, default=2)
ap.add_argument("--vfl_out", action="store_true", help="절 AK.31: 바닥 고착을 접선·안쪽 법선에만")
ap.add_argument("--gq", action="store_true", help="절 AK.33 (정의 변경): 탄성 적분점을 입자 대신 칸 (B-스플라인 칸 [base+½, base+3/2]³) 고정 2x2x2 가우스 점으로 -- "
                "점 q = 입자가 든 칸마다 8 점 (같은 가중 = 칸 입자 부피 합 / 8), F_q = 그 영역 입자 F 의 회전·신장 분리 평균 (절 AI A1: R̄ = polar(Σ V R), S̄ = exp(Σ V log S / Σ V)), "
                "서브스텝 X_q = A(x_q) F_q (B-스플라인 사상 기울기), G2P 입자마다 F_p ← A(x_p) F_p 그대로")
a = ap.parse_args()
dev = "cuda"
RTC.RAWPTR[0] = True
os.makedirs(a.out, exist_ok=True)
d = torch.load(a.data, map_location="cpu", weights_only=False)
cfg = d["cfg"]
h = float(cfg["frame_dt"])
hs = h / a.N
DM.REGION_DENSITY[0] = bool(a.region_density)
m64 = DM.traj_mass(d, dev).double()
EXT = float(d["_ext"])
rho = (d["_rho"].to(dev).double()) if a.region_density else None
vol64 = (m64 / rho) if a.region_density else (m64 / float(cfg["density"]))
mu, lam = PR.lame(cfg["E"], cfg["nu"])
gv = [float(q) for q in cfg["g"]]
Np = int(d["x"].shape[1])
L = min(a.frames, int(d["x"].shape[0]) - 1)
x = d["x"][0].to(dev).float()
v = d["v"][0].to(dev).float()
F = DM.traj_F(d, h, dev)[0].to(dev).float().reshape(Np, 9)
VD_S = float(cfg.get("grid_v_damping_scale", 1.0))
DAMP = (VD_S ** (hs / float(cfg["substep_dt"]))) if (a.pg_vdamp and VD_S < 1.0) else 1.0
lat = SX.lattice(a.node_h, SX.HZ_CUBE * a.node_h, device=dev, dtype=torch.float64)
A64, Ai64 = lat.A, lat.Ai
Aif = Ai64.float().contiguous()
dxl = float(A64[:, 0].norm())
FLOOR = a.floor == "vel"
if FLOOR:
    pl = [b for b in cfg["boundary_conditions"] if b.get("type") == "surface_collider"][0]
    PT = [float(q) for q in pl["point"]]
    NR = np.array(pl["normal"], dtype=np.float64)
    NR = (NR / np.linalg.norm(NR)).tolist()
    PN = float(np.dot(PT, NR))
    DXG = float(cfg.get("grid_lim", 2.0)) / int(cfg.get("n_grid", 100))
    KEX = hs / float(cfg["substep_dt"])
else:
    PT, NR, PN, DXG, KEX = [0.0] * 3, [0.0, 0.0, 1.0], 0.0, 1.0, 1.0
BC_SEL = RT.cfg_bc_sel(cfg, x) if a.cfg_bc else None
cwave = math.sqrt((lam + 2 * mu) / float((rho.min() if rho is not None else cfg["density"])))
print(f"[절 AK.22 B-스플라인 요소 CUDA] 입자 {Np}, 프레임 {L}, N {a.N} (h_s {hs:.3e}, 전역 CFL 비 {hs / (dxl / cwave):.3f}), 격자 {dxl:.5f} m, θ {a.theta}, 바닥 {a.floor}", flush=True)
OFF = torch.stack(torch.meshgrid(torch.arange(3), torch.arange(3), torch.arange(3), indexing="ij"), -1).reshape(27, 3).to(dev)
f32 = lambda *s: torch.zeros(*s, device=dev)
S = dict(cid=torch.zeros(Np, dtype=torch.int32, device=dev), fx=f32(Np, 3), F=f32(Np, 9), V=f32(Np), r=f32(Np), x=f32(Np, 3), m=f32(Np),
         cand=torch.zeros(Np, dtype=torch.int32, device=dev), nc=torch.zeros(1, dtype=torch.int32, device=dev), nb=torch.zeros(3, dtype=torch.int32, device=dev),
         nbt=torch.zeros(1, dtype=torch.int32, device=dev), x2=f32(Np, 3), v2=f32(Np, 3), F2=f32(Np, 9))
NODE = dict(cap=0)
GQ = dict(cap=0)
GG = 0.5 / math.sqrt(3.0)                                                              # 칸 [½, 3/2] 의 2 점 가우스 = 1 ± ½/√3


def gq_alloc(q):
    if q <= GQ["cap"]:
        return False
    c = int(q * 1.3) + 1024
    GQ.update(cap=c, cid=torch.zeros(c, dtype=torch.int32, device=dev), fx=f32(c, 3), F=f32(c, 9), V=f32(c), r=torch.ones(c, device=dev))
    return True


def rs_mean(Fp, w, inv, nq):
    """절 AI A1 회전·신장 분리 평균 (float64): R̄ = polar(Σ w R_p), S̄ = exp(Σ w log S_p / Σ w); 뒤집힌 입자가 든 점은 유클리드 평균."""
    Fm = Fp.double().reshape(-1, 3, 3)
    U_, Sg, Vh = torch.linalg.svd(Fm)
    R = U_ @ Vh
    neg = torch.det(Fm) <= 0
    LS = Vh.transpose(1, 2) @ torch.diag_embed(torch.log(Sg.clamp_min(1e-12))) @ Vh
    W = w.double()[:, None, None]
    sR = torch.zeros(nq, 3, 3, device=dev, dtype=torch.float64).index_add_(0, inv, W * R)
    sL = torch.zeros(nq, 3, 3, device=dev, dtype=torch.float64).index_add_(0, inv, W * LS)
    sF = torch.zeros(nq, 3, 3, device=dev, dtype=torch.float64).index_add_(0, inv, W * Fm)
    sw = torch.zeros(nq, device=dev, dtype=torch.float64).index_add_(0, inv, w.double())[:, None, None]
    bad = torch.zeros(nq, device=dev, dtype=torch.bool).index_put_((inv[neg],), torch.ones(int(neg.sum()), device=dev, dtype=torch.bool))
    Ur, _, Vr = torch.linalg.svd(sR)
    Dg = torch.ones(nq, 3, device=dev, dtype=torch.float64); Dg[:, 2] = torch.sign(torch.det(Ur @ Vr))
    Rb = Ur @ torch.diag_embed(Dg) @ Vr
    ev, EV = torch.linalg.eigh(0.5 * (sL + sL.transpose(1, 2)) / sw.clamp_min(1e-300))
    Sb = EV @ torch.diag_embed(torch.exp(ev)) @ EV.transpose(1, 2)
    swc = sw.clamp_min(1e-300)
    Fq = torch.where(bad[:, None, None], sF / swc, Rb @ Sb)
    return Fq.reshape(nq, 9).float(), int(bad.sum())


def node_alloc(nbt):
    if nbt <= NODE["cap"]:
        return False
    c = int(nbt * 1.3) + 1024
    NODE.update(cap=c, mn=f32(c), mI=f32(c), f=f32(c, 3), vn=f32(c, 3), U=f32(c, 3), acc=f32(c), accN=f32(c), fx=torch.zeros(c, dtype=torch.uint8, device=dev), vbc=f32(c, 3))
    return True


GRAPH = [None]
STAT = {}
FIXD = [False]


def substeps():
    nbP = Np
    for _ in range(a.N):
        if a.gq:                                                                       # 절 AK.33: 칸 가우스 점 (용량 GQ cap, 남는 칸은 부피 0)
            BK.launch("bs_force", GQ["cap"], (np.int32(GQ["cap"]), BK.P(GQ["cid"]), BK.P(GQ["fx"]), BK.P(GQ["F"]), BK.P(GQ["V"]), BK.P(GQ["r"]), BK.P(Aif), BK.P(S["nb"]),
                                              BK.P(NODE["U"]), np.float32(mu), np.float32(lam), BK.P(NODE["f"])), b=128)
        else:
            BK.launch("bs_force", Np, (np.int32(Np), BK.P(S["cid"]), BK.P(S["fx"]), BK.P(S["F"]), BK.P(S["V"]), BK.P(S["r"]), BK.P(Aif), BK.P(S["nb"]),
                                       BK.P(NODE["U"]), np.float32(mu), np.float32(lam), BK.P(NODE["f"])), b=128)
        if FLOOR:
            BK.launch("bs_contact", nbP, (BK.P(S["nc"]), BK.P(S["cand"]), BK.P(S["cid"]), BK.P(S["fx"]), BK.P(S["x"]), BK.P(S["m"]), BK.P(S["nb"]), BK.P(NODE["U"]),
                                          *[np.float32(q) for q in PT], *[np.float32(q) for q in NR], np.float32(PN), np.float32(DXG), np.float32(a.npd),
                                          BK.P(NODE["acc"]), BK.P(NODE["accN"])), b=128)
        BK.launch("bs_node", NODE["cap"], (BK.P(S["nbt"]), BK.P(NODE["mn"]), BK.P(NODE["mI"]), BK.P(NODE["f"]), BK.P(NODE["vn"]), BK.P(NODE["U"]),
                                           *[np.float32(q) for q in gv], np.float32(hs), np.float32(DAMP), BK.P(NODE["fx"] if FIXD[0] else None), BK.P(NODE["vbc"]),
                                           BK.P(NODE["acc"]), BK.P(NODE["accN"]), np.int32(1 if FLOOR else 0), np.float32(KEX), *[np.float32(q) for q in NR],
                                           np.int32(1 if a.vfl_out else 0)), b=256)


def capture():
    BK.prepare()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        g.capture_begin()
        substeps()
        g.capture_end()
    torch.cuda.current_stream().wait_stream(s)
    GRAPH[0] = g


def frame(fi, x, v, F):
    y = x.double() @ Ai64.T
    base = torch.floor(y - 0.5).long()
    bmin = base.min(0).values
    bb = base - bmin
    nb = bb.max(0).values + 3
    nbl = nb.tolist()
    nbt = nbl[0] * nbl[1] * nbl[2]
    grew = node_alloc(nbt)
    cid = (bb[:, 0] * nbl[1] + bb[:, 1]) * nbl[2] + bb[:, 2]
    o = torch.argsort(cid)
    S["cid"].copy_(cid[o].int())
    S["fx"].copy_((y - base)[o].float())
    S["F"].copy_(F[o])
    S["x"].copy_(x[o])
    mo = m64[o]
    vo = vol64[o]
    S["m"].copy_(mo.float())
    S["V"].copy_(vo.float())
    S["nb"].copy_(nb.int())
    S["nbt"].fill_(nbt)
    # P2G (torch): 27 오프셋 노드 번호·기저
    fxo = (y - base)[o]
    w1 = torch.stack([0.5 * (1.5 - fxo) ** 2, 0.75 - (fxo - 1) ** 2, 0.5 * (fxo - 0.5) ** 2], 1)
    d1 = torch.stack([fxo - 1.5, -2 * (fxo - 1), fxo - 0.5], 1)
    wi, wj, wk = w1[:, OFF[:, 0], 0], w1[:, OFF[:, 1], 1], w1[:, OFF[:, 2], 2]
    W = wi * wj * wk
    GW = torch.stack([d1[:, OFF[:, 0], 0] * wj * wk, wi * d1[:, OFF[:, 1], 1] * wk, wi * wj * d1[:, OFF[:, 2], 2]], -1) @ Ai64
    bo = bb[o]
    nid = ((bo[:, None, 0] + OFF[None, :, 0]) * nbl[1] + bo[:, None, 1] + OFF[None, :, 1]) * nbl[2] + bo[:, None, 2] + OFF[None, :, 2]
    mn = torch.zeros(nbt, device=dev, dtype=torch.float64).index_add_(0, nid.reshape(-1), (W * mo[:, None]).reshape(-1))
    mv = torch.zeros(nbt, 3, device=dev, dtype=torch.float64).index_add_(0, nid.reshape(-1), (W[..., None] * (mo[:, None] * v[o].double())[:, None, :]).reshape(-1, 3))
    if a.gq:                                                                           # 절 AK.33: 입자 -> 칸 팔분 영역 가우스 점
        cso = cid[o].long()
        oi = ((fxo >= 1.0).long() * torch.tensor([4, 2, 1], device=dev)).sum(1)
        uc, cinv_ = torch.unique(cso, return_inverse=True)                            # 점유 칸 -> 8 점 모두 (가우스 규칙 그대로: 같은 가중 V_c / 8)
        nc_ = uc.numel()
        uq = (uc[:, None] * 8 + torch.arange(8, device=dev)[None]).reshape(-1)
        nq = uq.numel()
        grew = gq_alloc(nq) or grew
        qb = torch.stack([(uq % 8) // 4, (uq % 4) // 2, uq % 2], 1).double()
        qfx = 1.0 + GG * (2 * qb - 1)
        Vc = torch.zeros(nc_, device=dev, dtype=torch.float64).index_add_(0, cinv_, vo)
        qV = Vc.repeat_interleave(8) / 8.0
        qinv = cinv_ * 8 + oi                                                          # 팔분 영역 입자 F 의 회전·신장 분리 평균; 입자 없는 팔분 영역은 칸 전체 평균
        qF, nbad = rs_mean(F[o], vo, qinv, nq)
        cF, _ = rs_mean(F[o], vo, cinv_, nc_)
        has = torch.zeros(nq, device=dev, dtype=torch.bool).index_put_((qinv,), torch.ones_like(qinv, dtype=torch.bool))
        qF = torch.where(has[:, None], qF, cF.repeat_interleave(8, 0))
        cap = GQ["cap"]
        GQ["cid"][:nq].copy_((uq // 8).int()); GQ["cid"][nq:].fill_(int(uq[-1] // 8))
        GQ["fx"][:nq].copy_(qfx.float()); GQ["fx"][nq:].fill_(1.0)
        GQ["F"][:nq].copy_(qF); GQ["F"][nq:].zero_()
        GQ["V"][:nq].copy_(qV.float()); GQ["V"][nq:].zero_()
        STAT.update(nq=nq, nbad=nbad)
        qw = torch.stack([0.5 * (1.5 - qfx) ** 2, 0.75 - (qfx - 1) ** 2, 0.5 * (qfx - 0.5) ** 2], 1)
        qd = torch.stack([qfx - 1.5, -2 * (qfx - 1), qfx - 0.5], 1)
        qi, qj, qk = qw[:, OFF[:, 0], 0], qw[:, OFF[:, 1], 1], qw[:, OFF[:, 2], 2]
        QGW = torch.stack([qd[:, OFF[:, 0], 0] * qj * qk, qi * qd[:, OFF[:, 1], 1] * qk, qi * qj * qd[:, OFF[:, 2], 2]], -1) @ Ai64
        qc = (uq // 8)
        qbz = qc % nbl[2]; qby = (qc // nbl[2]) % nbl[1]; qbx = qc // (nbl[1] * nbl[2])
        qnid = ((qbx[:, None] + OFF[None, :, 0]) * nbl[1] + qby[:, None] + OFF[None, :, 1]) * nbl[2] + qbz[:, None] + OFF[None, :, 2]
        gq_ = QGW.norm(dim=-1)
        kn = torch.zeros(nbt, device=dev, dtype=torch.float64).index_add_(0, qnid.reshape(-1), (qV[:, None] * (lam + 2 * mu) * gq_ * gq_.sum(1, keepdim=True)).reshape(-1))
    else:
        gn = GW.norm(dim=-1)
        kn = torch.zeros(nbt, device=dev, dtype=torch.float64).index_add_(0, nid.reshape(-1), (vo[:, None] * (lam + 2 * mu) * gn * gn.sum(1, keepdim=True)).reshape(-1))
    NODE["mn"][:nbt].copy_(mn.float())
    NODE["mI"][:nbt].copy_(torch.maximum(mn, (hs / 2) ** 2 * kn).float().clamp_min(1e-30))
    NODE["vn"][:nbt].copy_(torch.where(mn[:, None] > 0, mv / mn.clamp_min(1e-30)[:, None], torch.zeros_like(mv)).float())
    NODE["U"].zero_()
    NODE["f"].zero_()
    NODE["acc"].zero_()
    NODE["accN"].zero_()
    # 전단 몫 가중 r = min(1, θ dx³ / 칸 입자 부피 합)
    cell = torch.floor(y[o]).long()
    cell = cell - cell.min(0).values
    cm = int(cell.max()) + 1
    _, cinv = torch.unique((cell[:, 0] * cm + cell[:, 1]) * cm + cell[:, 2], return_inverse=True)
    vc = torch.zeros(int(cinv.max()) + 1, device=dev, dtype=torch.float64).index_add_(0, cinv, vo)
    S["r"].copy_((torch.clamp(a.theta * dxl ** 3 / vc[cinv], max=1.0) if a.theta > 0 else torch.ones_like(vo)).float())
    if FLOOR:
        s0 = (x[o].double() - torch.tensor(PT, device=dev, dtype=torch.float64)) @ torch.tensor(NR, device=dev, dtype=torch.float64)
        cc = torch.nonzero(s0 < float(v.norm(dim=-1).max()) * h + 0.05).flatten()
        S["cand"][:cc.numel()].copy_(cc.int())
        S["nc"].fill_(cc.numel())
    if BC_SEL is not None:
        kq = torch.stack(torch.meshgrid(*[torch.arange(q, device=dev) for q in nbl], indexing="ij"), -1).reshape(-1, 3) + bmin
        npos = (kq.double() @ A64.T).float()
        fx_, vb_ = RT.cfg_bc_nodes(cfg, [], npos, torch.zeros(1, 4, dtype=torch.long, device=dev), torch.zeros(1, 4, device=dev), m64[:1].float(), fi * h, h)
        if fx_ is not None:
            NODE["fx"][:nbt].copy_(fx_.to(torch.uint8))
            NODE["vbc"][:nbt].copy_(vb_.float())
            FIXD[0] = True
    if a.no_graph:
        substeps()
    else:
        if GRAPH[0] is None or grew:
            capture()
        GRAPH[0].replay()
    BK.launch("bs_g2p", Np, (np.int32(Np), BK.P(S["cid"]), BK.P(S["fx"]), BK.P(Aif), BK.P(S["nb"]), BK.P(NODE["U"]), BK.P(NODE["vn"]),
                             BK.P(S["x"]), BK.P(S["F"]), BK.P(S["x2"]), BK.P(S["v2"]), BK.P(S["F2"])), b=128)
    x2 = torch.empty_like(x).index_copy_(0, o, S["x2"])
    v2 = torch.empty_like(v).index_copy_(0, o, S["v2"])
    F2 = torch.empty_like(F).index_copy_(0, o, S["F2"])
    return x2, v2, F2


flog = open(os.path.join(a.out, "fem.jsonl"), "w")
errs, secs = [], []
div_at = None
PRED = torch.empty(L, Np, 3) if a.dump else None
for fi in range(L):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    x2, v2, F2 = frame(fi, x, v, F)
    torch.cuda.synchronize()
    dt_ = time.perf_counter() - t0
    gt = d["x"][fi + 1].to(dev).float()
    e = float((x2 - gt).norm(dim=-1).mean()) / EXT
    fin = bool(torch.isfinite(x2).all()) and math.isfinite(e)
    errs.append(e if fin else float("nan"))
    secs.append(dt_)
    flog.write(json.dumps(dict(t=fi, err=e, sec=dt_)) + "\n")
    flog.flush()
    print(f"  프레임 {fi + 1}/{L} {1e3 * dt_:.2f} ms 오차 {100 * e:.3f}%" + (f" 가우스 점 {STAT.get('nq')} (입자 {Np}, 뒤집힌 입자 든 점 {STAT.get('nbad')})" if a.gq else ""), flush=True)
    if PRED is not None:
        PRED[fi] = x2.cpu()
    x, v, F = x2, v2, F2
    if not fin or e > 2.0:
        div_at = fi + 1
        print(f"[발산] 프레임 {div_at}", flush=True)
        break
sc = secs[a.warm:]
summ = dict(args=vars(a), frames=L, frames_done=len(errs), diverged=div_at, err_mean=(float(np.mean(errs)) if div_at is None else None),
            err_mean_done=float(np.nanmean(errs)), ms_mean=1e3 * float(np.mean(sc)) if sc else None, hs=hs)
json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1, ensure_ascii=False)
if PRED is not None and div_at is None:
    torch.save({"pred": PRED}, os.path.join(a.out, "pred.pt"))
print(f"[롤아웃] {a.tag} B-스플라인 요소 CUDA N {a.N}: {len(errs)}/{L} 프레임{' (발산 %d)' % div_at if div_at else ''}, 평균 {100 * summ['err_mean_done']:.3f}%, 프레임당 {summ['ms_mean']} ms", flush=True)
