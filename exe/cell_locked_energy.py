"""셀별 잠긴 탄성 에너지 -- 셀 하나에 아핀 변형 하나를 더 줘도 풀 수 없는 입자 F 퍼짐의 에너지.

셀 분할은 exe/cell_fspread.py 와 같다 (simplex.grid_for_nodes(h_fix=node_h, hz=1/√6, off=None) -> locate -> tet_id,
궤적마다 자기 현재 위치에 프레임마다, 덤프 노드 위치로 검산). 단 **점유 셀 전부**(입자 1 개 셀 포함)를 쓴다.

프레임마다 셀 c 마다
  E_c = Σ_{p∈c} V_p ψ(F_p)
  L_c = min_{J, det J>0} Σ_{p∈c} V_p ψ(J F_p)       (J: 3×3, 셀 하나에 아핀 하나)
ψ = phys_resid.psi_of 의 jelly(고정 공회전, FCR) 식 μ Σ(σ_i−1)² + λ/2 (Πσ_i − 1)², σ = F 의 특이값(부호 없음, 0.01 아래 자름),
E·ν = config 그대로. V_p = dx³ / (PG 격자 칸 입자 수) (0 프레임 배치, ficus_v0/energy/fen.py 와 같은 식).
E 는 psi_of(float32 F) 로 잰 값(기존 energy.json 과 같은 경로)과 float64 식 값 둘 다 남기고, L/E 는 float64 식끼리 나눈다.

최소화: 셀별 배치 사영 뉴턴. 입자 G = J F 의 ψ 헤시안을 해석 고유계(스케일 3×3 블록 + 쌍마다 비틀림·뒤집힘)로 만들고
음의 고유값을 0 으로 잘라 반정부호로, J 공간으로 옮겨 셀별 9×9 계를 푼다. 되돌이 선 탐색(Armijo 1e-4, det J>0 유지, 장벽 없음).
시작점 두 개: J = F̄_c⁻¹ (질량가중 평균, det F̄<0 이면 −F̄⁻¹ -- ψ 가 부호 없는 특이값이라 같은 값) 와 J = I, 작은 쪽.
입자 1 개 셀은 L=0. 수렴 기록: 셀별 반복 수, 최종 기울기 노름(척도 Σ V_p (2μ+λ)‖F_p‖ 로 나눈 값), 두 시작점의 L 차이.

주의: C¹ 롤아웃은 셀 안 매핑이 아핀이 아니라 3 차라서, "아핀 하나로 못 푸는 에너지" L 은 C¹ 에겐 실제 잠긴 양보다 과대 추정이다.

  python exe/cell_locked_energy.py --config CFG --vol_pt F_pg_v0.pt --node_h 0.056569 --out OUT \
     --traj gt:정답:#222222:h5:DIR --traj lin:선형:#d9480f:dump:roll_ov.pt --traj c1:C1:#1c7ed6:dump:roll_ov.pt \
     --energy_json ficus_c1/energy.json --fig fig.png --table table.md
"""
import argparse
import glob
import json
import math
import os
import re
import sys
import time

import h5py
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from anchorflow import simplex as SX  # noqa: E402
from anchorflow import deform_model as DM  # noqa: E402
from anchorflow import phys_resid  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--vol_pt", required=True, help="입자 부피용 F_pg_v0.pt (cfg·x[0]; fen.py 와 같은 식)")
ap.add_argument("--node_h", type=float, required=True)
ap.add_argument("--hz_ratio", type=float, default=SX.HZ_CUBE)
ap.add_argument("--n_nodes", type=float, default=32)
ap.add_argument("--traj", action="append", required=True, help="키:이름:색:h5|dump:경로 (첫 번째가 정답 h5)")
ap.add_argument("--out", required=True)
ap.add_argument("--energy_json", default="", help="검산용 기존 에너지 (ficus_c1/energy.json: 라벨별 E 1..125)")
ap.add_argument("--energy_keys", default="gt=정답 (PG),lin=선형 (기존 bary),c1=C1 N1")
ap.add_argument("--fig", default="")
ap.add_argument("--table", default="")
ap.add_argument("--max_frames", type=int, default=0)
ap.add_argument("--max_iter", type=int, default=100)
ap.add_argument("--gtol", type=float, default=1e-10, help="수렴: ‖g‖ / 척도 Σ V(2μ+λ)‖F‖")
ap.add_argument("--min_pts4", type=int, default=4)
ap.add_argument("--eps", type=float, default=1e-3, help="색: log10(밀도+eps)")
ap.add_argument("--q", type=float, nargs=2, default=[1.0, 99.0])
ap.add_argument("--hist_frames", type=int, nargs="+", default=[30, 60, 124])
ap.add_argument("--table_frames", type=int, nargs="+", default=[1, 10, 30, 60, 124])
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = "cuda"
DT = torch.float64
TR = []
for s in a.traj:
    k, lab, col, kind, p = s.split(":", 4)
    TR.append(dict(key=k, lab=lab, col=col, kind=kind, path=p))
cfg = json.load(open(a.config))
cfg.setdefault("n_grid", 50); cfg.setdefault("grid_lim", 2.0)
LOG = open(f"{a.out}/locked.log", "a")


def say(*m):
    s = " ".join(str(q) for q in m)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


assert phys_resid.mat_name(cfg) == "jelly", phys_resid.mat_name(cfg)
for r_ in cfg.get("additional_material_params", []):
    assert float(r_["E"]) == float(cfg["E"]) and float(r_["nu"]) == float(cfg["nu"]), ("영역 물성 E·ν 가 다르다", r_)
MU, LAM = phys_resid.lame(cfg["E"], cfg["nu"])
FDT = float(cfg.get("frame_dt", 1 / 60))
say(f"[설정] {time.strftime('%F %T')} node_h {a.node_h} hz {a.hz_ratio:.6f} off None, 점유 셀 전부; jelly FCR E {cfg['E']} ν {cfg['nu']} "
    f"-> μ {MU:.6g} λ {LAM:.6g}; 뉴턴 최대 {a.max_iter} 회, gtol {a.gtol}")


# ---------------- ψ, 기울기, 헤시안 (float64) ----------------
def psi_sv(S):
    """S: 특이값 [...,3] (부호 없음) -> ψ (자르지 않음)."""
    J = S.prod(-1)
    return MU * ((S - 1.0) ** 2).sum(-1) + 0.5 * LAM * (J - 1.0) ** 2


def psi_clamped(G):
    """psi_of 와 같은 식: σ 를 0.01 아래 자른다 (float64)."""
    S = torch.linalg.svdvals(G).clamp_min(0.01)
    return psi_sv(S)


def eval_G(G, need_h=False, project=True):
    """G [n,3,3] -> ψ [n], P=∂ψ/∂G [n,3,3], (모드 Q [n,9,3,3], 고유값 lam [n,9])."""
    U, S, Vh = torch.linalg.svd(G)
    J = S.prod(-1)
    s0, s1, s2 = S[:, 0], S[:, 1], S[:, 2]
    pr = torch.stack([s1 * s2, s0 * s2, s0 * s1], -1)          # Π_{j≠i} σ_j
    dpsi = 2 * MU * (S - 1.0) + LAM * (J - 1.0)[:, None] * pr
    psi = MU * ((S - 1.0) ** 2).sum(-1) + 0.5 * LAM * (J - 1.0) ** 2
    P = U @ torch.diag_embed(dpsi) @ Vh
    if not need_h:
        return psi, P, None, None
    n = G.shape[0]
    A = 2 * MU * torch.eye(3, dtype=DT, device=G.device).expand(n, 3, 3) + LAM * pr[:, :, None] * pr[:, None, :]
    Jm = LAM * (J - 1.0)
    for i, j, k in ((0, 1, 2), (0, 2, 1), (1, 2, 0)):
        A[:, i, j] = A[:, i, j] + Jm * S[:, k]
        A[:, j, i] = A[:, j, i] + Jm * S[:, k]
    ew, ev = torch.linalg.eigh(A)                                # 스케일 모드
    Q = torch.zeros(n, 9, 3, 3, dtype=DT, device=G.device)
    lam = torch.zeros(n, 9, dtype=DT, device=G.device)
    for m in range(3):
        Q[:, m] = U @ torch.diag_embed(ev[:, :, m]) @ Vh
        lam[:, m] = ew[:, m]
    r2 = 1.0 / math.sqrt(2.0)
    m = 3
    for i, j, k in ((0, 1, 2), (0, 2, 1), (1, 2, 0)):
        E1 = torch.zeros(3, 3, dtype=DT, device=G.device); E1[i, j] = 1.0
        Tw = (E1 - E1.T) * r2; Fl = (E1 + E1.T) * r2
        Q[:, m] = U @ Tw @ Vh
        lam[:, m] = 2 * MU * (1.0 - 2.0 / (S[:, i] + S[:, j]).clamp_min(1e-300)) + Jm * S[:, k]
        Q[:, m + 1] = U @ Fl @ Vh
        lam[:, m + 1] = 2 * MU - Jm * S[:, k]
        m += 2
    if project:
        lam = lam.clamp_min(0.0)
    return psi, P, Q, lam


class Cells:
    """한 프레임의 셀 묶음 (입자 2 개 이상 셀만 최적화)."""

    def __init__(self, c, F, w, C):
        self.c, self.F, self.w, self.C = c, F, w, C

    def f(self, Jc, sub=None):
        """셀 목적함수 [C]. sub: 셀 마스크 (그 셀 입자만 계산, 나머지 0)."""
        if sub is None:
            ps, _, _, _ = eval_G(Jc[self.c] @ self.F)
            return torch.zeros(self.C, dtype=DT, device=dev).index_add_(0, self.c, self.w * ps)
        pm = sub[self.c]
        cc = self.c[pm]
        ps, _, _, _ = eval_G(Jc[cc] @ self.F[pm])
        return torch.zeros(self.C, dtype=DT, device=dev).index_add_(0, cc, self.w[pm] * ps)

    def fgh(self, Jc, sub, project=True):
        pm = sub[self.c]
        cc, Fp, wp = self.c[pm], self.F[pm], self.w[pm]
        ps, P, Q, lam = eval_G(Jc[cc] @ Fp, need_h=True, project=project)
        f = torch.zeros(self.C, dtype=DT, device=dev).index_add_(0, cc, wp * ps)
        g = torch.zeros(self.C, 3, 3, dtype=DT, device=dev).index_add_(0, cc, wp[:, None, None] * (P @ Fp.transpose(1, 2)))
        QF = (Q @ Fp.transpose(1, 2)[:, None]).reshape(-1, 9, 9)     # 모드 m 의 J 공간 벡터
        Hp = torch.einsum("nm,nmi,nmj->nij", lam * wp[:, None], QF, QF)
        H = torch.zeros(self.C, 9, 9, dtype=DT, device=dev).index_add_(0, cc, Hp)
        return f, g.reshape(-1, 9), H


def newton(cells, J0, active):
    """active 셀만 J 를 고친다. 반환 (J, f, 반복 수, 최종 ‖g‖, 수렴 여부)."""
    C = cells.C
    J = J0.clone()
    it = torch.zeros(C, dtype=torch.long, device=dev)
    gn = torch.zeros(C, dtype=DT, device=dev)
    conv = ~active
    f = cells.f(J)
    for k in range(a.max_iter):
        sub = ~conv
        if not bool(sub.any()):
            break
        fk, g, H = cells.fgh(J, sub)
        gnk = g.norm(dim=-1)
        gn = torch.where(sub, gnk, gn)
        done = sub & (gnk <= a.gtol * SCALE)
        conv = conv | done
        sub = sub & ~done
        if not bool(sub.any()):
            break
        ids = sub.nonzero().squeeze(1)
        Hs, gs = H[ids], g[ids]
        tr = Hs.diagonal(dim1=1, dim2=2).sum(-1).clamp_min(1e-300) / 9.0
        I9 = torch.eye(9, dtype=DT, device=dev)
        d = torch.zeros_like(gs)
        solved = torch.zeros(ids.numel(), dtype=torch.bool, device=dev)
        for delta in (1e-12, 1e-9, 1e-6, 1e-3):
            Lc, info = torch.linalg.cholesky_ex(Hs + (delta * tr)[:, None, None] * I9)
            ok = (info == 0) & ~solved
            if bool(ok.any()):
                dd = torch.cholesky_solve(-gs[:, :, None], Lc).squeeze(-1)
                d = torch.where(ok[:, None], dd, d)
                solved = solved | ok
            if bool(solved.all()):
                break
        d = torch.where(solved[:, None], d, -gs / tr[:, None])          # 끝내 실패하면 기울기 방향
        FAILCH[0] += int((~solved).sum())
        slope = (gs * d).sum(-1)
        d = torch.where((slope < 0)[:, None], d, -gs / tr[:, None]); slope = (gs * d).sum(-1)
        # 선 탐색
        Jcur = J[ids]; fcur = fk[ids]
        alpha = torch.ones(ids.numel(), dtype=DT, device=dev)
        acc = torch.zeros(ids.numel(), dtype=torch.bool, device=dev)
        Jnew = Jcur.clone(); fnew = fcur.clone()
        for _ls in range(40):
            todo = ~acc
            if not bool(todo.any()):
                break
            Jt = Jcur + alpha[:, None, None] * d.reshape(-1, 3, 3)
            detok = torch.linalg.det(Jt) > 0
            Jfull = J.clone(); Jfull[ids] = Jt
            m_ = torch.zeros(C, dtype=torch.bool, device=dev); m_[ids[todo & detok]] = True
            ft = cells.f(Jfull, m_)[ids]
            good = todo & detok & (ft <= fcur + 1e-4 * alpha * slope)
            Jnew = torch.where(good[:, None, None], Jt, Jnew); fnew = torch.where(good, ft, fnew)
            acc = acc | good
            alpha = torch.where(acc, alpha, alpha * 0.5)
        # 선 탐색 실패 셀 = 더 못 줄인다 (반올림 바닥) -> 수렴으로 친다 (기울기 노름은 그대로 기록)
        J[ids] = Jnew; f[ids] = fnew
        it[ids] += 1
        stall = torch.zeros(C, dtype=torch.bool, device=dev); stall[ids[~acc]] = True
        STALL[0] += int(stall.sum())
        conv = conv | stall
    # 최종 기울기
    sub = active.clone()
    if bool(sub.any()):
        _f, g, _H = cells.fgh(J, sub)
        gn = torch.where(sub, g.norm(dim=-1), gn)
    gconv = ~active | (gn <= a.gtol * SCALE)
    return J, f, it, gn, gconv


STALL = [0]
FAILCH = [0]
SCALE = None


# ---------------- 자체 시험: 해석 기울기·헤시안 vs 유한차분 ----------------
def selftest():
    torch.manual_seed(0)
    n = 12
    F = torch.eye(3, dtype=DT, device=dev) + 0.3 * torch.randn(n, 3, 3, dtype=DT, device=dev)
    F[3] = torch.diag(torch.tensor([1.2, 0.9, -0.8], dtype=DT, device=dev))       # det<0 입자
    c = torch.tensor([0] * 5 + [1] * 7, device=dev)
    w = torch.rand(n, dtype=DT, device=dev) + 0.5
    cl = Cells(c, F, w, 2)
    J = torch.eye(3, dtype=DT, device=dev).expand(2, 3, 3) + 0.1 * torch.randn(2, 3, 3, dtype=DT, device=dev)
    allc = torch.ones(2, dtype=torch.bool, device=dev)
    f0, g0, H0 = cl.fgh(J, allc, project=False)
    h = 1e-6
    gfd = torch.zeros(2, 9, dtype=DT, device=dev); Hfd = torch.zeros(2, 9, 9, dtype=DT, device=dev)
    for q in range(9):
        E_ = torch.zeros(9, dtype=DT, device=dev); E_[q] = h; E_ = E_.reshape(3, 3)
        fp, gp, _ = cl.fgh(J + E_, allc, project=False); fm, gm, _ = cl.fgh(J - E_, allc, project=False)
        gfd[:, q] = (fp - fm) / (2 * h); Hfd[:, :, q] = (gp - gm) / (2 * h)
    eg = float((gfd - g0).norm() / g0.norm()); eh = float((Hfd - H0).norm() / H0.norm())
    say(f"[자체 시험] 기울기 상대오차 {eg:.2e}, 헤시안(사영 없음) 상대오차 {eh:.2e}")
    if eg > 1e-6 or eh > 1e-5:
        say("[중단] 해석 미분이 유한차분과 다르다"); sys.exit(5)


# ---------------- 데이터 ----------------
def load_traj(t):
    if t["kind"] == "h5":
        fs = sorted(glob.glob(os.path.join(t["path"], "**", "sim_*.h5"), recursive=True),
                    key=lambda p: int(re.findall(r"(\d+)", os.path.basename(p))[-1]))
        X, F = [], []
        for f in tqdm(fs, desc=f"{t['key']} h5 읽기"):
            with h5py.File(f, "r") as h:
                X.append(torch.from_numpy(np.array(h["x"]).T.copy()))
                F.append(torch.from_numpy(np.array(h["f_tensor"]).T.reshape(-1, 3, 3).copy()))
        if float(F[0].abs().max()) == 0.0:
            F[0] = torch.eye(3).expand_as(F[0]).clone()
            say(f"[{t['key']}] 0 프레임 F 가 0 (PG 미초기화) -> 항등")
        t["X"], t["F"], t["nodes"] = X, F, None
        say(f"[{t['key']}] h5 {len(fs)} 프레임, 입자 {X[0].shape[0]}")
    else:
        D = torch.load(t["path"], map_location="cpu", weights_only=False, mmap=True)
        assert D["F"].dtype == torch.float32 and int(D.get("t0", 0)) == 0
        L = D["F"].shape[0]
        I = torch.eye(3).expand(D["F"].shape[1], 3, 3)
        t["X"] = [D["x0"].float()] + [D["pred"][i] for i in range(L)]
        t["F"] = [I] + [D["F"][i] for i in range(L)]
        t["nodes"] = D.get("nodes")
        say(f"[{t['key']}] 덤프 F {tuple(D['F'].shape)} {D['F'].dtype}, nodes {'있음' if D.get('nodes') else '없음'}")
    return t


selftest()
for t in TR:
    load_traj(t)
T = min(len(t["X"]) for t in TR)
if a.max_frames:
    T = min(T, a.max_frames)
N = TR[0]["X"][0].shape[0]
say(f"[공통] 프레임 0..{T - 1}, 입자 {N}")
for t in TR[1:]:
    say(f"[검산] {t['key']} 0 프레임 위치 vs 정답 최대 차 {float((t['X'][0].float() - TR[0]['X'][0].float()).abs().max()):.3e}")
# 질량 (F̄ 시작점용, cell_fspread 와 같음)
DM.REGION_DENSITY[0] = True
MASS = DM.traj_mass({"cfg": cfg, "x": [TR[0]["X"][0].float()]}, dev=dev).double()
# 부피 (fen.py 와 같은 식: F_pg_v0.pt 의 cfg·x[0])
Dv = torch.load(a.vol_pt, map_location="cpu", weights_only=False, mmap=True)
cv = Dv["cfg"]; ngv = int(cv.get("n_grid", 100)); dxv = float(cv.get("grid_lim", 2.0)) / ngv
x0v = Dv["x"][0].float()
assert x0v.shape[0] == N
say(f"[부피] {a.vol_pt}: n_grid {ngv} (cfg 값 {cv.get('n_grid')}), grid_lim {cv.get('grid_lim', 2.0)}, x[0] vs 정답 0 프레임 최대 차 "
    f"{float((x0v - TR[0]['X'][0].float()).abs().max()):.3e}")
vi = (x0v / dxv).long().clamp(0, ngv - 1); fl = (vi[:, 0] * ngv + vi[:, 1]) * ngv + vi[:, 2]
cn = torch.zeros(ngv ** 3).index_add_(0, fl, torch.ones(N))
VOL = (dxv ** 3 / cn[fl]).double().to(dev)
del Dv
say(f"[부피] 합 {float(VOL.sum()):.6g}, 질량 합 {float(MASS.sum()):.6g}")
EJ = None
if a.energy_json and os.path.exists(a.energy_json):
    EJ0 = json.load(open(a.energy_json))
    EJ = {}
    for kv in a.energy_keys.split(","):
        k, lab = kv.split("=", 1)
        if lab in EJ0:
            EJ[k] = EJ0[lab]["E"]          # 프레임 1..
    say(f"[검산 기준] {a.energy_json}: {list(EJ)}")


def frame(x32, F32, nodes_t):
    global SCALE
    lo, lat, nn = SX.grid_for_nodes(x32, a.n_nodes, hz_ratio=a.hz_ratio, off=None, h_fix=a.node_h)
    idx, _lam, aux = SX.locate(x32, lo, lat, nn)
    chk = None
    if nodes_t is not None:
        _rows, uniq = SX.active_nodes(idx)
        npos = SX.node_pos(lo, lat, nn, uniq)
        nd = nodes_t[0].to(dev).float()
        if npos.shape != nd.shape:
            return None, f"노드 수 {tuple(npos.shape)} vs 덤프 {tuple(nd.shape)}"
        chk = float((npos - nd).abs().max())
        if chk > 1e-5:
            return None, f"노드 위치 최대 차 {chk:.3e}"
    tid = SX.tet_id(lo, lat, nn, aux)
    _u, c, cnt = torch.unique(tid, return_inverse=True, return_counts=True)
    C = int(cnt.numel())
    F = F32.to(dev).double()
    w = VOL
    # E: psi_of (float32, 기존 경로) 와 float64 식
    with torch.no_grad():
        p32 = torch.empty(N, device=dev)
        for s in range(0, N, 65536):
            p32[s:s + 65536] = phys_resid.psi_of(F32[s:s + 65536].to(dev).float(), cfg, FDT)[0]
    p64 = psi_clamped(F)
    Ec = torch.zeros(C, dtype=DT, device=dev).index_add_(0, c, w * p64)
    E32 = float((w * p32.double()).sum())
    cells = Cells(c, F, w, C)
    SCALE = torch.zeros(C, dtype=DT, device=dev).index_add_(0, c, w * (2 * MU + LAM) * F.flatten(1).norm(dim=-1))
    active = cnt >= 2
    M = torch.zeros(C, dtype=DT, device=dev).index_add_(0, c, MASS)
    Fb = torch.zeros(C, 3, 3, dtype=DT, device=dev).index_add_(0, c, MASS[:, None, None] * F) / M[:, None, None]
    dF = torch.linalg.det(Fb)
    I3 = torch.eye(3, dtype=DT, device=dev).expand(C, 3, 3)
    good = dF.abs() > 1e-12
    Ja = torch.where(good[:, None, None], torch.linalg.inv(torch.where(good[:, None, None], Fb, I3)), I3)
    Ja = torch.where((dF < 0)[:, None, None], -Ja, Ja)
    STALL[0] = 0; FAILCH[0] = 0
    Ja_, fa, ita, gna, cva = newton(cells, Ja, active)
    sa = STALL[0]; STALL[0] = 0
    Jb_, fb, itb, gnb, cvb = newton(cells, I3.clone(), active)
    sb = STALL[0]
    # 최종 L 은 psi_of 와 같게 σ 를 0.01 아래 자른 식으로 다시 잰다
    La = torch.zeros(C, dtype=DT, device=dev).index_add_(0, c, w * psi_clamped(Ja_[c] @ F))
    Lb = torch.zeros(C, dtype=DT, device=dev).index_add_(0, c, w * psi_clamped(Jb_[c] @ F))
    Lc = torch.minimum(La, Lb)
    Lc = torch.where(active, Lc, torch.zeros_like(Lc))
    Lc = torch.minimum(Lc, Ec)        # J=I 가 E 이므로 L<=E (반올림 차만 자름)
    useb = Lb < La
    it = torch.where(useb, itb, ita); gn = torch.where(useb, gnb, gna); cv = torch.where(useb, cvb, cva)
    Vc = torch.zeros(C, dtype=DT, device=dev).index_add_(0, c, w)
    dens = (Lc / Vc)[c]          # 입자별 = 그 셀의 잠긴 에너지 밀도
    act = active
    rel_ab = ((La - Lb).abs() / Ec.clamp_min(1e-300))[act]
    sig_low = int((torch.linalg.svdvals(torch.where(useb[:, None, None], Jb_, Ja_)[c] @ F) < 0.01).any(-1).sum())
    m4 = cnt >= a.min_pts4
    st = dict(C=C, C1=int((cnt == 1).sum()), C4=int(m4.sum()), E=float(Ec.sum()), E_psiof32=E32, L=float(Lc.sum()),
              E4=float(Ec[m4].sum()), L4=float(Lc[m4].sum()),
              it_max=int(it[act].max()) if bool(act.any()) else 0, it_mean=float(it[act].double().mean()) if bool(act.any()) else 0,
              gn_rel_max=float((gn / SCALE.clamp_min(1e-300))[act].max()) if bool(act.any()) else 0,
              gn_rel_p99=float(torch.quantile((gn / SCALE.clamp_min(1e-300))[act].float(), 0.99)) if bool(act.any()) else 0,
              nonconv=int((~cv[act]).sum()), stall=[sa, sb], start_b_wins=int((useb & act).sum()),
              start_diff_rel_med=float(rel_ab.median()) if rel_ab.numel() else 0.0,
              start_diff_rel_max=float(rel_ab.max()) if rel_ab.numel() else 0.0,
              start_diff_abs_sum=float((La - Lb).abs()[act].sum()),
              sig_below_001=sig_low, chol_fail=FAILCH[0], node_chk=chk)
    return (Ec.cpu().numpy(), Lc.cpu().numpy(), cnt.cpu().numpy(), dens.float().cpu().numpy(), st), None


RES = {}
for t in TR:
    k = t["key"]
    Es, Ls, Ns, Ds, ST = [], [], [], [], []
    for f in tqdm(range(T), desc=f"{k} 잠긴 에너지"):
        nod = t["nodes"][f] if (t["nodes"] is not None and f < len(t["nodes"])) else None
        r, err = frame(t["X"][f].to(dev).float(), t["F"][f], nod)
        if r is None:
            say(f"[중단] {k} 프레임 {f}: 격자 재구성이 덤프와 다르다 -- {err}"); sys.exit(4)
        Ec, Lc, cnt, dens, st = r
        Es.append(Ec); Ls.append(Lc); Ns.append(cnt); Ds.append(dens); ST.append(st)
        if f in (1, 2) or f % 20 == 0 or f == T - 1:
            say(f"[{k} 프레임 {f}] 셀 {st['C']} (1 입자 {st['C1']}, ≥4 {st['C4']}) E {st['E']:.4e} (psi_of f32 {st['E_psiof32']:.4e}) "
                f"L {st['L']:.4e} L/E {st['L'] / max(st['E'], 1e-300):.4f}; 반복 평균 {st['it_mean']:.1f} 최대 {st['it_max']}, "
                f"‖g‖/척도 최대 {st['gn_rel_max']:.2e} (99% {st['gn_rel_p99']:.2e}), 미수렴 {st['nonconv']}, 선탐색 정지 {st['stall']}, "
                f"I 시작이 이긴 셀 {st['start_b_wins']}, 두 시작 |ΔL|/E_c 중앙 {st['start_diff_rel_med']:.1e} 최대 {st['start_diff_rel_max']:.1e}, "
                f"σ<0.01 입자 {st['sig_below_001']}, 촐레스키 실패 {st['chol_fail']}")
    off = np.cumsum([0] + [len(v) for v in Es])
    np.savez_compressed(f"{a.out}/cells_{k}.npz", E=np.concatenate(Es), L=np.concatenate(Ls), n=np.concatenate(Ns), offsets=off)
    np.save(f"{a.out}/dens_{k}.npy", np.stack(Ds).astype(np.float32))
    RES[k] = dict(lab=t["lab"], col=t["col"], path=t["path"], st=ST)
    json.dump({kk: dict(lab=v["lab"], path=v["path"], st=v["st"]) for kk, v in RES.items()},
              open(f"{a.out}/summary.json", "w"), ensure_ascii=False, indent=0)
    # 검산
    if EJ is not None and k in EJ:
        ref = np.array(EJ[k][:T - 1]); e32 = np.array([s["E_psiof32"] for s in ST[1:1 + len(ref)]])
        e64 = np.array([s["E"] for s in ST[1:1 + len(ref)]])
        say(f"[검산 E] {k}: psi_of(f32)·V 합 vs energy.json 최대 상대차 {np.max(np.abs(e32 - ref) / np.abs(ref)):.2e}, "
            f"float64 식 vs energy.json {np.max(np.abs(e64 - ref) / np.abs(ref)):.2e} (프레임 1..{len(ref)})")
        if np.max(np.abs(e32 - ref) / np.abs(ref)) > 1e-4:
            say("[중단] E 합이 기존 energy.json 과 다르다"); sys.exit(6)
    del t["X"], t["F"]
say(f"[계산 끝] {time.strftime('%F %T')}")

# ---------------- 표 ----------------
if a.table:
    with open(a.table, "w") as fo:
        fo.write("# ficus_v0 셀별 잠긴 탄성 에너지 (E = Σ V ψ(F), L = Σ_c min_J Σ_{p∈c} V ψ(J F), 셀 = 우리 격자 Kuhn 사면체 node_h "
                 f"{a.node_h}, 점유 셀 전부)\n\n")
        fo.write("주의: C¹ 은 셀 안 매핑이 3 차라서 아핀 하나로 못 푸는 에너지 L 이 실제 잠긴 양보다 과대 추정이다.\n\n")
        fo.write("| 프레임 | 궤적 | E [J] | L [J] | L/E | 셀 수 (1 입자) | L/E (입자 ≥4 셀만) |\n|---|---|---|---|---|---|---|\n")
        for f in a.table_frames:
            if f >= T:
                continue
            for t in TR:
                s = RES[t["key"]]["st"][f]
                fo.write(f"| {f} | {t['lab']} | {s['E']:.4e} | {s['L']:.4e} | {s['L'] / max(s['E'], 1e-300):.4f} | {s['C']} ({s['C1']}) | "
                         f"{s['L4'] / max(s['E4'], 1e-300):.4f} |\n")
    say(f"[표] {a.table}")

# ---------------- 그림 ----------------
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
_FONTS = [os.path.expanduser("~/.fonts/NotoSansCJKkr-Regular.otf")] + [p for p in fm.findSystemFonts() if "NotoSansCJK" in os.path.basename(p)]
for _p in _FONTS:
    if os.path.exists(_p):
        fm.fontManager.addfont(_p); plt.rcParams["font.family"] = fm.FontProperties(fname=_p).get_name(); break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"
if a.fig:
    fig = plt.figure(figsize=(20, 11), dpi=110)
    gs = fig.add_gridspec(2, 6)
    ax1 = fig.add_subplot(gs[0, 0:3]); ax2 = fig.add_subplot(gs[0, 3:6])
    tt = np.arange(1, T)
    for t in TR:
        S_ = RES[t["key"]]["st"]
        E_ = np.array([S_[f]["E"] for f in tt]); L_ = np.array([S_[f]["L"] for f in tt])
        ax1.plot(tt, E_, color=t["col"], lw=1.8, label=f"{t['lab']} E")
        ax1.plot(tt, L_, color=t["col"], lw=1.6, ls="--", label=f"{t['lab']} L")
        ax2.plot(tt, L_ / E_, color=t["col"], lw=1.8, label=t["lab"])
    ax1.set_yscale("log"); ax1.set_xlabel("프레임"); ax1.set_ylabel("에너지 [J] (세로축 로그)")
    ax1.set_title("1. 탄성 에너지 E (실선) · 잠긴 에너지 L (점선)", fontsize=13); ax1.grid(alpha=0.3, which="both"); ax1.legend(fontsize=9, ncol=3)
    ax2.set_xlabel("프레임"); ax2.set_ylabel("L / E"); ax2.set_ylim(0, 1)
    ax2.set_title("2. 잠긴 비율 L/E (셀 아핀 하나로 못 푸는 몫)", fontsize=13); ax2.grid(alpha=0.3); ax2.legend(fontsize=10)
    ed = np.logspace(-6, 0, 61)
    for i, f in enumerate([f for f in a.hist_frames if f < T][:3]):
        ax = fig.add_subplot(gs[1, 2 * i:2 * i + 2])
        txt = []
        for t in TR:
            Z = np.load(f"{a.out}/cells_{t['key']}.npz"); o = Z["offsets"]
            E_ = Z["E"][o[f]:o[f + 1]]; L_ = Z["L"][o[f]:o[f + 1]]
            ok = E_ > 0
            r = L_[ok] / E_[ok]
            h, _ = np.histogram(np.clip(r, ed[0], ed[-1]), ed)
            ax.stairs(h / max(r.size, 1), ed, color=t["col"], lw=1.6, label=t["lab"])
            txt.append(f"{t['lab']}: 중앙 {np.median(r):.2e}, 0 (1 입자) {np.mean(r == 0) * 100:.1f}%, 셀 {r.size}")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(ed[0], 1)
        ax.set_xlabel("셀별 L_c / E_c (가로축 로그, 1e-6 아래·0 은 첫 칸)"); ax.set_ylabel("셀 비율 (세로축 로그)")
        ax.set_title(f"{3 + i}. 프레임 {f} 셀별 잠긴 비율 분포", fontsize=12)
        ax.text(0.02, 0.98, "\n".join(txt), transform=ax.transAxes, va="top", fontsize=8.5, bbox=dict(fc="white", ec="0.7", alpha=0.85))
        ax.grid(alpha=0.25)
        if i == 0:
            ax.legend(fontsize=9, loc="lower left")
    fig.suptitle(f"ficus_v0 셀별 잠긴 탄성 에너지 — 셀(Kuhn 사면체, node_h {a.node_h}, 각자 현재 위치) 하나에 아핀 변형 하나를 더 줘도 남는 에너지\n"
                 "주의: C¹ 은 셀 안 매핑이 3 차라서 L(아핀 하나로 못 푸는 몫)이 실제 잠긴 양보다 과대 추정", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(a.fig); plt.close(fig)
    say(f"[그림] {a.fig}")

# ---------------- 색 (영상용, 공통 범위) ----------------
cmap = matplotlib.colormaps["turbo"]
LG = {t["key"]: np.log10(np.load(f"{a.out}/dens_{t['key']}.npy") + a.eps) for t in TR}
both = np.concatenate([v[1:].ravel() for v in LG.values()])
lo, hi = (float(v) for v in np.percentile(both, a.q))
del both
json.dump(dict(lo=lo, hi=hi, eps=a.eps, q=a.q, cmap="turbo", unit="Pa (J/m^3)", quantity="셀 잠긴 에너지 밀도 L_c / Σ_{p∈c} V_p",
               runs=[t["key"] for t in TR], frames=f"1..{T - 1}"), open(f"{a.out}/range.json", "w"), ensure_ascii=False, indent=1)
say(f"[색 범위] log10(밀도+{a.eps}) [{lo:.4f}, {hi:.4f}] (세 궤적 1..{T - 1} 프레임 합친 {a.q} 분위수)")
for k, v in LG.items():
    Cc = np.empty(v.shape + (3,), np.float16)
    for f in range(v.shape[0]):
        Cc[f] = cmap(np.clip((v[f] - lo) / (hi - lo), 0.0, 1.0))[:, :3].astype(np.float16)
    np.save(f"{a.out}/col_{k}.npy", Cc)
say("LOCKED_CALC_DONE")
