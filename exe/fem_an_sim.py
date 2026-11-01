"""절 AN: 암시적 B-스플라인 하이브리드 판 (검증판: 틀은 torch, 적분점 커널은 lib/anchorflow/fem_an_cuda.py, 전부 float64).
프레임: P2G 한 번 (입자 -> 노드 질량·운동량, 입자 -> 칸 가우스 점 F_q = 절 AI A1 회전·신장 분리 평균, 가우스 점 부피·질량 = 칸 입자 합 / 8)
  -> 풀이 K 걸음 (입자 배열을 읽거나 쓰지 않음; 노드·가우스 점 배열만) -> G2P 한 번 (x += Σ w U, v = Σ w v_n, F_p ← (I + Σ U ⊗ ∇w) F_p).
걸음 (길이 Δt = h / K): 후퇴 오일러 (be, τ = Δt, y = U_{k+1}) 또는 암시적 중점 (mid, τ = Δt/2, y = (U_k + U_{k+1})/2, U_{k+1} = 2y − U_k, v_{k+1} = 2(y − U_k)/τ − v_k).
  둘 다 같은 최소화: E(y) = Σ_n m_n/(2τ²) |y − Ũ|² + Σ_q V_q ψ(X_q(y)) + 바닥 고착 항, Ũ = U_k + τ v_k + τ² g.
바닥 (노드 바닥 아님, 입자 판정 아님): 가우스 점 q 의 걸음 끝 위치 부호 거리 s_q, 정답 sticky 가중 w_st(s_q) (정답 격자 dx 의 평면 아래 노드 몫, s_q < 0 일 때만 = 절 AH 띠 0),
  노드 몫 r_n = Σ_q w_qn m_q w_st / Σ_q w_qn m_q, 걸음 고착 인자 k_n = (1 − r_n)^(Δt / dt_정답) (절 AH 정의 그대로), 고착은 v_{k+1} 의 접선·안쪽 법선 성분에만 (vfl_out, 절 AK.31).
  고착 항은 v_{k+1} 에 대한 2 차 감쇠 (계수 = 고착 인자를 정확히 내는 값) 이고, 변수 바꿈 y = ŷ + T η (T = √k P + (I − P)) 로 무한대 없이 푼다 (k = 0 이면 그 성분 고정).
뉴턴: 투영 헤시안 (FCR 해석 고유계, 음의 고유값 0), 행렬 없는 PCG (블록 야코비 전처리), 되돌이 선탐색 (아르미호), 고정 반복 예산, 첫 추정 = Ũ.
고착 몫 (r_n, P) 은 뉴턴 반복마다 그 반복의 y 로 다시 잰다 (반복 안에서는 고정)."""
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
from anchorflow import fem_an_cuda as AK                                            # noqa: E402
from anchorflow import fem_rt as RT                                                 # noqa: E402
from anchorflow import fem_rt_cuda as RTC                                           # noqa: E402
from anchorflow import phys_resid as PR                                             # noqa: E402
from anchorflow import simplex as SX                                                # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--frames", type=int, default=80)
ap.add_argument("--cell", type=float, default=1.0, help="셀 크기 = 정답 MPM dx 의 몇 배 (씬 공통 규칙)")
ap.add_argument("--floor", choices=["vel", "none"], default="vel")
ap.add_argument("--cfg_bc", action="store_true")
ap.add_argument("--region_density", action="store_true")
ap.add_argument("--pg_vdamp", action="store_true")
ap.add_argument("--K", type=int, default=2, help="프레임당 암시적 걸음 수")
ap.add_argument("--integ", choices=["mid", "be"], default="mid")
ap.add_argument("--newton", type=int, default=6)
ap.add_argument("--cg", type=int, default=100)
ap.add_argument("--cg_tol", type=float, default=1e-4, help="PCG 상대 잔차 (전처리 노름) 조기 종료")
ap.add_argument("--nt_tol", type=float, default=1e-6, help="뉴턴 조기 종료: |g_η| / |g_η 첫 반복| (고정 예산 안에서)")
ap.add_argument("--no_vfl_out", action="store_true")
ap.add_argument("--pc", choices=["jacobi", "mg"], default="jacobi", help="PCG 전처리: 노드 블록 야코비 또는 기하 멀티그리드 (세계 고정 격자, 간격 2 배씩)")
ap.add_argument("--mg_levels", type=int, default=5)
ap.add_argument("--mg_min", type=int, default=300, help="가장 거친 층 노드 수가 이 아래면 멈춤")
ap.add_argument("--mg_omega", type=float, default=0.6, help="감쇠 블록 야코비 평활 계수")
ap.add_argument("--mg_nc", type=int, default=10, help="가장 거친 층 평활 횟수")
ap.add_argument("--dump", action="store_true")
ap.add_argument("--tag", default="")
ap.add_argument("--warm", type=int, default=2)
ap.add_argument("--prof", action="store_true", help="커널별 GPU 시간 (CUDA 이벤트) 을 fem.jsonl 에 -- 단독 카드에서만 의미")
ap.add_argument("--check", action="store_true", help="프레임 0 에서 기울기·헤시안 곱을 유한 차분과 대조하고 끝")
a = ap.parse_args()
dev = "cuda"
RTC.RAWPTR[0] = True
D64 = torch.float64
os.makedirs(a.out, exist_ok=True)
d = torch.load(a.data, map_location="cpu", weights_only=False)
cfg = d["cfg"]
h = float(cfg["frame_dt"])
DT = h / a.K
TAU = DT if a.integ == "be" else 0.5 * DT
DM.REGION_DENSITY[0] = bool(a.region_density)
m64 = DM.traj_mass(d, dev).double()
EXT = float(d["_ext"])
rho = (d["_rho"].to(dev).double()) if a.region_density else None
vol64 = (m64 / rho) if a.region_density else (m64 / float(cfg["density"]))
mu, lam = PR.lame(cfg["E"], cfg["nu"])
mu, lam = float(mu), float(lam)
gv = torch.tensor([float(q) for q in cfg["g"]], device=dev, dtype=D64)
Np = int(d["x"].shape[1])
L = min(a.frames, int(d["x"].shape[0]) - 1)
x = d["x"][0].to(dev).double()
v = d["v"][0].to(dev).double()
F = DM.traj_F(d, h, dev)[0].to(dev).double().reshape(Np, 9)
DXG = float(cfg.get("grid_lim", 2.0)) / int(cfg.get("n_grid", 100))                  # 정답 MPM dx
NODE_H = a.cell * DXG * math.sqrt(2.0)                                                # 회전 정육면체 격자 칸 = node_h / √2
VD_S = float(cfg.get("grid_v_damping_scale", 1.0))
DAMP = (VD_S ** (DT / float(cfg["substep_dt"]))) if (a.pg_vdamp and VD_S < 1.0) else 1.0
lat = SX.lattice(NODE_H, SX.HZ_CUBE * NODE_H, device=dev, dtype=D64)
A64, Ai64 = lat.A, lat.Ai
dxl = float(A64[:, 0].norm())
FLOOR = a.floor == "vel"
VOUT = not a.no_vfl_out
if FLOOR:
    pl = [b for b in cfg["boundary_conditions"] if b.get("type") == "surface_collider"][0]
    PT = torch.tensor([float(q) for q in pl["point"]], device=dev, dtype=D64)
    NR = torch.tensor(pl["normal"], device=dev, dtype=D64)
    NR = NR / NR.norm()
    PN = float(PT @ NR)
    KEX = DT / float(cfg["substep_dt"])
cwave = math.sqrt((lam + 2 * mu) / float((rho.min() if rho is not None else cfg["density"])))
print(f"[절 AN 암시적 B-스플라인] 입자 {Np}, 프레임 {L}, 걸음 {a.K}/프레임 ({a.integ}, Δt {DT:.4e}, CFL 수 {DT * cwave / dxl:.1f}), 셀 {dxl:.5f} m (정답 dx {DXG:.4f} 의 {a.cell} 배), "
      f"뉴턴 {a.newton} x PCG {a.cg}, 바닥 {a.floor}{' vfl_out' if (FLOOR and VOUT) else ''}", flush=True)
OFF = torch.stack(torch.meshgrid(torch.arange(3), torch.arange(3), torch.arange(3), indexing="ij"), -1).reshape(27, 3).to(dev)
GG = 0.5 / math.sqrt(3.0)
I3 = torch.eye(3, device=dev, dtype=D64)


def bsw(fx, Ai_=None):
    Ai_ = Ai64 if Ai_ is None else Ai_
    w = torch.stack([0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2], 1)
    dw = torch.stack([fx - 1.5, -2 * (fx - 1), fx - 0.5], 1)
    wi, wj, wk = w[:, OFF[:, 0], 0], w[:, OFF[:, 1], 1], w[:, OFF[:, 2], 2]
    W = wi * wj * wk
    GW = torch.stack([dw[:, OFF[:, 0], 0] * wj * wk, wi * dw[:, OFF[:, 1], 1] * wk, wi * wj * dw[:, OFF[:, 2], 2]], -1) @ Ai_     # ∇_x = Aiᵀ ∇_y
    return W, GW


def rs_mean(Fp, w, inv, nq):
    """절 AI A1 회전·신장 분리 평균 (float64, CUDA 커널): 뒤집힌 입자가 든 점·비유한 점은 유클리드 평균 (절 AK.33 과 같은 정의)."""
    n = Fp.shape[0]
    Fc = Fp.contiguous()
    R = torch.empty(n, 9, device=dev, dtype=D64)
    LS = torch.empty(n, 9, device=dev, dtype=D64)
    neg = torch.empty(n, device=dev, dtype=torch.uint8)
    AK.launch("an_rsp", n, (np.int32(n), AK.P(Fc), AK.P(R), AK.P(LS), AK.P(neg)))
    W = w[:, None]
    sR = torch.zeros(nq, 9, device=dev, dtype=D64).index_add_(0, inv, W * R)
    sL = torch.zeros(nq, 9, device=dev, dtype=D64).index_add_(0, inv, W * LS)
    sF = torch.zeros(nq, 9, device=dev, dtype=D64).index_add_(0, inv, W * Fc)
    sw = torch.zeros(nq, device=dev, dtype=D64).index_add_(0, inv, w)
    bad = torch.zeros(nq, device=dev, dtype=torch.bool).index_put_((inv[neg.bool()],), torch.tensor(True, device=dev))
    RS = torch.empty(nq, 9, device=dev, dtype=D64)
    nf = torch.empty(nq, device=dev, dtype=torch.uint8)
    AK.launch("an_rsf", nq, (np.int32(nq), AK.P(sR), AK.P(sL), AK.P(sw), AK.P(RS), AK.P(nf)))
    bad = bad | nf.bool()
    return torch.where(bad[:, None], sF / sw.clamp_min(1e-300)[:, None], RS), int(bad.sum())


def wstk(s):
    """정답 sticky 가중 (절 AH 띠 0, fem_bsel_cuda.wstk 와 같은 식): 부호 거리 s 의 점이 정답 격자 (dx DXG) 에서 평면 아래 노드에 주는 2 차 B-스플라인 몫, s < 0 일 때만."""
    u = (s + PN) / DXG
    bf = torch.floor(u - 0.5)
    fx = u - bf
    w = torch.stack([0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2], -1)
    kk = bf[:, None] + torch.arange(3, device=dev, dtype=D64)[None]
    o = (w * ((kk * DXG - PN) < 0).double()).sum(-1)
    return torch.where(s < 0, o, torch.zeros_like(o))


class Elem:
    """가우스 점 집합 위 탄성 (커널 호출)."""

    def __init__(s, qn, G, Fq, Vq, na):
        s.qn, s.G, s.Fq, s.Vq, s.na, s.nq = qn, G, Fq, Vq, na, qn.shape[0]
        s.E = torch.empty(s.nq, device=dev, dtype=D64)
        s.H = torch.empty(s.nq, 81, device=dev, dtype=D64)
        s.nmv = 0

    def eval(s, y, mode):
        f = torch.zeros(s.na, 3, device=dev, dtype=D64)
        AK.launch("an_elem", s.nq, (np.int32(s.nq), AK.P(s.qn), AK.P(s.G), AK.P(s.Fq), AK.P(s.Vq), AK.P(y.contiguous()), np.float64(mu), np.float64(lam),
                                    np.int32(mode), AK.P(s.E), AK.P(f), AK.P(s.H)))
        return (float(s.E.sum()) if mode & 4 else None), f

    def mv(s, dy):
        o = torch.zeros(s.na, 3, device=dev, dtype=D64)
        AK.launch("an_mv", s.nq, (np.int32(s.nq), AK.P(s.qn), AK.P(s.G), AK.P(s.H), AK.P(s.Vq), AK.P(dy.contiguous()), AK.P(o)))
        s.nmv += 1
        return o

    def bd(s):
        B = torch.zeros(s.na, 9, device=dev, dtype=D64)
        AK.launch("an_bd", s.nq, (np.int32(s.nq), AK.P(s.qn), AK.P(s.G), AK.P(s.H), AK.P(s.Vq), AK.P(B)))
        return B.reshape(s.na, 3, 3)


_CK = {}


def ckern(ch):
    """2 차 B-스플라인 두 배 관계 계수 (1, 3, 3, 1)/4 의 텐서곱 (groups = ch)."""
    if ch not in _CK:
        c = torch.tensor([1.0, 3.0, 3.0, 1.0], device=dev, dtype=D64) / 4
        k3 = c[:, None, None] * c[None, :, None] * c[None, None, :]
        _CK[ch] = k3[None, None].repeat(ch, 1, 1, 1, 1).contiguous()
    return _CK[ch]


def restrict_dense(xd, nbf, Nc):
    """조밀 배열 (nx, ny, nz, ch) -> 거친 층 (Nc..., ch): r_c = Pᵀ r. 거친 노드 J 는 고운 층 노드 번호 2J − 3/2 자리, 고운 노드 2J − 3 + k (k = 0..3)."""
    ch = xd.shape[-1]
    t = xd.permute(3, 0, 1, 2)[None]
    npad = [2 * n - 4 for n in Nc]
    t = torch.nn.functional.pad(t, (0, npad[2] - nbf[2], 0, npad[1] - nbf[1], 0, npad[0] - nbf[0]))
    o = torch.nn.functional.conv3d(t, ckern(ch), stride=2, padding=3, groups=ch)
    return o[0].permute(1, 2, 3, 0)


def prolong_dense(xc, nbf):
    ch = xc.shape[-1]
    o = torch.nn.functional.conv_transpose3d(xc.permute(3, 0, 1, 2)[None], ckern(ch), stride=2, padding=3, groups=ch)
    return o[0, :, :nbf[0], :nbf[1], :nbf[2]].permute(1, 2, 3, 0)


class MG:
    """기하 멀티그리드 전처리 (대칭 V 순환). 층 ℓ+1 = 간격 2 배 세계 고정 격자의 2 차 B-스플라인 (두 배 관계로 고운 층 기저의 정확한 선형 결합 = 연장 P),
    거친 층 연산자 = 다시 이산화: 거친 칸 2x2x2 가우스 점, F = 그 팔분 영역 고운 점의 지금 X 의 회전·신장 분리 평균 (절 AI A1, 같은 rs_mean), 부피 = 칸 합 / 8, 투영 헤시안 (같은 커널);
    질량 = Pᵀ m (집중). 0 층 연산자 = Mi + T K T 그대로, 0 -> 1 전달은 T 를 끼운 Pᵀ T / T P. 평활 = 감쇠 블록 야코비 (ω), 가장 거친 층 = 평활 mg_nc 번."""

    def __init__(s, ctx, y, el, Mi, T, Bd):
        s.T = T
        s.L = [dict(el=el, Mi=Mi, D=torch.linalg.inv(Mi[..., None] * I3 + T @ Bd @ T), nb=ctx["nb"], act=ctx["act"])]
        X = el.Fq + torch.einsum("qna,qnb->qab", y[el.qn.long()], el.G).reshape(-1, 9)
        tq, V, nbf, mdn = ctx["tq"], el.Vq, ctx["nb"], ctx["mnd"].reshape(*ctx["nb"], 1)
        Ai_ = Ai64
        for lv in range(1, a.mg_levels):
            Nc = [n // 2 + 3 for n in nbf]
            tc = (tq + 1.5) / 2
            base = torch.floor(tc - 0.5).long()
            fx = tc - base
            cid = (base[:, 0] * Nc[1] + base[:, 1]) * Nc[2] + base[:, 2]
            uc, cinv = torch.unique(cid, return_inverse=True)
            nc = uc.numel()
            oi = ((fx >= 1.0).long() * torch.tensor([4, 2, 1], device=dev)).sum(1)
            qinv = cinv * 8 + oi
            nq = nc * 8
            Fc, _ = rs_mean(X, V, qinv, nq)
            cF, _ = rs_mean(X, V, cinv, nc)
            has = torch.zeros(nq, device=dev, dtype=torch.bool).index_put_((qinv,), torch.tensor(True, device=dev))
            Fc = torch.where(has[:, None], Fc, cF.repeat_interleave(8, 0)).contiguous()
            Vc = (torch.zeros(nc, device=dev, dtype=D64).index_add_(0, cinv, V).repeat_interleave(8) / 8.0).contiguous()
            o8 = torch.arange(8, device=dev)
            qfx = (1.0 + GG * (2 * torch.stack([o8 // 4, (o8 % 4) // 2, o8 % 2], 1).double() - 1)).repeat(nc, 1)
            Ai_ = Ai_ / 2
            _, QGW = bsw(qfx, Ai_)
            cb = torch.stack([uc // (Nc[1] * Nc[2]), (uc // Nc[2]) % Nc[1], uc % Nc[2]], 1).repeat_interleave(8, 0)
            qnid = ((cb[:, None, 0] + OFF[None, :, 0]) * Nc[1] + cb[:, None, 1] + OFF[None, :, 1]) * Nc[2] + cb[:, None, 2] + OFF[None, :, 2]
            mdc = restrict_dense(mdn, nbf, Nc)
            mflat = mdc.reshape(-1)
            act = torch.nonzero(mflat > 0).flatten()
            na = act.numel()
            cmp_ = torch.full((mflat.numel(),), -1, device=dev, dtype=torch.long)
            cmp_[act] = torch.arange(na, device=dev)
            qn = cmp_[qnid]
            assert bool((qn >= 0).all())
            G = torch.einsum("qab,qna->qnb", Fc.reshape(nq, 3, 3), QGW).contiguous()
            elc = Elem(qn.int().contiguous(), G, Fc, Vc, na)
            elc.eval(torch.zeros(na, 3, device=dev, dtype=D64), 2)
            Mic = (mflat[act] / (TAU * TAU))[:, None]
            s.L.append(dict(el=elc, Mi=Mic, D=torch.linalg.inv(Mic[..., None] * I3 + elc.bd()), nb=Nc, act=act))
            tq, X, V, nbf, mdn = (cb.double() + qfx), Fc, Vc, Nc, mdc
            if na < a.mg_min:
                break
        s.nl = len(s.L)

    def op(s, l, z):
        lv = s.L[l]
        if l == 0:
            return lv["Mi"] * z + (s.T @ lv["el"].mv((s.T @ z[..., None]).squeeze(-1))[..., None]).squeeze(-1)
        return lv["Mi"] * z + lv["el"].mv(z)

    def sm(s, l, r):
        return (s.L[l]["D"] @ r[..., None]).squeeze(-1)

    def tr(s, l, r):
        """층 l 의 압축 벡터 -> 층 l+1 (Pᵀ, 0 층은 Tᵀ 먼저)."""
        A_, B_ = s.L[l], s.L[l + 1]
        if l == 0:
            r = (s.T @ r[..., None]).squeeze(-1)
        dn = torch.zeros(int(np.prod(A_["nb"])), 3, device=dev, dtype=D64)
        dn[A_["act"]] = r
        o = restrict_dense(dn.reshape(*A_["nb"], 3), A_["nb"], B_["nb"]).reshape(-1, 3)
        return o[B_["act"]]

    def pr(s, l, zc):
        A_, B_ = s.L[l], s.L[l + 1]
        dn = torch.zeros(int(np.prod(B_["nb"])), 3, device=dev, dtype=D64)
        dn[B_["act"]] = zc
        z = prolong_dense(dn.reshape(*B_["nb"], 3), A_["nb"]).reshape(-1, 3)[A_["act"]]
        if l == 0:
            z = (s.T @ z[..., None]).squeeze(-1)
        return z

    def vc(s, l, r):
        w = a.mg_omega
        z = w * s.sm(l, r)
        if l == s.nl - 1:
            for _ in range(a.mg_nc - 1):
                z = z + w * s.sm(l, r - s.op(l, z))
            return z
        z = z + s.pr(l, s.vc(l + 1, s.tr(l, r - s.op(l, z))))
        return z + w * s.sm(l, r - s.op(l, z))

    def __call__(s, r):
        return s.vc(0, r)


STAT = {}


def solve_step(el, mn, Uk, vk, fixd, vbc, QW, qn, xq0, mq, ctx):
    """한 걸음: y 를 구해 (U_{k+1}, v_{k+1}) 를 돌려준다. 입자 배열은 쓰지 않는다."""
    na = mn.shape[0]
    Ut = Uk + TAU * vk + TAU * TAU * gv
    z0 = Uk + (0.5 * TAU * vk if a.integ == "mid" else 0.0)                         # v_{k+1} = α (y − z0)
    al = (2.0 / TAU) if a.integ == "mid" else (1.0 / TAU)
    Mi = (mn / (TAU * TAU))[:, None]
    mq_n = torch.zeros(na, device=dev, dtype=D64).index_add_(0, qn.reshape(-1), (QW * mq[:, None]).reshape(-1)) if FLOOR else None
    y = Ut.clone()
    if fixd is not None:
        y[fixd] = Uk[fixd] + TAU * vbc[fixd]
    g0n = None
    nst = dict(nt=0, cg=0, ls=0, gr=None, stuck=0)
    for it in range(a.newton):
        # 고착 몫 (이 반복의 y 로) -> 변수 바꿈 y = ŷ + T η
        if FLOOR:
            Ue = (2 * y - Uk) if a.integ == "mid" else y
            xq = xq0 + (QW[..., None] * Ue[qn]).sum(1)
            sq = (xq - PT) @ NR
            wst = wstk(sq)
            an = torch.zeros(na, device=dev, dtype=D64).index_add_(0, qn.reshape(-1), (QW * (mq * wst)[:, None]).reshape(-1))
            r = torch.where(mq_n > 0, an / mq_n.clamp_min(1e-300), torch.zeros_like(an)).clamp(0.0, 1.0)     # 몫 (정의상 [0,1]; 반올림만)
            k = torch.exp(KEX * torch.log1p(-r))
            vn1 = al * (y - z0)
            if VOUT:
                outw = (vn1 @ NR) > 0
                Pm = torch.where(outw[:, None, None], I3 - NR[:, None] * NR[None, :], I3.expand(na, 3, 3))
            else:
                Pm = I3.expand(na, 3, 3)
            sk = torch.sqrt(k)[:, None, None]
            T = sk * Pm + (I3 - Pm)
            yh = Ut + ((1 - k)[:, None, None] * Pm @ (z0 - Ut)[..., None]).squeeze(-1)
            eta = ((I3 - Pm) @ (y - Ut)[..., None]).squeeze(-1) + torch.where(  # η_P = P (y − ŷ) / √k
                sk.squeeze(-1) > 0, (Pm @ (y - yh)[..., None]).squeeze(-1) / sk.squeeze(-1).clamp_min(1e-300), torch.zeros_like(y))
            nst["stuck"] = int((r > 0).sum())
        else:
            T = I3.expand(na, 3, 3)
            yh = Ut
            eta = y - Ut
        if fixd is not None:
            T = torch.where(fixd[:, None, None], torch.zeros_like(T), T)
            yh = torch.where(fixd[:, None], Uk + TAU * vbc, yh)
            eta = torch.where(fixd[:, None], torch.zeros_like(eta), eta)
        Tc = T.contiguous()
        yfun = lambda e_: yh + (Tc @ e_[..., None]).squeeze(-1)
        y = yfun(eta)
        Ee, fe = el.eval(y, 1 | 2 | 4)
        E0 = 0.5 * float((Mi * eta * eta).sum()) + Ee
        g = Mi * eta + (Tc @ fe[..., None]).squeeze(-1)
        gn = float(g.norm())
        if g0n is None:
            g0n = max(gn, 1e-300)
        nst["gr"] = gn / g0n
        if gn / g0n < a.nt_tol and it > 0:
            break
        # PCG: (Mi + Tᵀ K T) δ = −g, 블록 야코비 전처리
        Bd = el.bd()
        if a.pc == "mg":
            mg_ = MG(ctx, y, el, Mi, Tc, Bd)
            nst["lv"] = mg_.nl
            prec = mg_
        else:
            Pc = torch.linalg.inv(Mi[..., None] * I3 + Tc @ Bd @ Tc)
            prec = lambda r_: (Pc @ r_[..., None]).squeeze(-1)
        Aop = lambda p_: Mi * p_ + (Tc @ el.mv((Tc @ p_[..., None]).squeeze(-1))[..., None]).squeeze(-1)
        dl = torch.zeros_like(g)
        rr = -g
        zz = prec(rr)
        pp = zz.clone()
        rz = float((rr * zz).sum())
        rz0 = rz
        for ci in range(a.cg):
            Ap = Aop(pp)
            pAp = float((pp * Ap).sum())
            if not (pAp > 0):
                break
            alp = rz / pAp
            dl = dl + alp * pp
            rr = rr - alp * Ap
            zo = zz
            zz = prec(rr)
            rzn = float((rr * zz).sum())
            nst["cg"] += 1
            if rzn <= (a.cg_tol ** 2) * rz0:
                break
            pp = zz + (float((rr * (zz - zo)).sum()) / rz) * pp                      # 유연 PCG (폴라크-리비에르)
            rz = rzn
        # 되돌이 선탐색 (아르미호)
        gd = float((g * dl).sum())
        st = 1.0
        ok = False
        for li in range(12):
            et = eta + st * dl
            Et, _ = el.eval(yfun(et), 4)
            E1 = 0.5 * float((Mi * et * et).sum()) + Et
            if math.isfinite(E1) and E1 <= E0 + 1e-4 * st * gd:
                ok = True
                break
            st *= 0.5
            nst["ls"] += 1
        if not ok:
            st = 0.0
        eta = eta + st * dl
        y = yfun(eta)
        nst["nt"] += 1
        if st == 0.0:
            break
    if a.integ == "mid":
        U1 = 2 * y - Uk
        v1 = al * (y - z0)
    else:
        U1 = y
        v1 = (y - Uk) / TAU
    if fixd is not None:
        v1 = torch.where(fixd[:, None], vbc, v1)
    v1 = v1 * DAMP
    for kk_ in ("nt", "cg", "ls"):
        STAT[kk_] = STAT.get(kk_, 0) + nst[kk_]
    STAT["gr"] = max(STAT.get("gr", 0.0), nst["gr"] or 0.0)
    STAT["stuck"] = nst["stuck"]
    STAT["lv"] = nst.get("lv", 1)
    return U1, v1


def frame(fi, x, v, F):
    y = x @ Ai64.T
    base = torch.floor(y - 0.5).long()
    bmin = base.min(0).values
    bb = base - bmin
    nb = (bb.max(0).values + 3).tolist()
    nbt = nb[0] * nb[1] * nb[2]
    cid = (bb[:, 0] * nb[1] + bb[:, 1]) * nb[2] + bb[:, 2]
    fx = y - base
    W, GW = bsw(fx)
    nid = ((bb[:, None, 0] + OFF[None, :, 0]) * nb[1] + bb[:, None, 1] + OFF[None, :, 1]) * nb[2] + bb[:, None, 2] + OFF[None, :, 2]
    mnd = torch.zeros(nbt, device=dev, dtype=D64).index_add_(0, nid.reshape(-1), (W * m64[:, None]).reshape(-1))
    act = torch.nonzero(mnd > 0).flatten()
    na = act.numel()
    cmp_ = torch.full((nbt,), -1, device=dev, dtype=torch.long)
    cmp_[act] = torch.arange(na, device=dev)
    nidc = cmp_[nid]
    mn = mnd[act]
    mv = torch.zeros(na, 3, device=dev, dtype=D64).index_add_(0, nidc.reshape(-1), (W[..., None] * (m64[:, None] * v)[:, None, :]).reshape(-1, 3))
    vn = mv / mn[:, None]
    # 가우스 점 (칸마다 8 점, 같은 가중 = 칸 입자 부피 / 8, 질량 = 칸 입자 질량 / 8)
    uc, cinv = torch.unique(cid, return_inverse=True)
    nc = uc.numel()
    oi = ((fx >= 1.0).long() * torch.tensor([4, 2, 1], device=dev)).sum(1)
    qinv = cinv * 8 + oi
    nq = nc * 8
    Vc = torch.zeros(nc, device=dev, dtype=D64).index_add_(0, cinv, vol64)
    Mc = torch.zeros(nc, device=dev, dtype=D64).index_add_(0, cinv, m64)
    qV = Vc.repeat_interleave(8) / 8.0
    mq = Mc.repeat_interleave(8) / 8.0
    qF, nbad = rs_mean(F, vol64, qinv, nq)
    cF, _ = rs_mean(F, vol64, cinv, nc)
    has = torch.zeros(nq, device=dev, dtype=torch.bool).index_put_((qinv,), torch.tensor(True, device=dev))
    qF = torch.where(has[:, None], qF, cF.repeat_interleave(8, 0))
    o8 = torch.arange(8, device=dev)
    qb = torch.stack([o8 // 4, (o8 % 4) // 2, o8 % 2], 1).double()
    qfx = (1.0 + GG * (2 * qb - 1)).repeat(nc, 1)
    QW, QGW = bsw(qfx)
    cz = uc % nb[2]
    cy = (uc // nb[2]) % nb[1]
    cx = uc // (nb[1] * nb[2])
    cb = torch.stack([cx, cy, cz], 1).repeat_interleave(8, 0)
    qnid = ((cb[:, None, 0] + OFF[None, :, 0]) * nb[1] + cb[:, None, 1] + OFF[None, :, 1]) * nb[2] + cb[:, None, 2] + OFF[None, :, 2]
    qn = cmp_[qnid]
    assert bool((qn >= 0).all())
    Fq3 = qF.reshape(nq, 3, 3)
    G = torch.einsum("qab,qna->qnb", Fq3, QGW).contiguous()                           # G_qn = F_qᵀ ∇w_qn
    xq0 = ((cb + bmin).double() + qfx) @ A64.T
    el = Elem(qn.int().contiguous(), G, qF.contiguous(), qV.contiguous(), na)
    if a.check:
        torch.manual_seed(0)
        for sc_ in (1e-4, 1e-3):
            y0 = sc_ * dxl * torch.randn(na, 3, device=dev, dtype=D64)
            dd = dxl * torch.randn(na, 3, device=dev, dtype=D64)
            E0_, f0 = el.eval(y0, 1 | 2 | 4)
            ep = 1e-7
            Ep, fp = el.eval(y0 + ep * dd, 1 | 4)
            Em, fm = el.eval(y0 - ep * dd, 1 | 4)
            el.eval(y0, 2)
            Hd = el.mv(dd)
            fd = (fp - fm) / (2 * ep)
            print(f"[대조] 변위 {sc_} 칸: dE 유한 차분 {(Ep - Em) / (2 * ep):.6e} vs f·d {float((f0 * dd).sum()):.6e}; |Hd − 유한 차분| / |유한 차분| {float((Hd - fd).norm() / fd.norm()):.3e}; "
                  f"블록 대각 대칭 {float((el.bd() - el.bd().transpose(1, 2)).abs().max()):.2e}", flush=True)
        sys.exit(0)
    fixd = vbc = None
    if a.cfg_bc:
        kq = torch.stack([act // (nb[1] * nb[2]), (act // nb[2]) % nb[1], act % nb[2]], 1) + bmin
        npos = (kq.double() @ A64.T).float()
        fx_, vb_ = RT.cfg_bc_nodes(cfg, [], npos, torch.zeros(1, 4, dtype=torch.long, device=dev), torch.zeros(1, 4, device=dev), m64[:1].float(), fi * h, h)
        if fx_ is not None:
            fixd, vbc = fx_.bool(), vb_.double()
    STAT.clear()
    STAT.update(nq=nq, na=na, nbad=nbad)
    ctx = dict(nb=nb, act=act, mnd=mnd, tq=cb.double() + qfx)
    U = torch.zeros(na, 3, device=dev, dtype=D64)
    for _ in range(a.K):
        U, vn = solve_step(el, mn, U, vn, fixd, vbc, QW, qn, xq0, mq, ctx)
    STAT["nmv"] = el.nmv
    # G2P (프레임 끝 한 번)
    Ug = U[nidc]
    x2 = x + (W[..., None] * Ug).sum(1)
    v2 = (W[..., None] * vn[nidc]).sum(1)
    Am = I3 + torch.einsum("pna,pnb->pab", Ug, GW)
    F2 = (Am @ F.reshape(Np, 3, 3)).reshape(Np, 9)
    return x2, v2, F2


flog = open(os.path.join(a.out, "fem.jsonl"), "w")
errs, secs = [], []
div_at = None
PRED = torch.empty(L, Np, 3) if a.dump else None
AK.mod()
AK.PROF["on"] = bool(a.prof)
for fi in range(L):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    x2, v2, F2 = frame(fi, x, v, F)
    torch.cuda.synchronize()
    dt_ = time.perf_counter() - t0
    gt = d["x"][fi + 1].to(dev).double()
    e = float((x2 - gt).norm(dim=-1).mean()) / EXT
    fin = bool(torch.isfinite(x2).all()) and math.isfinite(e)
    errs.append(e if fin else float("nan"))
    secs.append(dt_)
    rec = dict(t=fi, err=e, sec=dt_, **{k_: v_ for k_, v_ in STAT.items()})
    if a.prof:
        rec["kernel_ms"] = AK.prof_take()
    flog.write(json.dumps(rec) + "\n")
    flog.flush()
    print(f"  프레임 {fi + 1}/{L} {dt_:.2f} s 오차 {100 * e:.3f}% 뉴턴 {STAT.get('nt')} PCG {STAT.get('cg')} 선탐색 {STAT.get('ls')} 층 {STAT.get('lv')} 헤시안 곱 {STAT.get('nmv')} 최대 |g|/|g0| {STAT.get('gr', 0):.1e} "
          f"고착 노드 {STAT.get('stuck')} 점 {STAT.get('nq')} 노드 {STAT.get('na')} 뒤집힌 점 {STAT.get('nbad')} max|v| {float(v2.norm(dim=-1).max()):.3g}", flush=True)
    if PRED is not None:
        PRED[fi] = x2.float().cpu()
    x, v, F = x2, v2, F2
    if not fin or e > 2.0:
        div_at = fi + 1
        print(f"[발산] 프레임 {div_at}", flush=True)
        break
sc = secs[a.warm:]
summ = dict(args=vars(a), frames=L, frames_done=len(errs), diverged=div_at, err_mean=(float(np.mean(errs)) if div_at is None else None),
            err_mean_done=float(np.nanmean(errs)), s_mean=float(np.mean(sc)) if sc else None, dt=DT, cell=dxl)
json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1, ensure_ascii=False)
if PRED is not None and div_at is None:
    torch.save({"pred": PRED}, os.path.join(a.out, "pred.pt"))
print(f"[롤아웃] {a.tag} 암시적 B-스플라인 {a.integ} K {a.K} 셀 {a.cell}: {len(errs)}/{L} 프레임{' (발산 %d)' % div_at if div_at else ''}, 평균 {100 * summ['err_mean_done']:.3f}%, "
      f"프레임당 {summ['s_mean']} s (검증판)", flush=True)
