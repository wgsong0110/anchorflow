"""절 AK.21 진단 (계산 노드, 측정 밖): 남는 탄성이 프레임 시작 노드 힘으로 돌아오는가.
우리 덤프 (roll_fem.pt: 위치·F) 와 정답 (같은 프레임 위치·F) 각각에 대해, 정답 MPM 과 같은 격자 (dx = 정답 dx, 축 정렬) 2 차 B-스플라인으로
프레임 시작 (변위 0) 노드 내부력 f_n = −Σ_p V_p P(F_p) F_pᵀ ∇w_n(x_p) (정답 MPM 힘 식 그대로, 표준 FCR) 을 계산해
  E = Σ V ψ (입자 탄성), |f| 합, 힘 일률 Π = Σ_n |f_n|² / m_n (첫 서브스텝에 생기는 운동 에너지 증가율의 척도), 효율 Π / E 와
  자기 평형 비 = |Σ_n f_n 의 셀 국소 합| 대신 '노드 힘 크기 / 입자 응력 크기' (Σ_n |f_n| ÷ Σ_p V_p |τ_p| |∇w| 상계) 를 우리 / 정답으로 비교.
사용: python exe/fem_ak_force.py <roll_fem.pt> <data.pt> <정답 최대 압축 프레임> [창 15]"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import torch                                                                         # noqa: E402
from anchorflow import deform_model as DM                                           # noqa: E402
from anchorflow import phys_resid as PR                                             # noqa: E402

rp, dp, f0 = sys.argv[1], sys.argv[2], int(sys.argv[3])
W = int(sys.argv[4]) if len(sys.argv) > 4 else 15
dev = "cuda"
R = torch.load(rp, map_location="cpu", weights_only=False)
d = torch.load(dp, map_location="cpu", weights_only=False)
cfg = d["cfg"]
h = float(cfg["frame_dt"])
m = DM.traj_mass(d, dev).double()
V = m / float(cfg["density"])
mu, lam = PR.lame(cfg["E"], cfg["nu"])
GF = DM.traj_F(d, h, "cpu")
dx = float(cfg.get("grid_lim", 2.0)) / int(cfg.get("n_grid", 100))
_pl = [b for b in cfg["boundary_conditions"] if b.get("type") == "surface_collider"][0]
PTt = torch.tensor([float(q) for q in _pl["point"]], device=dev, dtype=torch.float64)
NRt = torch.tensor([float(q) for q in _pl["normal"]], device=dev, dtype=torch.float64)
NRt = NRt / NRt.norm()
q0 = d["x"][0].to(dev).double()
q0 = q0 - q0.mean(0)
OFF = torch.stack(torch.meshgrid(torch.arange(3), torch.arange(3), torch.arange(3), indexing="ij"), -1).reshape(27, 3).to(dev)


def cof(X):
    return torch.stack([torch.cross(X[:, 1], X[:, 2], dim=-1), torch.cross(X[:, 2], X[:, 0], dim=-1), torch.cross(X[:, 0], X[:, 1], dim=-1)], 1)


def stats(x, F):
    U_, S_, Vh_ = torch.linalg.svd(F)
    D = torch.ones(F.shape[0], 3, device=dev, dtype=F.dtype)
    D[:, 2] = torch.sign(torch.linalg.det(U_ @ Vh_))
    Rr = U_ @ torch.diag_embed(D) @ Vh_
    J = torch.linalg.det(F)
    P = 2 * mu * (F - Rr) + lam * (J - 1)[:, None, None] * cof(F)
    tau = P @ F.transpose(1, 2)
    E = float((V * (mu * ((F - Rr) ** 2).sum((1, 2)) + 0.5 * lam * (J - 1) ** 2)).sum())
    y = x / dx
    base = torch.floor(y - 0.5).long()
    fx = y - base
    w1 = torch.stack([0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2], 1)
    d1 = torch.stack([fx - 1.5, -2 * (fx - 1), fx - 0.5], 1)
    wi, wj, wk = w1[:, OFF[:, 0], 0], w1[:, OFF[:, 1], 1], w1[:, OFF[:, 2], 2]
    Wt = wi * wj * wk
    GW = torch.stack([d1[:, OFF[:, 0], 0] * wj * wk, wi * d1[:, OFF[:, 1], 1] * wk, wi * wj * d1[:, OFF[:, 2], 2]], -1) / dx
    keys = base[:, None, :] + OFF[None]
    kmin = keys.reshape(-1, 3).min(0).values
    kk = keys - kmin
    B = int(kk.max()) + 1
    uq, nid = torch.unique(((kk[..., 0] * B + kk[..., 1]) * B + kk[..., 2]).reshape(-1), return_inverse=True)
    M = int(uq.numel())
    fp = -torch.einsum("pab,pkb->pka", V[:, None, None] * tau, GW)
    f = torch.zeros(M, 3, device=dev, dtype=torch.float64).index_add_(0, nid, fp.reshape(-1, 3))
    mn = torch.zeros(M, device=dev, dtype=torch.float64).index_add_(0, nid, (Wt * m[:, None]).reshape(-1))
    ok = mn > 1e-12
    Pi = float(((f[ok] ** 2).sum(-1) / mn[ok]).sum())
    fsum = float(f.norm(dim=-1).sum())
    ub = float((V * tau.reshape(-1, 9).norm(dim=-1) * GW.norm(dim=-1).sum(1)).sum())   # 노드 힘 크기 상계 (모든 입자 힘이 같은 방향일 때)
    # 절 AK.23: 바닥 접촉 노드 (평면 위 1 cm 안 입자가 가중을 주는 노드) 의 힘·일률 몫
    near = ((x - PTt) @ NRt) < 0.01
    cn = torch.zeros(M, dtype=torch.bool, device=dev)
    cn[nid.view(-1, 27)[near].reshape(-1)] = True
    fs_c = float(f[cn].norm(dim=-1).sum()) / max(fsum, 1e-30)
    Pi_c = float(((f[ok & cn] ** 2).sum(-1) / mn[ok & cn]).sum()) / max(Pi, 1e-30)
    # 절 AK.23: 몸 좌표 변위의 다항식 모드 몫 (0 프레임 대비 Kabsch 를 뺀 변위를 상수·1 차·2 차 기저에 사영 -- 전체 사영, 이웃 추정 아님)
    c = x.mean(0)
    U2, S2, Vh2 = torch.linalg.svd(q0.T @ (x - c))
    Dg = torch.eye(3, device=dev, dtype=torch.float64)
    Dg[2, 2] = torch.sign(torch.det(Vh2.T @ U2.T))
    Rm = Vh2.T @ Dg @ U2.T
    dd = (x - c) @ Rm - q0
    qs = q0 / q0.abs().max()
    B1 = torch.cat([torch.ones_like(qs[:, :1]), qs], 1)
    B2 = torch.cat([B1, qs[:, [0]] * qs[:, [0]], qs[:, [1]] * qs[:, [1]], qs[:, [2]] * qs[:, [2]], qs[:, [0]] * qs[:, [1]], qs[:, [0]] * qs[:, [2]], qs[:, [1]] * qs[:, [2]]], 1)
    tot = float((dd ** 2).sum())
    r1 = dd - B1 @ torch.linalg.lstsq(B1, dd).solution
    r2 = dd - B2 @ torch.linalg.lstsq(B2, dd).solution
    lin = tot - float((r1 ** 2).sum())
    quad = float((r1 ** 2).sum()) - float((r2 ** 2).sum())
    rest = float((r2 ** 2).sum())
    rms = lambda q_: (q_ / x.shape[0]) ** 0.5
    return dict(E=E, fsum=fsum, Pi=Pi, eff=Pi / max(E, 1e-30), selfeq=fsum / max(ub, 1e-30), fs_c=fs_c, Pi_c=Pi_c,
                d_lin=rms(lin), d_quad=rms(quad), d_rest=rms(rest))


out = []
for f in range(f0, f0 + W):
    o = stats(R["pred"][f - 1].to(dev).double(), R["F"][f - 1].to(dev).double().reshape(-1, 3, 3))
    g = stats(d["x"][f].to(dev).double(), GF[f].to(dev).double().reshape(-1, 3, 3))
    out.append(dict(f=f, ours=o, gt=g))
    print(f"f{f}: 접촉 노드 힘 몫 {o['fs_c']:.3f}/{g['fs_c']:.3f} 일률 몫 {o['Pi_c']:.3f}/{g['Pi_c']:.3f}  변위 RMS (mm) 1 차 {1e3 * o['d_lin']:.1f}/{1e3 * g['d_lin']:.1f} 2 차 {1e3 * o['d_quad']:.1f}/{1e3 * g['d_quad']:.1f} 나머지 {1e3 * o['d_rest']:.1f}/{1e3 * g['d_rest']:.1f}", flush=True)
    print(f"f{f}: 탄성 {o['E']:.0f}/{g['E']:.0f} J  노드 힘 합 {o['fsum']:.0f}/{g['fsum']:.0f} N  힘 일률 Π {o['Pi']:.3g}/{g['Pi']:.3g}  Π/E {o['eff']:.3g}/{g['eff']:.3g}  힘 크기 ÷ 상계 {o['selfeq']:.3f}/{g['selfeq']:.3f}", flush=True)
json.dump(out, open(os.path.join(os.path.dirname(rp), "force.json"), "w"), indent=1)
