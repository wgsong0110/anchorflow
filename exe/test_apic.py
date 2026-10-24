"""--apic / --p2g_stress 단위 확인 (사용자 10-24). GPU 하나(또는 CPU), 수 초.

  (a) 합성 강체 회전·단순 전단: 노드를 사상 f 로 옮긴 셀 사상의 야코비안 J (SX.g2p_jac, 실행이 F 갱신에 쓰는 것과 같은 식) 로
      C = (J - I)/h 가 강체 회전이면 (R - I)/h (각속도 행렬 Ω 와 O(h|Ω|²) 차), 단순 전단이면 전단률 행렬과 같다
  (c) APIC P2G: 아핀 속도장 v = a + B x, C_p = B 이면 노드 속도 = v(x_i) (활성 노드 전부, 이웃 일부 빠져도); 비 APIC P2G 오차 참고
  응력: τ = P(F)Fᵀ 가 균일 신장·단순 전단의 FCR 해석값 (SVD 극분해) 과 같고, 강체 회전이면 0;
        균일 응력·고른 입자 배치면 내부 노드 f_i ≈ 0, 경계 노드만 0 아님; p2g_stress_feats == 노드별 직접 합
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from anchorflow import simplex as SX                       # noqa: E402
from anchorflow import simplex_gnn as SG                   # noqa: E402
from anchorflow import deform_model as DM                  # noqa: E402

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
D = torch.float64
OK = [True]
I3 = torch.eye(3, device=dev, dtype=D)


def chk(name, ok, msg=""):
    print(f"[{'통과' if ok else '실패'}] {name} {msg}", flush=True)
    OK[0] &= bool(ok)


def hat(w):
    return torch.tensor([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]], device=dev, dtype=D)


N = 4000
x = torch.rand(N, 3, device=dev, dtype=D) * 0.3 + 0.5
lo, lat, nn = SX.grid_for_nodes(x, 0, h_fix=0.028284)
nnl = [int(nn[k]) for k in range(3)]
allid = torch.arange(nnl[0] * nnl[1] * nnl[2], device=dev)
allpos = SX.node_pos(lo, lat, nn, allid)
h = 1.0 / 30.0
print(f"격자 h={lat.s:.6f} 노드 {allid.numel()}, 입자 {N}, 스텝 h={h:.5f}")

# ---------------- (a) 강체 회전 / 단순 전단 ----------------
w = torch.tensor([0.7, -1.3, 2.1], device=dev, dtype=D)
Om = hat(w)
R = torch.linalg.matrix_exp(h * Om)
ctr = x.mean(0)
dp = (allpos - ctr) @ R.T + ctr - allpos                       # 노드 변위: 강체 회전 사상
_, G = SX.g2p_jac(x, lo, lat, nn, dp)
C = G / h                                                      # (J - I)/h
e_ex = float((C - (R - I3) / h).abs().max())
e_om = float((C - Om).abs().max()) / float(Om.abs().max())
e_sk = float((0.5 * (C + C.transpose(-1, -2))).abs().max())
chk("강체 회전: C = (R - I)/h", e_ex < 1e-9, f"최대 차 {e_ex:.2e}")
chk("강체 회전: C ≈ Ω (차 = O(h|Ω|))", e_om < 2 * h * float(w.norm()), f"상대 차 {e_om:.3e} (h|ω| = {h * float(w.norm()):.3e})")
print(f"    강체 회전 C 의 대칭 부분 최대 {e_sk:.3e} (= O(h|Ω|²), 2 차 항)")
gd = 1.7
S = torch.zeros(3, 3, device=dev, dtype=D); S[0, 1] = gd
dp = (allpos - ctr) @ (h * S).T
_, G = SX.g2p_jac(x, lo, lat, nn, dp)
e_sh = float((G / h - S).abs().max())
chk("단순 전단: C = 전단률 행렬 (C_xy = γ̇)", e_sh < 1e-9, f"최대 차 {e_sh:.2e}")

# ---------------- (c) APIC P2G 가 아핀 속도장 재현 ----------------
idx, lam, aux = SX.locate(x, lo, lat, nn)
rows, uniq = SX.active_nodes(idx)
Mn = uniq.numel()
npos = SX.node_pos(lo, lat, nn, uniq)
f27, w27, dw27 = SX.bspline27(x, lo, lat, nn)
pos = torch.searchsorted(uniq, f27).clamp(max=Mn - 1)
hit = uniq[pos] == f27
r27 = torch.where(hit, pos, torch.full_like(pos, Mn))
ww = torch.where(hit, w27, torch.zeros_like(w27))
print(f"    활성 노드 {Mn}, 27 이웃 중 활성 노드 비율 {100 * float(hit.double().mean()):.1f}%")
m = torch.rand(N, device=dev, dtype=D) + 0.5
F = I3 + 0.05 * torch.randn(N, 3, 3, device=dev, dtype=D)
av = torch.tensor([0.3, -0.2, 0.1], device=dev, dtype=D)
B = torch.randn(3, 3, device=dev, dtype=D)
v = av + x @ B.T
Cp = B.expand(N, 3, 3)
np1 = torch.cat([npos, npos[:1]], 0)
dx = np1[r27] - x.unsqueeze(1)
vs, hn = 2.5, float(lat.s)
out = SG.p2g_feats(v / vs, F, m, r27, ww, Mn + 1, C=Cp / vs, dx=dx, hn=hn)[:Mn]
chk("APIC 폭 22", out.shape == (Mn, SG.P2G_NCH + SG.P2G_APIC_NCH), str(tuple(out.shape)))
vtrue = (av + npos @ B.T) / vs
has = out[:, 0] > 0
e_ap = float((out[has, 1:4] - vtrue[has]).abs().max()) / float(vtrue.abs().max())
out0 = SG.p2g_feats(v / vs, F, m, r27, ww, Mn + 1)[:Mn]
e_pic = float((out0[has, 1:4] - vtrue[has]).abs().max()) / float(vtrue.abs().max())
chk("APIC P2G 노드 속도 = v(x_i) (아핀장, 질량 있는 활성 노드 전부)", e_ap < 1e-10, f"상대 최대 차 {e_ap:.2e} (비 APIC P2G {e_pic:.2e})")
chk("APIC C_i 칸 = B·h/vel_scale", float((out[has, 13:22] - (B * hn / vs).reshape(1, 9)).abs().max()) < 1e-10)
chk("비 APIC 칸 (질량·F) 불변", float((out[:, [0] + list(range(4, 13))] - out0[:, [0] + list(range(4, 13))]).abs().max()) < 1e-12)
# 노드별 직접 합 (임의 C)
Cr = torch.randn(N, 3, 3, device=dev, dtype=D)
out = SG.p2g_feats(v / vs, F, m, r27, ww, Mn + 1, C=Cr / vs, dx=dx, hn=hn)[:Mn]
err = 0.0
for jj in torch.randint(0, Mn, (20,), device=dev).tolist():
    sel = (r27 == jj)
    wj = (ww * sel).sum(1) * m
    mi = wj.sum()
    dxj = (npos[jj] - x)
    vi = (wj.unsqueeze(-1) * (v + torch.einsum("nab,nb->na", Cr, dxj))).sum(0) / mi / vs
    Ci = (wj.reshape(-1, 1, 1) * Cr).sum(0) / mi * hn / vs
    err = max(err, float((vi - out[jj, 1:4]).abs().max()), float((Ci.reshape(9) - out[jj, 13:22]).abs().max()))
chk("APIC 노드별 직접 합과 같음 (임의 C)", err < 1e-10, f"{err:.2e}")

# ---------------- 응력 ----------------
cfg = {"E": 2.0e5, "nu": 0.3, "material": "jelly", "density": 200.0}
mu, la = cfg["E"] / (2 * (1 + cfg["nu"])), cfg["E"] * cfg["nu"] / ((1 + cfg["nu"]) * (1 - 2 * cfg["nu"]))


def tau_ref(Fm):
    U, s, Vh = torch.linalg.svd(Fm)
    Rm = U @ Vh
    J = torch.linalg.det(Fm)
    return 2 * mu * (Fm - Rm) @ Fm.transpose(-1, -2) + (la * J * (J - 1)).reshape(-1, 1, 1) * I3


Fs = torch.stack([torch.diag(torch.tensor([1.02, 0.99, 1.005], device=dev, dtype=D)),
                  I3 + torch.tensor([[0, 0.03, 0], [0, 0, 0], [0, 0, 0]], device=dev, dtype=D),
                  torch.linalg.matrix_exp(hat(torch.tensor([0.4, 0.2, -0.5], device=dev, dtype=D))) @ torch.diag(torch.tensor([1.03, 0.98, 1.0], device=dev, dtype=D))])
tq = DM.kirchhoff_tau(Fs, cfg)
tr = tau_ref(Fs)
for k, nm in enumerate(["균일 신장", "단순 전단", "회전 x 신장"]):
    e = float((tq[k] - tr[k]).abs().max()) / float(tr[k].abs().max())
    chk(f"τ 해석값 일치 ({nm})", e < 1e-8, f"상대 차 {e:.2e}, |τ|/E {float(tr[k].abs().max()) / cfg['E']:.3e}")
chk("τ 대칭", float((tq - tq.transpose(-1, -2)).abs().max()) < 1e-6 * float(tq.abs().max()))
Rr = torch.linalg.matrix_exp(hat(torch.tensor([0.9, -0.4, 1.2], device=dev, dtype=D))).unsqueeze(0)
chk("강체 회전 τ = 0", float(DM.kirchhoff_tau(Rr, cfg).abs().max()) < 1e-6 * cfg["E"], f"|τ| 최대 {float(DM.kirchhoff_tau(Rr, cfg).abs().max()):.2e}")

# 균일 응력: 격자 좌표에서 간격 1/2 로 고르게 (셀마다 8 입자) 놓은 블록
nb = 8
g1 = (torch.arange(2 * nb, device=dev, dtype=D) + 0.5) / 2.0 + 4.0      # 격자 좌표 4.25 .. 4+nb-0.25
Y = torch.stack(torch.meshgrid(g1, g1, g1, indexing="ij"), -1).reshape(-1, 3)
lo2 = torch.zeros(3, device=dev, dtype=D)                               # 블록 전용 격자 (원점 0, 같은 능면체 격자)
nn2 = torch.tensor([nb + 10] * 3, device=dev)
xb = Y @ lat.A.T + lo2
Nb = xb.shape[0]
f27, w27, dw27 = SX.bspline27(xb, lo2, lat, nn2)
uq = torch.unique(f27.reshape(-1))                                      # 노드 = 27 이웃 전부 (응력 검사는 빠진 이웃 없이)
Mb = uq.numel(); nposb = SX.node_pos(lo2, lat, nn2, uq)
r27 = torch.searchsorted(uq, f27)
gw = dw27.to(D) @ lat.Ai
Fu = torch.diag(torch.tensor([1.02, 0.99, 1.005], device=dev, dtype=D)).expand(Nb, 3, 3)
tb = DM.kirchhoff_tau(Fu, cfg)
mb = torch.full((Nb,), 1e-3, device=dev, dtype=D)
V0 = mb / cfg["density"]
sf = SG.p2g_stress_feats(tb, V0, mb, r27, w27.to(D), gw, Mb, cfg["E"], hn)
chk("응력 폭 9", sf.shape == (Mb, SG.P2G_STRESS_NCH), str(tuple(sf.shape)))
yn = (nposb - lo2) @ lat.Ai.T
inner = ((yn >= 4.0 + 2) & (yn <= 4.0 + nb - 2)).all(-1)
fmag = sf[:, :3].norm(dim=-1)
print(f"    균일 응력 블록: 입자 {Nb}, 노드 {Mb}, 내부 노드 {int(inner.sum())}")
chk("균일 응력: 내부 노드 f_i ≈ 0", float(fmag[inner].max()) < 1e-10 * float(fmag.max()),
    f"내부 최대 {float(fmag[inner].max()):.2e}, 경계 최대 {float(fmag[~inner].max()):.2e}")
chk("균일 응력: 경계 노드는 0 아님", float(fmag[~inner].max()) > 0)
chk("균일 응력: 내부력 합 0 (Σ_i ∇w_ip = 0)", float(sf[:, :3].sum(0).abs().max()) < 1e-10 * float(fmag.max()))
tsym = torch.stack([tb[0, 0, 0], tb[0, 1, 1], tb[0, 2, 2], tb[0, 0, 1], tb[0, 0, 2], tb[0, 1, 2]]) / cfg["E"]
chk("균일 응력: 평균 τ_i/E = τ/E (모든 노드)", float((sf[:, 3:] - tsym).abs().max()) < 1e-12)
# 직접 합 (임의 응력)
tr_ = torch.randn(Nb, 3, 3, device=dev, dtype=D); tr_ = tr_ + tr_.transpose(-1, -2)
sf = SG.p2g_stress_feats(tr_, V0, mb, r27, w27.to(D), gw, Mb, cfg["E"], hn)
err = 0.0
for jj in torch.randint(0, Mb, (20,), device=dev).tolist():
    sel = (r27 == jj).to(D)
    fj = -(V0.reshape(-1, 1) * torch.einsum("nab,nkb,nk->na", tr_, gw, sel)).sum(0) / (cfg["E"] * hn ** 2)
    err = max(err, float((fj - sf[jj, :3]).abs().max()))
chk("내부력 노드별 직접 합과 같음", err < 1e-10, f"{err:.2e}")
print("TEST_APIC OK" if OK[0] else "TEST_APIC FAIL", flush=True)
