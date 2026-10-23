"""고친 로컬 변형(lib/anchorflow/skin.py gregory_skin) 단위 시험 (CPU, float64).

  (a) U=0, ρ=ρ_ref            -> 변위 0, ∇u 0 (F = I)
  (b) U=0, ρ≠ρ_ref            -> 셀 안 입자가 움직인다; 꼭짓점 고정; 셀 경계를 가로지를 때 연속 (양쪽 셀로 따로 계산)
  (c) ρ=ρ_ref, U 임의         -> 선형 barycentric 과 같다; 옛 gregory(v0) 와의 차이 크기
  (d) ∇u 를 중심 유한차분과 대조 (셀 고정)
  (e) det F 범위 (U=0 · U≠0, 반경 표준편차 여러 개)
  python exe/test_greg_local.py [--pt 입자.pt]
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import torch                                                     # noqa: E402
from anchorflow import simplex as sx                             # noqa: E402
from anchorflow.skin import gregory_skin, gregory_skin_v0        # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pt", default="")
ap.add_argument("--n", type=int, default=20000)
ap.add_argument("--h", type=float, default=0.056569)
a = ap.parse_args()
torch.manual_seed(0)
dt = torch.float64
h = a.h
if a.pt:
    d = torch.load(a.pt, map_location="cpu", weights_only=False, mmap=True)
    P = d["x"][0].to(dt)
    P = P[torch.randperm(P.shape[0])[:a.n]]
else:
    P = torch.rand(a.n, 3, dtype=dt) * (6 * h) + 0.5
lo, lat, nn = sx.grid_for_nodes(P, 1.0, h_fix=h)


def bind(Q):
    idx, lam, _ = sx.locate(Q, lo, lat, nn)
    return idx


idx = bind(P)
rows, uniq = sx.active_nodes(idx)
Xn = sx.node_pos(lo, lat, nn, uniq).to(dt)
M = Xn.shape[0]
I3 = torch.eye(3, dtype=dt)
ok = True


def chk(name, cond):
    global ok
    ok &= bool(cond)
    print(f"  -> {name}: {'통과' if cond else '실패'}", flush=True)


print(f"[격자] 노드 {M}, 입자 {P.shape[0]}, h {h}")
# (a)
du, Du = gregory_skin(P, Xn, rows, torch.zeros(M, 3, dtype=dt), torch.zeros(M, dtype=dt), h)
print(f"(a) U=0, ρ=ρ_ref: 변위 최대 {float(du.abs().max()):.2e}, ∇u 최대 {float(Du.abs().max()):.2e}")
chk("(a) 항등", du.abs().max() < 1e-12 and Du.abs().max() < 1e-12)

# (b)
for s in (0.5, 1.0, 3.0):
    R = s * torch.randn(M, dtype=dt)
    du, Du, lam, lam2 = gregory_skin(P, Xn, rows, torch.zeros(M, 3, dtype=dt), R, h, return_parts=True)
    m = du.norm(dim=1) / h
    print(f"(b) U=0, ρ_raw~N(0,{s}²): |du|/h 평균 {float(m.mean()):.4f} 중앙 {float(m.median()):.4f} 최대 {float(m.max()):.4f}; "
          f"λ' 최소 {float(lam2.min()):.4f} (음수 비율 {float((lam2 < 0).double().mean()):.4f})")
chk("(b) 셀 안 이동 ≠ 0", float(m.mean()) > 1e-4)
# 꼭짓점: 노드 위치에서 아주 조금 안쪽
Q = Xn[rows[:2000, 0]] + 1e-9 * (P[:2000] - Xn[rows[:2000, 0]])
duv, _ = gregory_skin(Q, Xn, rows[:2000], torch.zeros(M, 3, dtype=dt), R, h)
print(f"(b) 꼭짓점 근처(1e-9) 변위 최대 {float(duv.abs().max()):.2e}")
chk("(b) 꼭짓점 고정", duv.abs().max() < 1e-8)
# 경계 연속: 셀 면 위의 점 x_f (맞은편 꼭짓점 k 의 λ_k = 0) 을 이 셀과 면 건너편 이웃 셀로 **각각** 계산해 비교
lut = torch.full((int(nn.prod()),), -1, dtype=torch.long)
lut[uniq] = torch.arange(M)
U = 0.1 * h * torch.randn(M, 3, dtype=dt)
n0 = min(P.shape[0], 20000)
k = torch.randint(0, 4, (n0,))
mu = torch.rand(n0, 4, dtype=dt)
mu[torch.arange(n0), k] = 0.0
mu = mu / mu.sum(1, keepdim=True)
r_in = rows[:n0]
xf = (mu[..., None] * Xn[r_in]).sum(1)                         # 면 위의 점
cen = Xn[r_in].mean(1)
nb = bind(xf - 1e-10 * h * torch.nn.functional.normalize(cen - xf, dim=1))   # 면 건너편 셀
r_nb = lut[nb]
good = (r_nb >= 0).all(1) & (torch.sort(r_nb, 1).values != torch.sort(r_in, 1).values).any(1)
for nm, UU in (("U=0", torch.zeros(M, 3, dtype=dt)), ("U≠0", U)):
    da, _ = gregory_skin(xf[good], Xn, r_in[good], UU, R, h)
    db, _ = gregory_skin(xf[good], Xn, r_nb[good], UU, R, h)
    jmp = (da - db).norm(dim=1)
    print(f"(b) 셀 면 위 점, 두 셀로 각각: 쌍 {int(good.sum())}, |du(이 셀) − du(이웃 셀)| 최대 {float(jmp.max()):.2e} "
          f"(|du| 평균 {float(da.norm(dim=1).mean()):.2e})")
    chk(f"(b) 면에서 연속 {nm}", int(good.sum()) > 100 and float(jmp.max()) < 1e-9 * h)

# (c)
U = 0.1 * h * torch.randn(M, 3, dtype=dt)
du, Du = gregory_skin(P, Xn, rows, U, torch.zeros(M, dtype=dt), h)
idx2, lamb, _ = sx.locate(P, lo, lat, nn)
dub = (lamb.to(dt)[..., None] * U[rows]).sum(1)
dv0, Dv0 = gregory_skin_v0(P, Xn, rows, U, torch.zeros(M, dtype=dt), h)
print(f"(c) ρ=ρ_ref: barycentric 과 최대차 {float((du - dub).abs().max()):.2e}; 옛 gregory(v0, ρ_raw=0) 와 최대차 "
      f"{float((du - dv0).abs().max()):.3e} (|U| 평균 {float(U.norm(dim=1).mean()):.3e}), ∇u 상대차 "
      f"{float((Du - Dv0).norm(dim=(1, 2)).mean() / Du.norm(dim=(1, 2)).mean()):.3f}")
chk("(c) barycentric 과 같다", (du - dub).abs().max() < 1e-12)

# (d)
R = 1.0 * torch.randn(M, dtype=dt)
sel = torch.arange(3000)
x0 = P[sel].clone()
r0 = rows[sel]
_, Du = gregory_skin(x0, Xn, r0, U, R, h)
fd = torch.zeros_like(Du)
step = 1e-7
for k in range(3):
    dx = torch.zeros(3, dtype=dt)
    dx[k] = step
    up, _ = gregory_skin(x0 + dx, Xn, r0, U, R, h)
    um, _ = gregory_skin(x0 - dx, Xn, r0, U, R, h)
    fd[:, :, k] = (up - um) / (2 * step)
err = (Du - fd).abs().max()
print(f"(d) ∇u vs 유한차분: 최대차 {float(err):.2e} (∇u 크기 최대 {float(Du.abs().max()):.2e})")
chk("(d) 야코비안 일치", float(err) < 1e-5 * max(1.0, float(Du.abs().max())))

# (e)
for su, s in ((0.0, 0.5), (0.0, 1.0), (0.0, 3.0), (0.0, 10.0), (0.05, 1.0), (0.2, 1.0), (0.2, 3.0)):
    UU = su * h * torch.randn(M, 3, dtype=dt)
    RR = s * torch.randn(M, dtype=dt)
    _, Du = gregory_skin(P, Xn, rows, UU, RR, h)
    dF = torch.linalg.det(I3 + Du)
    print(f"(e) |U|~{su}h, ρ_raw~N(0,{s}²): det F 최소 {float(dF.min()):.4f} 1% {float(dF.quantile(0.01)):.4f} "
          f"중앙 {float(dF.median()):.4f} 최대 {float(dF.max()):.4f}, det≤0 비율 {float((dF <= 0).double().mean()):.5f}")
# (f) 방사형 함수 ψ 의 꺾임(r = ρ, 안쪽 기울기 −2a/ρ, 바깥 −(1−a)/(h/2)): 반경을 아주 조금 바꿔도
#     ρ_ref 와 ρ 사이 얇은 껍질의 입자는 ∇λ' 가 O(1) 만큼 바뀐다 (변위는 O(Δρ)). 크기와 비율을 기록만 한다.
for s in (1e-4, 1e-3, 1e-2):
    RR = s * torch.randn(M, dtype=dt)
    du, Du = gregory_skin(P, Xn, rows, torch.zeros(M, 3, dtype=dt), RR, h)
    g = Du.flatten(1).norm(dim=1)
    print(f"(f) U=0, ρ_raw~N(0,{s:g}²): |du|/h 최대 {float(du.norm(dim=1).max()) / h:.2e}, ‖∇u‖ 최대 {float(g.max()):.3f}, "
          f"‖∇u‖>0.1 입자 비율 {float((g > 0.1).double().mean()):.2e}, 중앙 {float(g.median()):.2e}")
print("[전체]", "통과" if ok else "실패", flush=True)
