"""셀 안 버블 회전(lib/anchorflow/localrot.py, --local_rot bubble) 단위 시험 (CPU).

  (a) θ = 0                      -> 선형 경로(g2p_pre·g2p_jac_pre)와 비트 단위로 같다 (float32·float64)
  (b) 경계 고정                   -> 셀 면 위 점·꼭짓점에서 y = x (θ 큼)
  (c) 이웃 셀 연속                -> 공유 면 위 점을 두 셀로 각각 계산한 x' 가 같다 (U ≠ 0, θ ≠ 0)
  (d) 야코비안                    -> Du = J_Φ J_loc − I 를 중심 유한차분과 대조 (상한 안·포화 둘 다)
  (e) 강제 큰 θ (상한의 10~1000 배) -> det J_loc ≥ (1−κ)³ > 0, 셀 밖 입자 0 (입자 + 셀 안 무작위 점 + 면 근처 점);
                                     상한을 끄면(cap=False) 같은 θ 에서 뒤집힘·이탈이 생기는지 (상한이 실제로 막고 있는지)
  (f) 상한 크기: θmax 분포와 포화 θ 에서 실제로 낼 수 있는 국소 변형 ‖J_loc − I‖ 크기 (기록만)
  python exe/test_local_rot.py [--pt 입자.pt]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import torch                                                     # noqa: E402
from anchorflow import simplex as sx                             # noqa: E402
from anchorflow import localrot as LR                            # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pt", default="")
ap.add_argument("--n", type=int, default=20000)
ap.add_argument("--h", type=float, default=0.056569)
ap.add_argument("--kappa", type=float, default=LR.KAPPA)
a = ap.parse_args()
torch.manual_seed(0)
dt = torch.float64
h, kap = a.h, a.kappa
if a.pt:
    d = torch.load(a.pt, map_location="cpu", weights_only=False, mmap=True)
    P = d["x"][0].to(dt)
    P = P[torch.randperm(P.shape[0])[:a.n]]
else:
    P = torch.rand(a.n, 3, dtype=dt) * (6 * h) + 0.5
lo, lat, nn = sx.grid_for_nodes(P, 1.0, h_fix=h)


def bind(Q):
    idx, lam, aux = sx.locate(Q, lo, lat, nn)
    return idx, lam, aux


idx, lam_l, aux = bind(P)
rows, uniq = sx.active_nodes(idx)
Xn = sx.node_pos(lo, lat, nn, uniq).to(dt)
M = Xn.shape[0]
I3 = torch.eye(3, dtype=dt)
ok = True


def chk(name, cond):
    global ok
    ok &= bool(cond)
    print(f"  -> {name}: {'통과' if cond else '실패'}", flush=True)


def full(Q, r, U, th, cap=True, stats=False):
    """임의 점 Q 를 셀 r 로 (λ 는 그 셀 꼭짓점으로 직접) -> x' − x, J_Φ J_loc − I"""
    Xv, dlam, c, r_c, g_c, S_c = LR.cell_geom(Xn, r)
    Minv = dlam[:, 1:]
    l3 = (Minv @ (Q - Xv[:, 0])[..., None]).squeeze(-1)
    lam = torch.cat([1 - l3.sum(1, keepdim=True), l3], 1)
    Ur = U[r]
    dul = (lam[..., None] * Ur).sum(1)
    Dul = (Ur[..., :, None] * dlam[..., None, :]).sum(1)
    return LR.local_rot(Q, Xn, r, lam, th, dul, Dul, kappa=kap, with_stats=stats, cap=cap)


tmax = LR.theta_max(*LR.cell_geom(Xn, rows)[3:], kappa=kap)
print(f"[격자] 노드 {M}, 입자 {P.shape[0]}, h {h}, κ {kap}; θmax (rad) 최소 {float(tmax.min()):.4f} 중앙 {float(tmax.median()):.4f} "
      f"최대 {float(tmax.max()):.4f}")

# (a) θ = 0: 선형 경로와 비트 단위로 같다 (deform_model.advance 의 선형 분기와 같은 함수·같은 인자)
for dd in (torch.float32, torch.float64):
    Pd, Xd = P.to(dd), Xn.to(dd)
    Ud = (0.1 * h * torch.randn(M, 3, dtype=dt)).to(dd)
    lam_d = lam_l.to(dd)
    dul = sx.g2p_pre(rows, lam_d, Ud)
    _, Dul = sx.g2p_jac_pre(rows, lam_d, aux, lat, Ud)
    du, Du = LR.local_rot(Pd, Xd, rows, lam_d, torch.zeros(M, 3, dtype=dd), dul, Dul, kappa=kap)
    same = torch.equal(du, dul) and torch.equal(Du, Dul)
    print(f"(a) {dd}: θ=0 -> du 최대차 {float((du - dul).abs().max()):.1e}, Du 최대차 {float((Du - Dul).abs().max()):.1e} "
          f"(torch.equal {same})")
    chk(f"(a) θ=0 선형과 비트 단위로 같다 {dd}", same)
# 선형 경로의 J_Φ − I (g2p_jac_pre) 가 Σ U ⊗ ∇λ (localrot 가 쓰는 셀 기하) 와 같은지
_, dlam_r, *_ = LR.cell_geom(Xn, rows)
U = 0.1 * h * torch.randn(M, 3, dtype=dt)
_, Dul = sx.g2p_jac_pre(rows, lam_l.to(dt), aux, lat, U)
Dmine = (U[rows][..., :, None] * dlam_r[..., None, :]).sum(1)
print(f"(a) g2p_jac_pre 와 Σ U⊗∇λ 최대차 {float((Dul - Dmine).abs().max()):.2e}")
chk("(a) J_Φ 같은 정의", float((Dul - Dmine).abs().max()) < 1e-10)

# (b) 경계 고정: 면 위 점 (λ_k = 0), 꼭짓점
TH = 100.0 * float(tmax.median()) * torch.randn(M, 3, dtype=dt)     # 상한 포화
n0 = min(P.shape[0], 20000)
k = torch.randint(0, 4, (n0,))
mu = torch.rand(n0, 4, dtype=dt)
mu[torch.arange(n0), k] = 0.0
mu = mu / mu.sum(1, keepdim=True)
r_in = rows[:n0]
xf = (mu[..., None] * Xn[r_in]).sum(1)
du0, _ = full(xf, r_in, torch.zeros(M, 3, dtype=dt), TH)
print(f"(b) 면 위 점 {n0} 개, U=0, θ 포화: |y − x|/h 최대 {float(du0.norm(dim=1).max()) / h:.2e}")
chk("(b) 면 위 점 고정", float(du0.norm(dim=1).max()) < 1e-12 * h)
xv = Xn[r_in[:, 0]]
duv, _ = full(xv, r_in, torch.zeros(M, 3, dtype=dt), TH)
print(f"(b) 꼭짓점: |y − x|/h 최대 {float(duv.norm(dim=1).max()) / h:.2e}")
chk("(b) 꼭짓점 고정", float(duv.norm(dim=1).max()) < 1e-12 * h)
duP, _ = full(P, rows, torch.zeros(M, 3, dtype=dt), TH)
print(f"(b) 참고: 같은 θ 에서 셀 안 입자 |y − x|/h 평균 {float(duP.norm(dim=1).mean()) / h:.3e} 최대 {float(duP.norm(dim=1).max()) / h:.3e}")
chk("(b) 셀 안 입자는 움직인다", float(duP.norm(dim=1).mean()) > 1e-6 * h)

# (c) 이웃 셀 연속: 면 위 점을 이 셀·면 건너편 셀로 각각
lut = torch.full((int(nn.prod()),), -1, dtype=torch.long)
lut[uniq] = torch.arange(M)
cen = Xn[r_in].mean(1)
nb, _, _ = bind(xf - 1e-10 * h * torch.nn.functional.normalize(cen - xf, dim=1))
r_nb = lut[nb]
good = (r_nb >= 0).all(1) & (torch.sort(r_nb, 1).values != torch.sort(r_in, 1).values).any(1)
for nm, sc in (("상한 안", 0.5), ("포화", 100.0)):
    THc = sc * float(tmax.median()) * torch.randn(M, 3, dtype=dt)
    da, Da = full(xf[good], r_in[good], U, THc)
    db, Db = full(xf[good], r_nb[good], U, THc)
    jmp = (da - db).norm(dim=1)
    print(f"(c) {nm}: 면 위 점 쌍 {int(good.sum())}, |x'(이 셀) − x'(이웃 셀)|/h 최대 {float(jmp.max()) / h:.2e} "
          f"(|du|/h 평균 {float(da.norm(dim=1).mean()) / h:.2e})")
    chk(f"(c) 이웃 셀 연속 {nm}", int(good.sum()) > 100 and float(jmp.max()) < 1e-10 * h)

# (d) 야코비안 유한차분 (셀 고정)
sel = torch.arange(3000)
x0, r0 = P[sel].clone(), rows[sel]
for nm, sc in (("상한 안 (0.3 θmax)", 0.3), ("포화 (30 θmax)", 30.0)):
    THd = sc * float(tmax.median()) * torch.randn(M, 3, dtype=dt)
    _, Du = full(x0, r0, U, THd)
    fd = torch.zeros_like(Du)
    step = 1e-7 * h
    for kk in range(3):
        e = torch.zeros(3, dtype=dt)
        e[kk] = step
        up, _ = full(x0 + e, r0, U, THd)
        um, _ = full(x0 - e, r0, U, THd)
        fd[:, :, kk] = (up - um) / (2 * step)
    err = (Du - fd).abs().max()
    print(f"(d) {nm}: Du vs 중심 유한차분 최대차 {float(err):.2e} (‖Du‖ 최대 {float(Du.abs().max()):.2e})")
    chk(f"(d) 야코비안 일치 {nm}", float(err) < 1e-6 * max(1.0, float(Du.abs().max())))
# (d') 로드리게스 미분 ∂(R v)/∂φ 를 큰 각(t ≥ 0.5, 급수 밖)에서도: 상한 끄고 같은 대조
THb = 200.0 * float(tmax.median()) * torch.randn(M, 3, dtype=dt)
_, Du = full(x0, r0, U, THb, cap=False)
fd = torch.zeros_like(Du)
for kk in range(3):
    e = torch.zeros(3, dtype=dt)
    e[kk] = 1e-7 * h
    up, _ = full(x0 + e, r0, U, THb, cap=False)
    um, _ = full(x0 - e, r0, U, THb, cap=False)
    fd[:, :, kk] = (up - um) / (2e-7 * h)
_, _, stb = full(x0, r0, U, THb, cap=False, stats=True)
err = (Du - fd).abs().max()
print(f"(d') 상한 없음, 큰 각 (|φ| 최대 {float(stb[4]):.2f} rad): 최대차 {float(err):.2e} (‖Du‖ 최대 {float(Du.abs().max()):.2e})")
chk("(d') 큰 각 야코비안 일치", float(err) < 1e-6 * max(1.0, float(Du.abs().max())))

# (e) 강제 큰 θ: 입자 + 셀 안 무작위 점 (면 근처 포함)
nR = 200000
rr = rows[torch.randint(0, rows.shape[0], (nR,))]
g = -torch.log(torch.rand(nR, 4, dtype=dt))
mu = g / g.sum(1, keepdim=True)
near = torch.rand(nR) < 0.3                                       # 30 % 는 면 근처 (한 좌표 1e-4 ~ 1e-2)
kk = torch.randint(0, 4, (nR,))
mu[near, kk[near]] = 10 ** (-4 + 2 * torch.rand(int(near.sum()), dtype=dt))
mu = mu / mu.sum(1, keepdim=True)
xr = (mu[..., None] * Xn[rr]).sum(1)
lo3 = (1 - kap) ** 3
for sc in (10.0, 100.0, 1000.0):
    THe = sc * float(tmax.median()) * torch.randn(M, 3, dtype=dt)
    for nm, Q, r in (("입자", P, rows), ("셀 안 무작위 점", xr, rr)):
        _, _, st = full(Q, r, torch.zeros(M, 3, dtype=dt), THe, stats=True)
        _, _, sn = full(Q, r, torch.zeros(M, 3, dtype=dt), THe, cap=False, stats=True)
        print(f"(e) θ ~ {sc:g}·θmax 중앙, {nm} {Q.shape[0]}: 상한 -> det J_loc 최소 {float(st[0]):.4f} (하한 (1−κ)³ {lo3:.4f}), 셀 밖 {int(st[1])}, "
              f"λ(y) 최소 {float(st[2]):.2e}, |φ| 최대 {float(st[4]):.4f}, ‖K‖_F 최대 {float(st[7]):.3f} | 상한 끔 -> det 최소 {float(sn[0]):.3f}, "
              f"셀 밖 {int(sn[1])}")
        chk(f"(e) 상한: det > (1−κ)³, 셀 밖 0 ({nm}, {sc:g}x)", float(st[0]) >= lo3 - 1e-9 and int(st[1]) == 0)
        if sc >= 100 and nm == "셀 안 무작위 점":
            chk(f"(e) 상한을 끄면 실제로 깨진다 ({sc:g}x, 상한이 일을 하고 있다)", float(sn[0]) <= 0 or int(sn[1]) > 0)

# (f) 상한 크기 (기록): 포화 θ 에서 낼 수 있는 국소 변형
THf = 1000.0 * float(tmax.median()) * torch.randn(M, 3, dtype=dt)
_, Du = full(P, rows, torch.zeros(M, 3, dtype=dt), THf)
gJ = Du.flatten(1).norm(dim=1)
sv = torch.linalg.svdvals(I3 + Du)
print(f"(f) 포화 θ, U=0: ‖J_loc − I‖_F 중앙 {float(gJ.median()):.3e} 99% {float(gJ.quantile(0.99)):.3e} 최대 {float(gJ.max()):.3e}; "
      f"특이값 [{float(sv.min()):.4f}, {float(sv.max()):.4f}]; |y−x|/h 최대 {float(full(P, rows, torch.zeros(M, 3, dtype=dt), THf)[0].norm(dim=1).max()) / h:.3e}")
print("[전체]", "통과" if ok else "실패", flush=True)
