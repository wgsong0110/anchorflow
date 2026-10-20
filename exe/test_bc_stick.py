"""--bc_stick 단위 시험 (CPU, 빌드 없음): deform_model.bc_stick_map (+ 뒤따르는 bc_softplus_map, advance 순서 그대로)

  1) 바닥 먼 입자(s̃ > 20ε) 위치·야코비안 정확히 항등
  2) 접촉 입자(s̃ ≤ -0.1) 접선 변위 ≈ 0 (w 배), 법선 성분 그대로
  3) 합성 매핑 야코비안 == autograd 미분 (x_n 고정, ∂x/∂x̃) -- w 의 s̃ 의존 항 포함
     --bc_stick_xn: 반환 F == 선형장 x̃(x_n) 을 거친 x_n 전미분 (autograd)
  4) det > 0, det(합성) = w³ det(옛)
  5) 역전파·jvp 기울기 유한
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import torch

from anchorflow.deform_model import bc_softplus_map, bc_stick_map

torch.manual_seed(0)
dt = torch.float64
EPS = 0.01
nr = torch.tensor([0.1, 0.2, 1.0], dtype=dt); nr = nr / nr.norm()
pt = [0.3, -0.2, 0.5]
cfg = {"boundary_conditions": [{"type": "surface_collider", "point": pt, "normal": nr.tolist(), "surface": "sticky"},
                               {"type": "bounding_box"}]}
P = torch.tensor(pt, dtype=dt)
I3 = torch.eye(3, dtype=dt)
N = 4000
s0 = torch.cat([torch.linspace(-0.5, -0.1, 1000), torch.linspace(-0.05, 0.05, 1000),
                torch.linspace(0.21, 2.0, 1000), torch.rand(1000) * 0.3 - 0.1]).to(dt)
tang = torch.randn(N, 3, dtype=dt); tang = tang - (tang @ nr).unsqueeze(-1) * nr
q = P + tang + s0.unsqueeze(-1) * nr                      # x̃ (스텝 끝, g2p 결과)
x0 = q - 0.05 * torch.randn(N, 3, dtype=dt)               # x_n (스텝 시작)
Jm = 0.1 * torch.randn(N, 3, 3, dtype=dt)
ok = True


def chk(name, cond, info=""):
    global ok
    ok = ok and bool(cond)
    print(f"[{'통과' if cond else '실패'}] {name} {info}", flush=True)


def full(x0_, q_, Jm_, xn=False):
    q1, J1 = bc_stick_map(x0_, q_, Jm_, cfg, EPS, xn)
    return bc_softplus_map(q1, J1, cfg, EPS)


q1, J1 = bc_stick_map(x0, q, Jm, cfg, EPS)
q2, J2 = full(x0, q, Jm)
far = s0 > 20 * EPS
chk("1) 먼 입자 위치 항등", bool((q2[far] == q[far]).all()), f"{int(far.sum())} 개")
chk("   먼 입자 야코비안 항등", float((J2[far] - Jm[far]).abs().max()) < 1e-12, f"{float((J2[far] - Jm[far]).abs().max()):.1e}")
d0 = q - x0; dt0 = d0 - (d0 @ nr).unsqueeze(-1) * nr
d1 = q2 - x0; dt1 = d1 - (d1 @ nr).unsqueeze(-1) * nr
con = s0 <= -0.1
rt = (dt1[con].norm(dim=-1) / dt0[con].norm(dim=-1).clamp_min(1e-30))
chk("2) 접촉 입자 접선 변위 ≈ 0", float(rt.max()) < 1e-4, f"|Δt'|/|Δt| 최대 {float(rt.max()):.1e} (s̃ 최대 {float(s0[con].max()):.2f})")
dn = ((q1 - q) @ nr).abs().max()
chk("   접선 축소가 법선 좌표를 안 바꿈", float(dn) < 1e-12, f"{float(dn):.1e}")
s2 = ((q2 - P) * nr).sum(-1)
chk("   softplus 뒤 s' >= -반올림", bool((s2 > -1e-13).all()), f"최소 {float(s2.min()):.3e}")
# 3) autograd
idx = [0, 500, 999, 1200, 1500, 1800, 2500, 3200, 3600, 3900]
e1, e2 = [], []
Z = torch.zeros(1, 3, 3, dtype=dt)
for i in idx:
    f = lambda y: full(x0[i:i + 1], y.unsqueeze(0), Z)[0][0]
    Ja = torch.autograd.functional.jacobian(f, q[i])
    e1.append(float((Ja - (full(x0[i:i + 1], q[i:i + 1], Z)[1][0] + I3)).abs().max()))
    # x_n 전미분: x̃(x_n) = x_n + A (x_n - c) + b 선형장 (grad u = A) -- 반환 F 와 비교
    A = Jm[i]; c = x0[i] + 0.01; b = q[i] - x0[i] - A @ (x0[i] - c)
    g = lambda xn: full(xn.unsqueeze(0), (xn + A @ (xn - c) + b).unsqueeze(0), A.unsqueeze(0), True)[0][0]
    Jx = torch.autograd.functional.jacobian(g, x0[i])
    e2.append(float((Jx - (full(x0[i:i + 1], q[i:i + 1], A.unsqueeze(0), True)[1][0] + I3)).abs().max()))
chk("3) 합성 야코비안 ∂x/∂x̃ == autograd", max(e1) < 1e-9, f"최대 차 {max(e1):.1e}")
chk("   --bc_stick_xn 반환 F == x_n 전미분 autograd", max(e2) < 1e-9, f"최대 차 {max(e2):.1e}")
# 4) det
sg = torch.where(s0 / EPS > 20.0, torch.ones_like(s0), torch.sigmoid(s0 / EPS))
dold = torch.linalg.det(I3 + Jm); dnew = torch.linalg.det(I3 + J2)
chk("4) det > 0 (기본)", bool((dnew * dold > 0).all()) and float(((dnew - sg ** 3 * dold) / dold.abs()).abs().max()) < 1e-8,
    f"w 최소 {float(sg.min()):.2e}, det새/det옛 = w³ 최대 상대차 {float(((dnew - sg ** 3 * dold) / dold.abs()).abs().max()):.1e}")
_, J2x = full(x0, q, Jm, True)
dx_ = torch.linalg.det(I3 + J2x)
chk("   det > 0 (--bc_stick_xn, |Jm|=0.1 에서)", bool((dx_ * dold > 0).all()), f"최소 det {float(dx_.min()):.2e}")
# 강체 평행이동(grad u = 0) 접촉 입자: 기본 det = w³, xn 전미분 det = w
rig_def = torch.linalg.det(I3 + full(x0[con], q[con], torch.zeros(int(con.sum()), 3, 3, dtype=dt))[1])
rig_xn = torch.linalg.det(I3 + full(x0[con], q[con], torch.zeros(int(con.sum()), 3, 3, dtype=dt), True)[1])
print(f"[참고] 평행이동 접촉 입자 (s̃<=-0.1) 한 스텝 det F: 기본 최대 {float(rig_def.max()):.2e} (= w³), "
      f"--bc_stick_xn 최대 {float(rig_xn.max()):.2e} (= w)", flush=True)
s_mid = torch.tensor([0.0, 0.00693, 0.02], dtype=dt)
for sv in s_mid.tolist():
    w = float(torch.sigmoid(torch.tensor(sv / EPS)))
    print(f"[참고] s̃={sv:.4f}: w={w:.3f}  기본 접선 F 배율 w={w:.3f}/스텝 (det w³={w ** 3:.3f}),  xn 전미분 접선 배율 1", flush=True)
# 5) 역전파 / jvp
qq = q.clone().requires_grad_(True); JJ = Jm.clone().requires_grad_(True); xx = x0.clone().requires_grad_(True)
q3, J3 = full(xx, qq, JJ)
L = (q3 ** 2).sum() + (torch.linalg.det(I3 + J3) ** 2).sum()
L.backward()
chk("5) 역전파 기울기 유한·0 아님", all(bool(torch.isfinite(t.grad).all()) and float(t.grad.abs().sum()) > 0 for t in (qq, JJ, xx)))
qf, Jf32, xf = q.float(), Jm.float(), x0.float()
tau0 = torch.tensor(1.0)
fn = lambda t: full(xf, qf + t * nr.float() * (-0.3) + t * tang.float() * 0.1, Jf32)
(o, oj), (tq_, tj_) = torch.func.jvp(fn, (tau0,), (torch.ones_like(tau0),))
chk("   jvp (float32) 유한", bool(torch.isfinite(tq_).all() and torch.isfinite(tj_).all() and torch.isfinite(o).all()))
print("BC_STICK_TEST_" + ("OK" if ok else "FAIL"), flush=True)
sys.exit(0 if ok else 1)
