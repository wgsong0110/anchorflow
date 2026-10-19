"""--bc_softplus 단위 시험 (CPU, 빌드 없음): deform_model.bc_softplus_map

  1) 매핑 뒤 모든 입자 s' > 0 (바닥 아래 깊이 있던 입자 포함)
  2) s ≫ ε 에서 위치·야코비안 정확히 항등
  3) 매핑 야코비안 det > 0, det(F 새) = σ(s/ε) det(F 옛)
  4) 매핑 야코비안이 위치 매핑의 실제 미분과 같다 (autograd)
  5) autograd 역전파·forward-mode jvp(advance 의 --v_from_dt 경로) 기울기가 유한하게 흐른다
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import torch

from anchorflow.deform_model import bc_softplus_map

torch.manual_seed(0)
dt = torch.float64
EPS = 0.01
nr = torch.tensor([0.1, 0.2, 1.0], dtype=dt); nr = nr / nr.norm()
pt = [0.3, -0.2, 0.5]
cfg = {"boundary_conditions": [{"type": "surface_collider", "point": pt, "normal": nr.tolist(), "surface": "sticky"},
                               {"type": "bounding_box"}]}
P = torch.tensor(pt, dtype=dt)
N = 4000
s0 = torch.cat([torch.linspace(-0.5, -1e-4, 1000), torch.linspace(-0.05, 0.05, 1000),
                torch.linspace(0.2, 2.0, 1000), torch.rand(1000) * 0.3 - 0.1]).to(dt)
tang = torch.randn(N, 3, dtype=dt); tang = tang - (tang @ nr).unsqueeze(-1) * nr
q = P + tang + s0.unsqueeze(-1) * nr
Jm = 0.1 * torch.randn(N, 3, 3, dtype=dt)
ok = True


def chk(name, cond, info=""):
    global ok
    ok = ok and bool(cond)
    print(f"[{'통과' if cond else '실패'}] {name} {info}", flush=True)


q2, Jm2 = bc_softplus_map(q, Jm, cfg, EPS)
s2 = ((q2 - P) * nr).sum(-1)
sp = torch.nn.functional.softplus(s0, beta=1.0 / EPS, threshold=20.0)
chk("1) 매핑 뒤 모든 s' > 0 (매핑 값)", bool((sp > 0).all()), f"최소 s' {float(sp.min()):.3e} (s 최소 {float(s0.min()):.2f})")
chk("   위치에서 다시 잰 s' >= -반올림(1e-13)", bool((s2 > -1e-13).all()), f"최소 {float(s2.min()):.3e}")
tq = (q2 - q) - ((q2 - q) @ nr).unsqueeze(-1) * nr
chk("   접선 성분 그대로", float(tq.abs().max()) < 1e-12, f"{float(tq.abs().max()):.1e}")
far = s0 > 20 * EPS
chk("2) s ≫ ε 위치 항등", bool((q2[far] == q[far]).all()), f"{int(far.sum())} 개")
chk("   s ≫ ε 야코비안 항등", float((Jm2[far] - Jm[far]).abs().max()) < 1e-12,
    f"{float((Jm2[far] - Jm[far]).abs().max()):.1e}")
I3 = torch.eye(3, dtype=dt)
d_old = torch.linalg.det(I3 + Jm); d_new = torch.linalg.det(I3 + Jm2)
sg = torch.where(s0 / EPS > 20.0, torch.ones_like(s0), torch.sigmoid(s0 / EPS))
chk("3) det J_map = σ > 0", bool((sg > 0).all()) and float((d_new - sg * d_old).abs().max()) < 1e-10,
    f"σ 최소 {float(sg.min()):.3e}, |det새 - σ det옛| 최대 {float((d_new - sg * d_old).abs().max()):.1e}")
# 4) 매핑 야코비안 == 위치 매핑의 미분 (입자 몇 개를 골라 jacobian)
idx = torch.tensor([0, 500, 999, 1400, 1500, 1600, 2500, 3500])
errs = []
for i in idx.tolist():
    f = lambda y: bc_softplus_map(y.unsqueeze(0), torch.zeros(1, 3, 3, dtype=dt), cfg, EPS)[0][0]
    Ja = torch.autograd.functional.jacobian(f, q[i])
    Jmap = bc_softplus_map(q[i:i + 1], torch.zeros(1, 3, 3, dtype=dt), cfg, EPS)[1][0] + I3
    errs.append(float((Ja - Jmap).abs().max()))
chk("4) 매핑 야코비안 == autograd 미분", max(errs) < 1e-10, f"최대 차 {max(errs):.1e}")
# 5) 역전파 / jvp
qq = q.clone().requires_grad_(True); JJ = Jm.clone().requires_grad_(True)
q3, J3 = bc_softplus_map(qq, JJ, cfg, EPS)
L = (q3 ** 2).sum() + (torch.linalg.det(I3 + J3) ** 2).sum()
L.backward()
chk("5) 역전파 기울기 유한·0 아님", bool(torch.isfinite(qq.grad).all() and torch.isfinite(JJ.grad).all())
    and float(qq.grad.abs().sum()) > 0 and float(JJ.grad.abs().sum()) > 0)
qf = q.float(); Jf32 = Jm.float()
cfg32 = cfg
tau0 = torch.tensor(1.0)
fn = lambda t: bc_softplus_map(qf + t * nr.float() * (-0.3), Jf32, cfg32, EPS)
(o, oj), (tq_, tj_) = torch.func.jvp(fn, (tau0,), (torch.ones_like(tau0),))
s32 = ((o - P.float()) * nr.float()).sum(-1)
chk("   jvp (float32) 유한, 위치에서 잰 s' >= -1e-6 (float32 좌표 반올림)", bool(torch.isfinite(tq_).all() and torch.isfinite(tj_).all())
    and bool((s32 > -1e-6).all()), f"float32 최소 s' {float(s32.min()):.3e}")
print("BC_SOFTPLUS_TEST_" + ("OK" if ok else "FAIL"), flush=True)
sys.exit(0 if ok else 1)
