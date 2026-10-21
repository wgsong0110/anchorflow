"""--bc_feat contact 단위 시험 (CPU float64, 빌드 없음): deform_model.node_feats 의 바닥 특징을 독립 계산(파이썬 반복)과 대조

  1) 공중 상태 (모든 점유 사면체 꼭짓점이 바닥 위, 관성 예측도 위): 바닥 5 채널이 모든 노드에서 0, 폭 = dist 판본 + 1
  2) 바닥 근처 상태 (입자 일부가 바닥 아래·근처): 플래그가 선 노드 집합 == 독립 계산(점유 사면체별 꼭짓점 판정의 합집합),
     선 노드는 [1, 부호거리/hn, 법선], 안 선 노드는 전부 0, 둘 다 존재; 바닥 채널 외 특징은 dist 판본과 같다
  3) 관성 예측 조건 (b): 바닥에서 떨어진 상태에서 속도 0 이면 전부 0, 큰 아래 속도면 플래그가 서고 그 셀들은 현재 위치로는
     접촉이 아니다(조건 a 불성립) -- 플래그 노드 집합 == 독립 계산
  4) 기울기 차단 (x·v requires_grad 여도 바닥 채널은 그래프 없음), h 없이 부르면 오류
"""
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import torch

from anchorflow import deform_model as DM
from anchorflow import simplex as SX

torch.manual_seed(0)
dt = torch.float64
torch.set_default_dtype(dt)
H = 0.056569
h = 1.0 / 30.0
FLOOR = 0.2
ok = True


def chk(name, cond, info=""):
    global ok
    ok = ok and bool(cond)
    print(f"[{'통과' if cond else '실패'}] {name} {info}", flush=True)


def mk_a(bc_feat):
    return types.SimpleNamespace(
        hex_jac=False, agg="bary", rel_pe=0, bary_pe=0, interp="bary", n_nodes=32, hz_ratio=SX.HZ_CUBE, node_h=H,
        net_inertial=True, bc_ipg=False, gregory=False, bc_stick=0.01, bc_stick_xn=True, bc_softplus=0.01, bc_soft=False,
        bc_mode="project", bc_level="node", v_from_dt=False, dt_cond=False, ip_feat=False, ip_feat_only=False,
        bc_feat=bc_feat)


cfg = {"boundary_conditions": [{"type": "surface_collider", "point": [0.0, 0.0, FLOOR], "normal": [0.0, 0.0, 1.0],
                                "surface": "sticky"}],
       "grid_lim": 2.0, "n_grid": 64, "g": [0.0, 0.0, -9.8], "E": 2.0e5, "nu": 0.3, "density": 200.0, "material": "jelly"}
NB = 5                                   # 바닥 채널 수 (contact): 플래그 1 + 거리 1 + 법선 3


def mk_state(N, z_lo, z_hi, vz=-0.5, vs=0.3, seed=0):
    g_ = torch.Generator().manual_seed(seed)
    x = torch.rand(N, 3, generator=g_) * torch.tensor([0.15, 0.15, z_hi - z_lo]) + torch.tensor([0.9, 0.9, z_lo])
    v = vs * torch.randn(N, 3, generator=g_) + torch.tensor([0.0, 0.0, vz])
    m = 200.0 * (0.01 ** 3) * (0.5 + torch.rand(N, generator=g_))
    return x, v, m


def feats(x, v, m, bc_feat="contact", hh=h):
    return DM.node_feats(mk_a(bc_feat), cfg, x, m, x, v, 1.0, h=hh)


def brute(ens, x, v, m, hh=h):
    """독립 계산: 노드 관성 예측(파이썬 반복 질량가중 평균) -> 점유 사면체별 판정 -> 꼭짓점 합집합. -> (플래그, 조건a셀 노드, 부호거리)"""
    npos, rows, lam = ens[1], ens[3], ens[4]
    Mn = npos.shape[0]
    ws = [0.0] * Mn
    vs = [[0.0, 0.0, 0.0] for _ in range(Mn)]
    for p in range(rows.shape[0]):
        for j in range(4):
            i = int(rows[p, j]); w = float(lam[p, j] * m[p])
            ws[i] += w
            for c in range(3):
                vs[i][c] += w * float(v[p, c])
    g = cfg["g"]
    sd = [float(npos[i, 2]) - FLOOR for i in range(Mn)]
    sp = [sd[i] + (hh * vs[i][2] / max(ws[i], 1e-30) + hh * hh * g[2]) for i in range(Mn)]
    fl = [False] * Mn
    fa = [False] * Mn
    for tet in {tuple(int(r) for r in rows[p]) for p in range(rows.shape[0])}:
        ca = any(sd[i] <= 0 for i in tet)
        cb = any(sp[i] <= 0 for i in tet)
        for i in tet:
            if ca or cb:
                fl[i] = True
            if ca:
                fa[i] = True
    return torch.tensor(fl), torch.tensor(fa), torch.tensor(sd)


# ---- 1) 공중 ----
x, v, m = mk_state(300, FLOOR + 0.4, FLOOR + 0.55, vz=-0.5)
e = feats(x, v, m)
e0 = feats(x, v, m, "dist")
B = e[0][:, -NB:]
fl, fa, sd = brute(e, x, v, m)
chk("1) 공중: 바닥 채널 모든 노드 0", float(B.abs().max()) == 0.0, f"(최대 {float(B.abs().max()):.3g}, 노드 {B.shape[0]}, 최소 부호거리 {float(sd.min()):.3f})")
chk("1) 폭 = dist 판본 + 1", e[0].shape[1] == e0[0].shape[1] + 1, f"({e0[0].shape[1]} -> {e[0].shape[1]})")
chk("1) 독립 계산도 플래그 없음", not bool(fl.any()))

# ---- 2) 바닥 근처 ----
x, v, m = mk_state(600, FLOOR - 0.03, FLOOR + 0.25, vz=-0.3, seed=1)
e = feats(x, v, m)
e0 = feats(x, v, m, "dist")
B = e[0][:, -NB:]
hn = float(e[2][1].s)
fl, fa, sd = brute(e, x, v, m)
flag = B[:, 0]
chk("2) 플래그 값은 0 또는 1", bool(((flag == 0) | (flag == 1)).all()))
chk("2) 플래그 노드 집합 == 독립 계산", bool(((flag == 1) == fl).all()),
    f"(선 노드 {int(fl.sum())} / {fl.numel()}, 불일치 {int(((flag == 1) != fl).sum())})")
chk("2) 선 노드도 안 선 노드도 있다", 0 < int(fl.sum()) < fl.numel())
on = flag == 1
exp_on = torch.cat([torch.ones(int(on.sum()), 1), (sd[on] / hn).unsqueeze(1),
                    torch.tensor([[0.0, 0.0, 1.0]]).expand(int(on.sum()), 3)], 1)
chk("2) 선 노드 = [1, 부호거리/hn, 법선]", torch.allclose(B[on], exp_on, atol=1e-12),
    f"(거리/hn 범위 {float(B[on, 1].min()):+.3f} ~ {float(B[on, 1].max()):+.3f})")
chk("2) 안 선 노드는 5 채널 모두 0", float(B[~on].abs().max()) == 0.0)
chk("2) 바닥 아닌 특징은 dist 판본과 같다", torch.equal(e[0][:, :-NB], e0[0][:, :-4]))
chk("2) 바닥 아래 꼭짓점 가진 노드는 모두 선다", bool(on[sd <= 0].all()), f"(부호거리≤0 노드 {int((sd <= 0).sum())})")

# ---- 3) 관성 예측 조건 (b) ----
z0 = FLOOR + 2.2 * H
x, v, m = mk_state(400, z0, z0 + 0.12, vz=0.0, vs=0.0, seed=2)
e = feats(x, v, m)
fl, fa, sd = brute(e, x, v, m)
_mn = min(float(sd[torch.unique(e[3].reshape(-1))].min()), 9.0)
chk("3) 속도 0: 바닥 채널 0 (조건 a·b 모두 불성립)", float(e[0][:, -NB:].abs().max()) == 0.0, f"(점유 사면체 꼭짓점 최소 부호거리 {_mn:.4f})")
vdown = -(_mn + 0.3 * H) / h                     # 가장 낮은 꼭짓점 아래로 넘을 만큼
x2, v2, m2 = x, torch.zeros_like(x) + torch.tensor([0.0, 0.0, vdown]), m
e = feats(x2, v2, m2)
B = e[0][:, -NB:]
fl, fa, sd = brute(e, x2, v2, m2)
chk("3) 큰 아래 속도: 플래그가 선다", int((B[:, 0] == 1).sum()) > 0, f"(v_z {vdown:.2f}, 선 노드 {int((B[:, 0] == 1).sum())} / {B.shape[0]})")
chk("3) 그 플래그는 조건 a 로는 안 선다 (전부 b 로만)", not bool(fa.any()))
chk("3) 플래그 노드 집합 == 독립 계산", bool(((B[:, 0] == 1) == fl).all()))
chk("3) 선 노드의 거리는 양수 (현재 위치는 바닥 위)", bool((B[B[:, 0] == 1, 1] > 0).all()))
v3 = torch.zeros_like(x) + torch.tensor([0.0, 0.0, 0.5 * vdown / 4])
e3 = feats(x, v3, m)
n1, n3 = int((B[:, 0] == 1).sum()), int((e3[0][:, -NB] == 1).sum())
chk("3) 아래 속도가 작으면 덜 선다", n3 < n1, f"({n3} < {n1})")

# ---- 4) 기울기 차단, h 필수 ----
x, v, m = mk_state(300, FLOOR - 0.03, FLOOR + 0.2, seed=3)
xg, vg = x.clone().requires_grad_(True), v.clone().requires_grad_(True)
Bc = DM.bc_contact_feats(cfg, xg, vg, m, *[feats(x, v, m)[k] for k in (3, 4, 1)], feats(x, v, m)[1].shape[0], h, hn)
chk("4) 기울기 차단", not Bc.requires_grad and Bc.grad_fn is None)
try:
    feats(x, v, m, hh=None)
    chk("4) h 없으면 오류", False)
except RuntimeError as ex:
    chk("4) h 없으면 오류", True, f"({str(ex)[:40]}...)")

print("BC_CONTACT_TEST_OK" if ok else "BC_CONTACT_TEST_FAIL", flush=True)
sys.exit(0 if ok else 1)
