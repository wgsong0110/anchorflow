"""--ip_feat 단위 시험 (CPU float64, 빌드 없음): deform_model.ip_node_feats 를 학습 경로(advance + pts_ip_energy)와 대조

  설정은 spstick 과 같은 바닥: --bc_stick 0.01 --bc_stick_xn --bc_softplus 0.01, AF_NO_BC=1 (사영 모드라 접촉 벌점 없음),
  --net_inertial (망 출력 0 = 관성 예측 u₀). 입자 일부는 바닥 근처·아래(softplus 영역), 속도 무작위(아래쪽 포함).

  1) 노드 기울기 g == 학습 경로 autograd ∂IP/∂Z (advance 에 탐침 Z 더함, 탄성 = 기본 _FcrPsiFast 경로)
     탄성·관성+중력 분리 기울기도 같은 경로의 항별 autograd 와 같고, 합 == 총합 (AF_NO_BC 라 접촉항 없음)
  2) 블록 대각 헤시안 H_ii == 전체 헤시안(autograd.functional.hessian, 탄성 = 고유값 autograd 경로)의 대각 3×3 블록
  3) 실제 IP 에서 노드 하나만 δ 움직인 변화 == g·δ + ½δᵀHδ + O(δ³) (테일러 나머지가 δ³ 로 줄어듦)
  4) 뉴턴 걸음 d 가 1-노드 이차 근사 m(δ)=g̃·δ+½δᵀ(H̃₊+εI)δ 의 최소점: 정류 잔차 ≈ 0, 무작위 섭동은 m 을 올림;
     양정부호 노드는 (H̃+εI)d = -g̃ 그대로
  5) 바닥 깊이 아래·큰 속도·거의 퇴화한 F 에서도 원시 값·특징 모두 유한, node_feats 폭 = 기본 + 18, 기울기 차단
"""
import math
import os
import sys
import types

os.environ.setdefault("AF_SIG_F64", "1")           # 기준(고유값 autograd) 경로를 float64 로
os.environ.setdefault("AF_NO_COMPILE_FCR", "1")    # CPU 시험: torch.compile 없이
os.environ["AF_NO_BC"] = "1"                       # spstick (bc_mode project) 과 같이 접촉 벌점 없음
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import torch

from anchorflow import deform_model as DM
from anchorflow import phys_resid as PR
from anchorflow import simplex as SX

torch.manual_seed(0)
dt = torch.float64
torch.set_default_dtype(dt)
H = 0.056569
h = 1.0 / 30.0
ok = True


def chk(name, cond, info=""):
    global ok
    ok = ok and bool(cond)
    print(f"[{'통과' if cond else '실패'}] {name} {info}", flush=True)


def mk_a(stick=0.01, sp=0.01, xn=True):
    return types.SimpleNamespace(
        hex_jac=False, agg="bary", rel_pe=0, bary_pe=0, interp="bary", n_nodes=32, hz_ratio=SX.HZ_CUBE, node_h=H,
        net_inertial=True, bc_ipg=False, gregory=False, bc_stick=stick, bc_stick_xn=xn, bc_softplus=sp, bc_soft=False,
        bc_mode="project", bc_level="node", v_from_dt=False, dt_cond=False, ip_feat=True, ip_feat_eps=1e-3)


FLOOR = 0.2
cfg = {"boundary_conditions": [{"type": "surface_collider", "point": [0.0, 0.0, FLOOR], "normal": [0.0, 0.0, 1.0],
                                "surface": "sticky"}],
       "grid_lim": 2.0, "n_grid": 64, "g": [0.0, 0.0, -9.8], "E": 2.0e5, "nu": 0.3, "density": 200.0, "material": "jelly"}


def mk_state(N, z_lo, z_hi, vs=0.3, fs=0.15, seed=0):
    g_ = torch.Generator().manual_seed(seed)
    x = torch.rand(N, 3, generator=g_) * torch.tensor([0.12, 0.12, z_hi - z_lo]) + torch.tensor([0.9, 0.9, z_lo])
    v = vs * torch.randn(N, 3, generator=g_) + torch.tensor([0.0, 0.0, -0.5])
    A = torch.randn(N, 3, 3, generator=g_)
    Qr, _ = torch.linalg.qr(A)
    Qr = Qr * torch.sign(torch.linalg.det(Qr)).reshape(-1, 1, 1)
    s = 1.0 + fs * (torch.rand(N, 3, generator=g_) - 0.5) + torch.tensor([0.0, 0.03, 0.06])   # 서로 다른 특이값
    B = torch.randn(N, 3, 3, generator=g_)
    Vr, _ = torch.linalg.qr(B)
    Vr = Vr * torch.sign(torch.linalg.det(Vr)).reshape(-1, 1, 1)
    F = Qr @ torch.diag_embed(s) @ Vr.transpose(-1, -2)
    m = 200.0 * (0.01 ** 3) * (0.5 + torch.rand(N, generator=g_))
    return x, v, F, m


x, v, F, m = mk_state(300, FLOOR - 0.02, FLOOR + 0.12)
vol = m / cfg["density"]
a = mk_a()
ips = dict(F=F, vol=vol, h=h, free=None)
ens = DM.node_feats(a, cfg, x, m, x, v, 1.0, ip_state=ips)
raw = DM.IPF_LAST[0]
_in, npos, (lo, lat, nn), rows, lam, uniq, _e, aux = ens
Mn = npos.shape[0]
print(f"[설정] 입자 {x.shape[0]}, 노드 {Mn}, 바닥 아래 입자 {int((x[:, 2] < FLOOR).sum())}, 특징 폭 {_in.shape[1]}", flush=True)
u0, (vn, gq) = DM.node_inertial(x, v, m, rows, lam, Mn, cfg["g"], h)


def E_ref(Z, parts=False):
    """학습 경로 그대로: advance(망 출력 = 0 + 탐침 Z, --net_inertial 관성 예측 더함) -> pts_ip_energy (정규화 없음)."""
    DM.PROBE[0], DM.PROBE[1] = "fixed", [Z.reshape(Mn, 3)]
    try:
        # dp_fast 를 쓰면 advance 는 망을 건너뛰고 관성 예측도 더하지 않는다 -> 망 출력 0 + τ·v_n + τ²g (τ = h) = u₀ 를 직접 준다
        x2, _vn, _dpn, Jf, _c, _act = DM.advance(None, a, [ens], x, h, cfg, dp_fast=lambda M, _d: (u0.clone(),),
                                                 inertial=[(vn, gq)])
    finally:
        DM.PROBE[0], DM.PROBE[1] = None, None
    E, _dl, _Ft, pr = PR.pts_ip_energy(x, x2 - x, v, F, Jf, m, vol, cfg, h, cfg["n_grid"], cfg["grid_lim"],
                                       g=torch.tensor(cfg["g"]), parts_grad=parts)
    return (E, pr) if parts else E


# 1) 기울기
Z = torch.zeros(Mn * 3, requires_grad=True)
E0, pr = E_ref(Z, parts=True)
gR, = torch.autograd.grad(E0, Z, retain_graph=True)
gelR, = torch.autograd.grad(pr[1], Z, retain_graph=True)
ginR, = torch.autograd.grad(pr[0], Z)
gR, gelR, ginR = gR.reshape(Mn, 3), gelR.reshape(Mn, 3), ginR.reshape(Mn, 3)
rel = lambda A, B: float((A - B).abs().max() / B.abs().max().clamp_min(1e-300))
chk("1) 노드 기울기 == 학습 경로 autograd", rel(raw["g"], gR) < 1e-9, f"상대 최대오차 {rel(raw['g'], gR):.1e}, |g|최대 {float(gR.abs().max()):.3e}")
chk("   탄성 기울기 == 학습 경로 탄성항 autograd", rel(raw["g_el"], gelR) < 1e-9, f"{rel(raw['g_el'], gelR):.1e}")
chk("   관성+중력 기울기 == 학습 경로 관성항 autograd", rel(raw["g_in"], ginR) < 1e-9, f"{rel(raw['g_in'], ginR):.1e}")
chk("   탄성 + 관성+중력 == 총합 (AF_NO_BC)", rel(raw["g_el"] + raw["g_in"], raw["g"]) < 1e-12, f"{rel(raw['g_el'] + raw['g_in'], raw['g']):.1e}")
chk("   IP 값 == 학습 경로", abs(float(raw["E"]) - float(E0)) <= 1e-10 * abs(float(E0)), f"{float(raw['E']):.6e} vs {float(E0):.6e}")

# 2) 블록 대각 헤시안 == 전체 헤시안의 대각 블록 (탄성 = 고유값 autograd 경로)
os.environ["AF_NO_FAST_FCR"] = "1"
Hfull = torch.autograd.functional.hessian(lambda z: E_ref(z), torch.zeros(Mn * 3))
del os.environ["AF_NO_FAST_FCR"]
Hd = torch.stack([Hfull[3 * i:3 * i + 3, 3 * i:3 * i + 3] for i in range(Mn)])
eh = rel(raw["H"], Hd)
chk("2) 블록 대각 헤시안 == 전체 헤시안 대각 블록", eh < 1e-7, f"상대 최대오차 {eh:.1e}, |H|최대 {float(Hd.abs().max()):.3e}")
_off = Hfull.clone()
for i in range(Mn):
    _off[3 * i:3 * i + 3, 3 * i:3 * i + 3] = 0
print(f"    (참고) 비대각 블록 최대 {float(_off.abs().max()):.3e} -- 블록 대각만 특징으로 쓴다", flush=True)

# 3) 실제 IP 의 1-노드 테일러: 나머지가 δ³ 로 줄어듦
with torch.no_grad():
    Eb = float(E_ref(torch.zeros(Mn * 3)))
    nod = torch.argsort(raw["M"], descending=True)[:5].tolist()
    worst = 0.0
    for i in nod:
        dvec = torch.randn(3)
        dvec = dvec / dvec.norm()
        rs = []
        for sc in (1e-3, 5e-4):
            dl = sc * H * dvec
            Zt = torch.zeros(Mn, 3)
            Zt[i] = dl
            dE = float(E_ref(Zt.reshape(-1))) - Eb
            qm = float(raw["g"][i] @ dl + 0.5 * dl @ raw["H"][i] @ dl)
            rs.append(abs(dE - qm))
        worst = max(worst, rs[1] / max(rs[0], 1e-300))
    chk("3) 1-노드 테일러 나머지 O(δ³) (δ 반으로 -> 나머지 1/8 근처)", worst < 0.2, f"나머지 비 최대 {worst:.3f}")

# 4) 뉴턴 걸음 = 1-노드 이차 근사 최소점
Ht, gt, d = raw["Ht"], raw["gt"], raw["step"]
eps = raw["eps"]
ev, Q = torch.linalg.eigh(Ht)
Hp = Q @ torch.diag_embed(ev.clamp_min(0.0) + eps) @ Q.transpose(-1, -2)
res = (Hp @ d.unsqueeze(-1)).squeeze(-1) + gt
chk("4) 정류: (H̃₊+εI) d + g̃ = 0", float(res.abs().max() / gt.abs().max()) < 1e-10, f"{float(res.abs().max() / gt.abs().max()):.1e}")
pd = ev.min(-1).values > 0
res2 = ((Ht[pd] + eps * torch.eye(3)) @ d[pd].unsqueeze(-1)).squeeze(-1) + gt[pd]
chk("   양정부호 노드: (H̃+εI) d = -g̃", float(res2.abs().max() / gt.abs().max()) < 1e-10, f"{int(pd.sum())}/{Mn} 노드, {float(res2.abs().max() / gt.abs().max()):.1e}")
mq = lambda dd: (gt * dd).sum(-1) + 0.5 * (dd.unsqueeze(-2) @ Hp @ dd.unsqueeze(-1)).squeeze(-1).squeeze(-1)
worse = True
for _ in range(20):
    pert = 1e-2 * d.abs().max() * torch.randn(Mn, 3)
    worse = worse and bool((mq(d + pert) >= mq(d) - 1e-15 * mq(d).abs().max()).all())
chk("   무작위 섭동은 이차 근사를 올린다", worse, f"음의 고윳값 노드 {int((~pd).sum())} 개 (클램프)")
f = _in[:, -DM.IP_FEAT_N:]
chk("   특징 채널 = 정규화 값 (g̃/h, H̃, d/h)", rel(f[:, 15:18], (d / raw["hn"]).clamp(-DM.FCAP, DM.FCAP)) < 1e-12
    and rel(f[:, 0:3], (gt / raw["hn"]).clamp(-DM.FCAP, DM.FCAP)) < 1e-12, "")

# 5) 바닥 아래 깊이·큰 속도·퇴화 F 에서 유한
x5, v5, F5, m5 = mk_state(400, FLOOR - 0.3, FLOOR + 0.05, vs=5.0, seed=3)
F5[:50] = F5[:50] * torch.tensor([1.0, 1.0, 1e-4]).reshape(1, 1, 3)       # 거의 퇴화
F5[50:60] = F5[50:60] * torch.tensor([1.0, 1.0, -0.5]).reshape(1, 1, 3)   # 뒤집힘
for nm, a5 in (("stick+xn+softplus", mk_a()), ("softplus 만", mk_a(stick=0.0)), ("매핑 없음", mk_a(stick=0.0, sp=0.0))):
    e5 = DM.node_feats(a5, cfg, x5, m5, x5, v5, 1.0, ip_state=dict(F=F5, vol=m5 / cfg["density"], h=h, free=None))
    r5 = DM.IPF_LAST[0]
    fin = all(bool(torch.isfinite(r5[k]).all()) for k in ("g", "g_el", "g_in", "H", "step"))
    chk(f"5) 유한 ({nm})", fin and bool(torch.isfinite(e5[0]).all()) and not e5[0].requires_grad,
        f"바닥 아래 입자 {int((x5[:, 2] < FLOOR).sum())}, 특징 |최대| {float(e5[0][:, -DM.IP_FEAT_N:].abs().max()):.2f}")
a0 = mk_a()
a0.ip_feat = False
w0 = DM.node_feats(a0, cfg, x, m, x, v, 1.0)[0].shape[1]
chk("   node_feats 폭 = 기본 + 18", _in.shape[1] == w0 + DM.IP_FEAT_N == w0 + 18, f"{w0} + {DM.IP_FEAT_N} = {_in.shape[1]}")
print("IP_FEAT_TEST_OK" if ok else "IP_FEAT_TEST_FAIL", flush=True)
