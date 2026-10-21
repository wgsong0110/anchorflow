"""--pfreq K 단위 시험 (CPU float64, 빌드 없음): simplex_gnn.particle_freq_feats / deform_model.node_feats

  1) 손 계산: 사면체 하나(노드 4) + 입자 없는 노드 1 개, 입자 3 개. 독립 계산(파이썬 반복·math.sin/cos)과 채널 전부 일치,
     입자 없는 노드(가중 0)는 전부 0, 폭 = 15(1+2K)
  2) node_feats 경로 (lego 와 같은 node_h, 바닥 surface_collider 하나): --pfreq K 폭 = 15(1+2K) + 바닥 4, pfreq 0 폭 = 29 + 4 = 33;
     --pfreq 앞 채널 == particle_freq_feats(같은 rows·lam·npos) 를 따로 부른 값 (san·상한 뒤), 바닥 4 채널 == pfreq 0 의 바닥 4 채널
  3) 독립 계산(점유 사면체 전 입자 파이썬 반복, 노드 몇 개 표본) 과 node_feats 출력 일치
  4) F 없이 부르면 오류, K 플래그(2·6) 에 따라 폭이 바뀐다
"""
import math
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import torch

from anchorflow import deform_model as DM
from anchorflow import simplex as SX
from anchorflow import simplex_gnn as SG

torch.manual_seed(0)
dt = torch.float64
torch.set_default_dtype(dt)
ok = True


def chk(name, cond, info=""):
    global ok
    ok = ok and bool(cond)
    print(f"[{'통과' if cond else '실패'}] {name} {info}", flush=True)


def brute_node(j, x, v, F, m, rows, lam, npos, hn, K):
    """노드 j 하나의 독립 계산 (파이썬 반복)."""
    num = [0.0] * (15 * (1 + 2 * K)); den = 0.0
    for p in range(x.shape[0]):
        for c in range(rows.shape[1]):
            if int(rows[p, c]) != j:
                continue
            w = float(lam[p, c]) * float(m[p])
            s = [(float(x[p, k]) - float(npos[j, k])) / hn for k in range(3)] + [float(v[p, k]) for k in range(3)] + \
                [float(F[p].reshape(9)[q]) - (1.0 if q in (0, 4, 8) else 0.0) for q in range(9)]
            for i, si in enumerate(s):
                b = i * (1 + 2 * K)
                num[b] += w * si
                for k in range(K):
                    num[b + 1 + k] += w * math.sin((2 ** k) * math.pi * si)
                    num[b + 1 + K + k] += w * math.cos((2 ** k) * math.pi * si)
            den += w
    return torch.tensor([q / max(den, 1e-12) for q in num])


# ---- 1) 손 계산 ----
K = 6
npos = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [5., 5., 5.]]) * 0.05
hn = 0.05
lam = torch.tensor([[0.25, 0.25, 0.25, 0.25], [0.7, 0.1, 0.1, 0.1], [0.0, 0.5, 0.3, 0.2]])
rows = torch.tensor([[0, 1, 2, 3]] * 3)
x = lam @ npos[:4]
v = torch.tensor([[0.1, -0.2, 0.3], [0.0, 0.0, -0.5], [0.4, 0.1, 0.0]])
F = torch.eye(3).expand(3, 3, 3) + 0.05 * torch.randn(3, 3, 3)
m = torch.tensor([1.0, 2.0, 0.5])
out = SG.particle_freq_feats(x, v, F, m, rows, lam, 5, npos, hn, K)
chk("1 폭", out.shape == (5, 15 * (1 + 2 * K)) and SG.pfreq_channels(K) == 195, f"{tuple(out.shape)} (15x(1+2·{K}) = 195)")
ref = torch.stack([brute_node(j, x, v, F, m, rows, lam, npos, hn, K) for j in range(5)])
chk("1 손 계산과 일치", torch.allclose(out, ref, atol=1e-12, rtol=1e-10), f"최대차 {float((out - ref).abs().max()):.2e}")
chk("1 입자 없는 노드 0", bool((out[4] == 0).all()), f"노드 4 최대 |.| {float(out[4].abs().max()):.1e}")
# 첫 채널 하나는 직접 손으로: 노드 0, 상대위치 x 성분의 원값 = Σ λ_p0 m_p (x_p - x_0)/h / Σ λ_p0 m_p
w0 = lam[:, 0] * m
hand = float((w0 * (x[:, 0] - npos[0, 0]) / hn).sum() / w0.sum())
chk("1 노드 0 채널 0 (상대위치 x 원값) 손 계산", abs(float(out[0, 0]) - hand) < 1e-12, f"{float(out[0, 0]):.6f} vs {hand:.6f}")
# 노드 0 의 F_xx-I 채널 (스칼라 6) sin(π s) 칸
i6 = 6 * (1 + 2 * K) + 1
hand6 = float((w0 * torch.sin(math.pi * (F[:, 0, 0] - 1))).sum() / w0.sum())
chk("1 노드 0 F_xx-I 의 sin(π s) 손 계산", abs(float(out[0, i6]) - hand6) < 1e-12, f"{float(out[0, i6]):.6f} vs {hand6:.6f}")

# ---- 2) node_feats 경로 ----
H = 0.028284


def mk_a(pfreq):
    return types.SimpleNamespace(
        hex_jac=False, agg="bary", rel_pe=0, bary_pe=0, interp="bary", n_nodes=32, hz_ratio=SX.HZ_CUBE, node_h=H,
        net_inertial=True, bc_ipg=False, gregory=False, bc_stick=0.01, bc_stick_xn=True, bc_softplus=0.01, bc_soft=False,
        bc_mode="project", bc_level="node", v_from_dt=False, dt_cond=False, ip_feat=False, ip_feat_only=False,
        bc_feat="dist", pfreq=pfreq)


cfg = {"boundary_conditions": [{"type": "surface_collider", "point": [0.0, 0.0, 0.2], "normal": [0.0, 0.0, 1.0],
                                "surface": "sticky"}],
       "grid_lim": 2.0, "n_grid": 64, "g": [0.0, 0.0, -9.8], "E": 2.0e5, "nu": 0.3, "density": 200.0, "material": "jelly"}
N = 600
g_ = torch.Generator().manual_seed(1)
X0 = torch.rand(N, 3, generator=g_) * 0.12 + torch.tensor([0.9, 0.9, 0.25])
Fp = torch.eye(3).expand(N, 3, 3) + 0.1 * torch.randn(N, 3, 3, generator=g_)
xp = torch.einsum("nij,nj->ni", Fp, X0 - X0.mean(0)) + X0.mean(0)
vp = 0.3 * torch.randn(N, 3, generator=g_)
mp = 200.0 * (0.01 ** 3) * (0.5 + torch.rand(N, generator=g_))
VS = 2.5
e0 = DM.node_feats(mk_a(0), cfg, X0, mp, xp, vp, VS, h=1 / 30)
e6 = DM.node_feats(mk_a(6), cfg, X0, mp, xp, vp, VS, h=1 / 30, pF=Fp)
e2 = DM.node_feats(mk_a(2), cfg, X0, mp, xp, vp, VS, h=1 / 30, pF=Fp)
chk("2 폭 pfreq 0 = 29 + 바닥 4", e0[0].shape[1] == 33, f"{e0[0].shape[1]}")
chk("2 폭 pfreq 6 = 195 + 바닥 4", e6[0].shape[1] == 199, f"{e6[0].shape[1]}")
chk("2 폭 pfreq 2 = 75 + 바닥 4", e2[0].shape[1] == 79, f"{e2[0].shape[1]}")
chk("2 바닥 4 채널 == pfreq 0", torch.equal(e6[0][:, -4:], e0[0][:, -4:]))
chk("2 노드·원소 동일 (npos, rows, lam)", torch.equal(e6[1], e0[1]) and torch.equal(e6[3], e0[3]) and torch.equal(e6[4], e0[4]))
npos6, rows6, lam6 = e6[1], e6[3], e6[4]
hn6 = e6[2][1].s
direct = SG.particle_freq_feats(xp, vp / VS, Fp, mp, rows6, lam6, npos6.shape[0], npos6, hn6, 6)
direct = torch.nan_to_num(direct, nan=0.0, posinf=0.0, neginf=0.0).clamp(-DM.FCAP, DM.FCAP)
chk("2 앞 195 채널 == particle_freq_feats 직접", torch.allclose(e6[0][:, :195], direct, atol=1e-12),
    f"최대차 {float((e6[0][:, :195] - direct).abs().max()):.2e}, 노드 {npos6.shape[0]}, h_node {hn6:.6f}")

# ---- 3) 독립 계산 (노드 표본) ----
Mn = npos6.shape[0]
samp = torch.randperm(Mn, generator=g_)[:12].tolist()
md = 0.0
for j in samp:
    r = brute_node(j, xp, vp / VS, Fp, mp, rows6, lam6, npos6, hn6, 6)
    md = max(md, float((e6[0][j, :195] - r).abs().max()))
chk("3 노드 12 개 독립 계산과 일치", md < 1e-10, f"최대차 {md:.2e}")
wsum = torch.zeros(Mn).index_add_(0, rows6.reshape(-1), (lam6 * mp.unsqueeze(1)).reshape(-1))
print(f"    가중 0 노드 {int((wsum == 0).sum())} 개 (있으면 그 특징 전부 0: {bool((e6[0][wsum == 0, :195] == 0).all())})")
chk("3 가중 0 노드는 0", bool((e6[0][wsum == 0, :195] == 0).all()))

# ---- 4) F 없이 부르면 오류 ----
try:
    DM.node_feats(mk_a(6), cfg, X0, mp, xp, vp, VS, h=1 / 30)
    chk("4 F 없으면 오류", False)
except RuntimeError as ex:
    chk("4 F 없으면 오류", True, str(ex)[:60])
from anchorflow.fast_roll import fast_supported
_af = mk_a(6); _af.bc_softplus = 0.0; _af.bc_stick = 0.0
_fs = fast_supported(_af)
chk("4 빠른 경로 거부 (pfreq 이유)", (not _fs[0]) and "pfreq" in _fs[1], _fs[1])
_af.pfreq = 0
print(f"    (같은 설정 pfreq 0: 빠른 경로 {fast_supported(_af)})")

print("PFREQ_TEST_OK" if ok else "PFREQ_TEST_FAIL", flush=True)
