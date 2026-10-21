"""--pfreq K + --bc_feat contact 함께 단위 시험 (CPU float64, 빌드 없음): deform_model.node_feats

  1) 폭: pfreq 6 + contact = 195 + 5 = 200 (pfreq 6 + dist = 199, pfreq 0 + contact = 34)
  2) 앞 195 채널 == pfreq 6 + dist 의 앞 195 (같은 노드·원소), 뒤 5 채널 == pfreq 0 + contact 의 뒤 5 (바닥 근처 상태: 플래그 선 노드·안 선 노드 둘 다 존재)
  3) 뒤 5 채널 == bc_contact_feats 직접 호출, 앞 195 == particle_freq_feats 직접 호출
  4) 공중 상태: 뒤 5 채널 전부 0, 앞 195 는 pfreq 6 + dist 와 같다; F 없으면 오류; 빠른 경로 거부
"""
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
H = 0.028284
h = 1.0 / 30.0
FLOOR = 0.2


def chk(name, cond, info=""):
    global ok
    ok = ok and bool(cond)
    print(f"[{'통과' if cond else '실패'}] {name} {info}", flush=True)


def mk_a(pfreq, bc_feat):
    return types.SimpleNamespace(
        hex_jac=False, agg="bary", rel_pe=0, bary_pe=0, interp="bary", n_nodes=32, hz_ratio=SX.HZ_CUBE, node_h=H,
        net_inertial=True, bc_ipg=False, gregory=False, bc_stick=0.01, bc_stick_xn=True, bc_softplus=0.01, bc_soft=False,
        bc_mode="project", bc_level="node", v_from_dt=False, dt_cond=False, ip_feat=False, ip_feat_only=False,
        bc_feat=bc_feat, pfreq=pfreq)


cfg = {"boundary_conditions": [{"type": "surface_collider", "point": [0.0, 0.0, FLOOR], "normal": [0.0, 0.0, 1.0],
                                "surface": "sticky"}],
       "grid_lim": 2.0, "n_grid": 64, "g": [0.0, 0.0, -9.8], "E": 2.0e5, "nu": 0.3, "density": 200.0, "material": "jelly"}
VS = 2.5


def state(z_lo, seed):
    g_ = torch.Generator().manual_seed(seed)
    N = 700
    X0 = torch.rand(N, 3, generator=g_) * torch.tensor([0.12, 0.12, 0.1]) + torch.tensor([0.9, 0.9, z_lo])
    Fp = torch.eye(3).expand(N, 3, 3) + 0.1 * torch.randn(N, 3, 3, generator=g_)
    x = torch.einsum("nij,nj->ni", Fp, X0 - X0.mean(0)) + X0.mean(0)
    v = 0.3 * torch.randn(N, 3, generator=g_) + torch.tensor([0.0, 0.0, -0.5])
    m = 200.0 * (0.01 ** 3) * (0.5 + torch.rand(N, generator=g_))
    return X0, x, v, m, Fp


# ---- 바닥 근처 상태 ----
X0, x, v, m, Fp = state(FLOOR - 0.02, 1)
eb = DM.node_feats(mk_a(6, "contact"), cfg, X0, m, x, v, VS, h=h, pF=Fp)
ep = DM.node_feats(mk_a(6, "dist"), cfg, X0, m, x, v, VS, h=h, pF=Fp)
ec = DM.node_feats(mk_a(0, "contact"), cfg, X0, m, x, v, VS, h=h)
chk("1 폭 pfreq 6 + contact = 200", eb[0].shape[1] == 200, f"{eb[0].shape[1]}")
chk("1 폭 pfreq 6 + dist = 199, pfreq 0 + contact = 34", ep[0].shape[1] == 199 and ec[0].shape[1] == 34, f"{ep[0].shape[1]}, {ec[0].shape[1]}")
chk("2 노드·원소 동일", torch.equal(eb[1], ep[1]) and torch.equal(eb[3], ec[3]) and torch.equal(eb[4], ec[4]))
chk("2 앞 195 == pfreq 6 + dist 앞 195", torch.equal(eb[0][:, :195], ep[0][:, :195]))
chk("2 뒤 5 == pfreq 0 + contact 뒤 5", torch.equal(eb[0][:, -5:], ec[0][:, -5:]))
fl = eb[0][:, -5]
chk("2 플래그 선 노드·안 선 노드 둘 다", bool((fl == 1).any()) and bool((fl == 0).any()), f"선 노드 {int((fl == 1).sum())}/{fl.numel()}")
chk("2 안 선 노드의 뒤 5 채널 0", bool((eb[0][fl == 0, -5:] == 0).all()))
npos, rows, lam = eb[1], eb[3], eb[4]
hn = eb[2][1].s
bcd = DM.bc_contact_feats(cfg, x, v, m, rows, lam, npos, npos.shape[0], h, hn)
bcd = torch.nan_to_num(bcd, nan=0.0, posinf=0.0, neginf=0.0).clamp(-DM.FCAP, DM.FCAP)
chk("3 뒤 5 == bc_contact_feats 직접", torch.allclose(eb[0][:, -5:], bcd, atol=1e-12), f"최대차 {float((eb[0][:, -5:] - bcd).abs().max()):.1e}")
pfd = SG.particle_freq_feats(x, v / VS, Fp, m, rows, lam, npos.shape[0], npos, hn, 6)
pfd = torch.nan_to_num(pfd, nan=0.0, posinf=0.0, neginf=0.0).clamp(-DM.FCAP, DM.FCAP)
chk("3 앞 195 == particle_freq_feats 직접", torch.allclose(eb[0][:, :195], pfd, atol=1e-12), f"최대차 {float((eb[0][:, :195] - pfd).abs().max()):.1e}")

# ---- 공중 상태 ----
X0, x, v, m, Fp = state(0.8, 2)
v = 0.0 * v
ea = DM.node_feats(mk_a(6, "contact"), cfg, X0, m, x, v, VS, h=h, pF=Fp)
eap = DM.node_feats(mk_a(6, "dist"), cfg, X0, m, x, v, VS, h=h, pF=Fp)
chk("4 공중: 뒤 5 채널 전부 0", bool((ea[0][:, -5:] == 0).all()))
chk("4 공중: 앞 195 == pfreq 6 + dist", torch.equal(ea[0][:, :195], eap[0][:, :195]))
try:
    DM.node_feats(mk_a(6, "contact"), cfg, X0, m, x, v, VS, h=h)
    chk("4 F 없으면 오류", False)
except RuntimeError as ex:
    chk("4 F 없으면 오류", True, str(ex)[:50])
from anchorflow.fast_roll import fast_supported
_af = mk_a(6, "contact"); _af.bc_softplus = 0.0; _af.bc_stick = 0.0
chk("4 빠른 경로 거부", not fast_supported(_af)[0], fast_supported(_af)[1])
print("PFREQ_BCC_TEST_OK" if ok else "PFREQ_BCC_TEST_FAIL", flush=True)
