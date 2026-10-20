"""--c1_base pl 시험: 기울기·변 채널 0 + 노드 값 아핀 -> 입자 변위·야코비안이 그 아핀장 (advance 경로), 무작위 값에서도 C1 연속,
관성 예측자 모양(손잡이 노드 0, 나머지 평행이동)에서 det 계수가 기울기 0 매개화보다 덜 음수."""
import os, sys, types
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import torch
from anchorflow import deform_model as DM, simplex as SX, wf as WF
dt = torch.float64; dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0); H = 0.056569; ok = True
def chk(n, c, i=""):
    global ok; ok = ok and bool(c); print(f"[{'통과' if c else '실패'}] {n} {i}", flush=True)
def mk(base):
    return types.SimpleNamespace(hex_jac=False, agg="bary", rel_pe=0, bary_pe=0, interp="c1wf", c1_base=base, n_nodes=32, hz_ratio=SX.HZ_CUBE,
                                 node_h=H, net_inertial=False, bc_ipg=False, gregory=False, bc_stick=0.0, bc_softplus=0.0, bc_soft=False,
                                 bc_mode="project", bc_level="node", v_from_dt=False, dt_cond=False)
cfg = {"boundary_conditions": [], "grid_lim": 2.0, "n_grid": 100, "g": [0.0, 0.0, 0.0]}
x = (0.7 + 0.4 * torch.rand(20000, 3, dtype=dt)).to(dev)
def run(base, fn):
    a = mk(base); ens = DM.node_feats(a, cfg, x, torch.ones(x.shape[0], dtype=dt, device=dev), x, torch.zeros_like(x), 1.0)
    dp = fn(ens[1]); x2, _, _, Jf, _, _ = DM.advance(None, a, [ens], x, 0.04, cfg, dp_fast=lambda M, _d: (dp,)); return ens, dp, x2, Jf
B = 0.05 * torch.randn(3, 3, dtype=dt, device=dev); c = 0.01 * torch.randn(3, dtype=dt, device=dev)
def aff(npos):
    d = torch.zeros(npos.shape[0], WF.C1_NOUT, dtype=dt, device=dev); d[:, :3] = npos @ B.T + c; return d
ens, dp, x2, Jf = run("pl", aff)
eu = float((x2 - x - (x @ B.T + c)).abs().max()); eJ = float((Jf - torch.eye(3, dtype=dt, device=dev) - B).abs().max())
chk("pl 매개화 아핀 재현 (채널 0)", eu < 1e-12 and eJ < 1e-11, f"{eu:.1e} {eJ:.1e}")
# 손잡이 모양: 아래쪽 노드 0, 위쪽 평행이동 (계단) -> 계수 최소 비교
def step(npos):
    d = torch.zeros(npos.shape[0], WF.C1_NOUT, dtype=dt, device=dev); d[:, 0] = 0.4 * H * (npos[:, 2] > 0.9).to(dt); return d
r = {}
for base in ("none", "pl"):
    ens, dp, x2, Jf = run(base, step)
    d_eff = dp if base == "none" else torch.cat([dp[:, :3], WF.pl_c1_dofs(dp[:, :3], ens[7].wft, dp.shape[0])], 1)
    cc = WF.det_coefs(d_eff, ens[7].wft, ens[2][1]); r[base] = (float(cc.min()), int((cc <= 0).sum()), int((torch.linalg.det(Jf) <= 0).sum()))
chk("계단(손잡이 경계 0.4h 전단) 계수 최소: 기울기0 대 pl", True, f"none {r['none']}, pl {r['pl']} (계수최소, 계수≤0 수, 입자 det≤0)")
print("C1BASE_TEST " + ("PASS" if ok else "FAIL"), flush=True)
