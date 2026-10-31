"""절 AK.10 충격 때 더 눌리는 원인 가르기 (계산 노드, 측정 밖): 우리 덤프 (roll_fem.pt: 위치·F·속도 f32) 와 정답 (같은 .pt 의 위치·속도·F) 을 충격 창 프레임마다 비교.
(a) 강성: 탄성 부피 합 (우리 셀 탄성 = 입자 적분점 Σ V_p ψ -- 상한 없음이라 정답과 같은 V_p), 같은 압축에서의 응력 (입자 Kirchhoff τ 를 압축 J 구간별 평균 -- 재질 식이 같으면 같아야 함),
    몸 전체 수직 응력 적분 Σ V_p τ_zz (탄성이 몸을 떠받치는 크기);
(b) 바닥: 바닥 힘 = 운동량 수지 M (Δv_c/h − g) (정답·우리 같은 식), 접촉 입자 수 (평면 아래 / 평면 위 1 cm 안), 접촉 입자의 수평 속도 (고착);
(c) 해상도: 바닥에서의 높이 띠별 평균 J (0 에서 2, 2 에서 5, 5 에서 10, 10 에서 20, 20 에서 40 cm), 정답 J 를 우리 셀 크기 (node_h 정육면체) 로 평균한 것과의 차 (셀 해상도에서 보이는 압축).
사용: python exe/fem_ak_press.py <roll_fem.pt> <data.pt> <첫 접촉 프레임 (1 부터)> [창 길이 20] [node_h 0.028284]"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import torch                                                                         # noqa: E402
from anchorflow import deform_model as DM                                           # noqa: E402
from anchorflow import phys_resid as PR                                             # noqa: E402

rp, dp, c0 = sys.argv[1], sys.argv[2], int(sys.argv[3])
W = int(sys.argv[4]) if len(sys.argv) > 4 else 20
NH = float(sys.argv[5]) if len(sys.argv) > 5 else 0.028284
dev = "cuda"
R = torch.load(rp, map_location="cpu", weights_only=False)
d = torch.load(dp, map_location="cpu", weights_only=False)
cfg = d["cfg"]
h = float(cfg["frame_dt"])
m = DM.traj_mass(d, dev).double()
vol = m / float(cfg["density"])
M = float(m.sum())
mu, lam = PR.lame(cfg["E"], cfg["nu"])
GF = DM.traj_F(d, h, "cpu")
pl = [b for b in cfg["boundary_conditions"] if b.get("type") == "surface_collider"][0]
pz = float(pl["point"][2])
g = torch.tensor([float(q) for q in cfg["g"]], device=dev, dtype=torch.float64)
BANDS = [0.0, 0.02, 0.05, 0.10, 0.20, 0.40]


def tau(Fm):
    U, S, Vh = torch.linalg.svd(Fm)
    Q = U @ Vh
    J = torch.linalg.det(Fm)
    I3 = torch.eye(3, device=dev, dtype=torch.float64)
    return 2 * mu * (Fm - Q) @ Fm.transpose(1, 2) + (lam * (J - 1) * J)[:, None, None] * I3, J


def cellavg(x, J):
    c = torch.floor(x / NH).long()
    c = c - c.min(0).values
    k = (c[:, 0] * 4096 + c[:, 1]) * 4096 + c[:, 2]
    u, inv = torch.unique(k, return_inverse=True)
    s = torch.zeros(u.numel(), device=dev, dtype=torch.float64).index_add_(0, inv, vol.to(dev) * J)
    w = torch.zeros(u.numel(), device=dev, dtype=torch.float64).index_add_(0, inv, vol.to(dev))
    return (s / w)[inv]


def stats(x, v, F, xp, vp):
    x, v, F = x.to(dev).double(), v.to(dev).double(), F.to(dev).double().reshape(-1, 3, 3)
    t, J = tau(F)
    hz = x[:, 2] - pz
    r = {}
    vv = vol.to(dev)
    r["tzz_int"] = float((vv * t[:, 2, 2]).sum())                                    # 몸 전체 Σ V τ_zz (음 = 압축으로 떠받침)
    r["p_q"] = [round(float(q), 1) for q in torch.quantile(-(t.diagonal(dim1=1, dim2=2).sum(-1) / 3)[:2_000_000], torch.tensor([0.5, 0.9, 0.99], device=dev, dtype=torch.float64)).tolist()]
    jb = []
    for lo, hi in zip(BANDS[:-1], BANDS[1:]):
        s_ = (hz >= lo) & (hz < hi)
        jb.append(round(float((vv[s_] * J[s_]).sum() / vv[s_].sum()), 4) if bool(s_.any()) else None)
    r["J_band"] = jb
    jc = cellavg(x, J)
    r["J_q"] = [round(float(q), 4) for q in torch.quantile(J[:2_000_000], torch.tensor([0.001, 0.01, 0.1], device=dev, dtype=torch.float64)).tolist()]
    r["Jcell_q"] = [round(float(q), 4) for q in torch.quantile(jc[:2_000_000], torch.tensor([0.001, 0.01, 0.1], device=dev, dtype=torch.float64)).tolist()]
    r["n_below"] = int((hz < 0).sum())
    r["n_band1cm"] = int(((hz >= 0) & (hz < 0.01)).sum())
    near = hz < 0.01
    r["vh_contact"] = float(v[near][:, :2].norm(dim=-1).mean()) if bool(near.any()) else None
    if xp is not None:
        vc = (m.to(dev)[:, None] * v).sum(0) / M
        vc0 = (m.to(dev)[:, None] * vp.to(dev).double()).sum(0) / M
        r["F_floor_z"] = float(M * ((vc[2] - vc0[2]) / h - g[2]))                    # 운동량 수지: 바닥이 몸에 준 평균 수직 힘 (N)
        r["F_floor_h"] = float(M * ((vc[:2] - vc0[:2]) / h).norm())
    # 같은 압축에서 응력 (재질 식 검산): J 0.9 에서 0.95 구간 평균 압력
    sel = (J > 0.9) & (J < 0.95)
    r["p_at_J09"] = float((-(t.diagonal(dim1=1, dim2=2).sum(-1) / 3))[sel].mean()) if bool(sel.any()) else None
    r["height"] = float(x[:, 2].max() - x[:, 2].min())
    return r


out = dict(src=rp, c0=c0, frames=[])
L = R["pred"].shape[0]
for f in range(max(1, c0 - 2), min(L, c0 + W) + 1):                                   # f = 프레임 (1 부터); 덤프 인덱스 f-1, 정답 d 인덱스 f
    i = f - 1
    o = stats(R["pred"][i], R["v"][i], R["F"][i], R["pred"][i - 1] if i > 0 else None, R["v"][i - 1] if i > 0 else None)
    gq = stats(d["x"][f], d["v"][f], GF[f], d["x"][f - 1], d["v"][f - 1])
    out["frames"].append(dict(f=f, ours=o, gt=gq))
    print(f"f{f}: 높이 {o['height']:.3f}/{gq['height']:.3f}  ΣVτzz {o['tzz_int']:.0f}/{gq['tzz_int']:.0f}  바닥힘z {o.get('F_floor_z', 0):.0f}/{gq.get('F_floor_z', 0):.0f} N"
          f"  아래 {o['n_below']}/{gq['n_below']} 띠1cm {o['n_band1cm']}/{gq['n_band1cm']}  J띠 {o['J_band']}/{gq['J_band']}  J 0.1·1·10% {o['J_q']}/{gq['J_q']} 셀평균 {o['Jcell_q']}/{gq['Jcell_q']}"
          f"  p(J0.9-0.95) {o['p_at_J09']}/{gq['p_at_J09']}", flush=True)
json.dump(out, open(os.path.join(os.path.dirname(rp), "press.json"), "w"), indent=1, ensure_ascii=False)
