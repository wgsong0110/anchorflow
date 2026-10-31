"""절 AK.15 남는 탄성의 자리·모양 (계산 노드, 측정 밖): 우리 f32 덤프 (roll_fem.pt) 와 정답 (.pt 위치·속도·F) 을 정답 최대 압축 프레임부터 W 프레임 비교.
(a) 입자 탄성 V ψ 의 공간 분포: 몸 좌표 (0 프레임 대비 Kabsch) 높이 8 띠 x 수평 4 부위 (몸 x·y 반쪽) 지도, 우리 / 정답 / 차
(b) 부피 몫 V λ/2 (J−1)² 와 편차 몫 V μ Σ(σ−1)² (FCR ψ 의 두 항 그대로), 가장 눌린 주 방향과 바닥 법선의 |cos| (에너지 가중)
(c) 셀 (node_h 정육면체) 안 상대 운동 에너지 Σ ½ m |v − v̄_셀|² 의 프레임 시계열 (전체 80 프레임) 과 그 시간 스펙트럼 (프레임 간격 표본 -- 서브스텝 안 진동은 못 봄)
(d) 갇힌 입자 (입자 탄성이 같은 입자 정답보다 큰 상위 10 %) 가 직전 5 프레임 동안 셀을 몇 번 바꿨는지 vs 나머지
사용: python exe/fem_ak_local.py <roll_fem.pt> <data.pt> <첫 접촉 프레임> [W 15] [node_h 0.028284] -> 덤프 폴더 local.json"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import torch                                                                         # noqa: E402
from anchorflow import deform_model as DM                                           # noqa: E402
from anchorflow import phys_resid as PR                                             # noqa: E402

rp, dp, c0 = sys.argv[1], sys.argv[2], int(sys.argv[3])
W = int(sys.argv[4]) if len(sys.argv) > 4 else 15
NH = float(sys.argv[5]) if len(sys.argv) > 5 else 0.028284
dev = "cuda"
R = torch.load(rp, map_location="cpu", weights_only=False)
d = torch.load(dp, map_location="cpu", weights_only=False)
cfg = d["cfg"]
h = float(cfg["frame_dt"])
m = DM.traj_mass(d, dev).double()
V = (m / float(cfg["density"])).double()
mu, lam = PR.lame(cfg["E"], cfg["nu"])
GF = DM.traj_F(d, h, "cpu")
nrm = torch.tensor([float(q) for q in [b for b in cfg["boundary_conditions"] if b.get("type") == "surface_collider"][0]["normal"]], device=dev, dtype=torch.float64)
x0 = d["x"][0].to(dev).double()
q0 = x0 - x0.mean(0)
zlo, zhi = float(q0[:, 2].min()), float(q0[:, 2].max())
L = R["pred"].shape[0]


def body(x):
    c = x.mean(0)
    U_, S_, Vh_ = torch.linalg.svd(q0.T @ (x - c))
    Dg = torch.eye(3, device=dev, dtype=torch.float64)
    Dg[2, 2] = torch.sign(torch.det(Vh_.T @ U_.T))
    Rm = Vh_.T @ Dg @ U_.T
    return (x - c) @ Rm, float(((x - c) @ Rm)[:, 2].max() - ((x - c) @ Rm)[:, 2].min())


def parts(F):
    U_, S_, Vh_ = torch.linalg.svd(F)
    J = torch.linalg.det(F)
    ev = V * 0.5 * lam * (J - 1) ** 2
    ed = V * mu * ((S_ - 1) ** 2).sum(-1)
    dmin = U_[:, :, 2]                                                               # 가장 작은 특이값 방향 (현재 배치)
    return ev, ed, (dmin @ nrm).abs(), J


# 정답 최대 압축 프레임 (몸 좌표 높이 최저, 첫 접촉부터 25 프레임 안)
hs = []
for f in range(c0, min(L, c0 + 25) + 1):
    hs.append((body(d["x"][f].to(dev).double())[1], f))
fmin = min(hs)[1]
frames = list(range(fmin, min(L, fmin + W - 1) + 1))
# (a)(b): 지도·몫
bins_z = 8
MAP = {"ours": torch.zeros(bins_z, 4, device=dev, dtype=torch.float64), "gt": torch.zeros(bins_z, 4, device=dev, dtype=torch.float64)}
SP = {k: dict(ev=0.0, ed=0.0, cosw=0.0) for k in ("ours", "gt")}
for f in frames:
    for k, (x, F) in (("ours", (R["pred"][f - 1], R["F"][f - 1])), ("gt", (d["x"][f], GF[f]))):
        x, F = x.to(dev).double(), F.to(dev).double().reshape(-1, 3, 3)
        b, _ = body(x)
        ev, ed, cs, J = parts(F)
        e = ev + ed
        zb = ((b[:, 2] - zlo) / (zhi - zlo) * bins_z).floor().clamp(0, bins_z - 1).long()     # 0 프레임 몸 높이 기준 띠 (눌리면 위 띠가 빈다)
        rg = (b[:, 0] > 0).long() * 2 + (b[:, 1] > 0).long()
        MAP[k].index_put_((zb, rg), e, accumulate=True)
        SP[k]["ev"] += float(ev.sum()) / len(frames)
        SP[k]["ed"] += float(ed.sum()) / len(frames)
        SP[k]["cosw"] += float((e * cs).sum() / e.sum().clamp_min(1e-30)) / len(frames)
for k in MAP:
    MAP[k] /= len(frames)
# (c) 셀 안 상대 운동 에너지 시계열 (전체 프레임)
def relke(x, v):
    c = torch.floor(x / NH).long()
    c = c - c.min(0).values
    key = (c[:, 0] * 4096 + c[:, 1]) * 4096 + c[:, 2]
    u, inv = torch.unique(key, return_inverse=True)
    mm = torch.zeros(u.numel(), device=dev, dtype=torch.float64).index_add_(0, inv, m)
    mv = torch.zeros(u.numel(), 3, device=dev, dtype=torch.float64).index_add_(0, inv, m[:, None] * v)
    vb = (mv / mm[:, None])[inv]
    return float(0.5 * (m * ((v - vb) ** 2).sum(-1)).sum()), float(0.5 * (m * (v ** 2).sum(-1)).sum())


RK = {"ours": [], "gt": []}
for f in range(1, L + 1):
    RK["ours"].append(relke(R["pred"][f - 1].to(dev).double(), R["v"][f - 1].to(dev).double()))
    RK["gt"].append(relke(d["x"][f].to(dev).double(), d["v"][f].to(dev).double()))
spec = {}
for k in RK:
    s_ = torch.tensor([q[0] for q in RK[k]], dtype=torch.float64)
    a_ = torch.fft.rfft(s_ - s_.mean()).abs()
    fr = torch.fft.rfftfreq(len(s_), d=h)
    top = torch.argsort(a_, descending=True)[:3]
    spec[k] = dict(top_hz=[round(float(fr[i]), 1) for i in top], frac_above10hz=float((a_[fr > 10] ** 2).sum() / (a_[1:] ** 2).sum().clamp_min(1e-30)))
# (d) 갇힌 입자와 셀 바뀜
def cell_id(x):
    return torch.floor(x / NH).long()


fz = frames[len(frames) // 2]
Fo = R["F"][fz - 1].to(dev).double().reshape(-1, 3, 3)
Fg = GF[fz].to(dev).double().reshape(-1, 3, 3)
eo = sum(parts(Fo)[:2])
eg = sum(parts(Fg)[:2])
ex = eo - eg
thr = torch.quantile(ex.float(), 0.9).double()
trap = ex > thr
ch = torch.zeros_like(ex)
for f in range(max(1, fz - 5), fz):
    ch += (cell_id(R["pred"][f].to(dev).double()) != cell_id(R["pred"][f - 1].to(dev).double())).any(-1).double()
out = dict(src=rp, c0=c0, fmin=fmin, frames=frames, map={k: MAP[k].tolist() for k in MAP}, parts=SP, relke=RK, spec=spec,
           trap=dict(frame=fz, excess_share_top10=float(ex[trap].sum() / ex.clamp_min(0).sum().clamp_min(1e-30)),
                     cellchg_trap=float(ch[trap].mean()), cellchg_rest=float(ch[~trap].mean()),
                     trap_height_band=[int(q) for q in torch.bincount(((body(R["pred"][fz - 1].to(dev).double())[0][:, 2] - zlo) / (zhi - zlo) * 8).floor().clamp(0, 7).long()[trap], minlength=8).tolist()],
                     trap_J_mean=float(torch.linalg.det(Fo)[trap].mean()), rest_J_mean=float(torch.linalg.det(Fo)[~trap].mean())))
json.dump(out, open(os.path.join(os.path.dirname(rp), "local.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: out[k] for k in ("fmin", "parts", "spec", "trap")}, ensure_ascii=False), flush=True)
